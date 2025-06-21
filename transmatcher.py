"""Class for the Transformer based image matcher
    Shengcai Liao and Ling Shao, "Transformer-Based Deep Image Matching for Generalizable Person Re-identification." 
    In arXiv preprint, arXiv:2105.14432, 2021.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.0
        May 25, 2021
    """

import copy
import torch
from torch import nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Module
from torch.nn.modules.container import ModuleList
from torch import einsum
from torch.nn import functional as F


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of feature matching and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
        >>> memory = torch.rand(10, 24, 8, 512)
        >>> tgt = torch.rand(20, 24, 8, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, seq_len, d_model=512, dim_feedforward=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Verify seq_len is correct and calculate spatial dimensions
        h = w = int(seq_len ** 0.5)
        assert h * w == seq_len, f"seq_len {seq_len} must be a perfect square (got h={h}, w={w})"
        self.h = h
        self.w = w
        
        # Initialize score embedding with correct spatial dimensions
        score_embed = torch.randn(h, w, h, w)  # [h, w, h, w]
        score_embed = score_embed + score_embed.permute(2, 3, 0, 1)  # Make symmetric
        self.score_embed = nn.Parameter(score_embed)  # [h, w, h, w]
        
        # Initialize layers with proper dimensions
        self.fc1 = nn.Linear(d_model, d_model)
        
        # Calculate the correct size for batch norm layers
        # The combined scores will have shape [batch, 2*seq_len]
        bn_size = 2 * seq_len
        
        # Initialize batch norm layers with correct size and DDP-friendly settings
        if torch.distributed.is_initialized():
            self.bn1 = nn.SyncBatchNorm(bn_size, momentum=0.1, track_running_stats=True)
            self.bn2 = nn.SyncBatchNorm(dim_feedforward, momentum=0.1, track_running_stats=True)
            self.bn3 = nn.SyncBatchNorm(1, momentum=0.1, track_running_stats=True)
        else:
            self.bn1 = nn.BatchNorm1d(bn_size, momentum=0.1, track_running_stats=True)
            self.bn2 = nn.BatchNorm1d(dim_feedforward, momentum=0.1, track_running_stats=True)
            self.bn3 = nn.BatchNorm1d(1, momentum=0.1, track_running_stats=True)
        
        self.fc2 = nn.Linear(bn_size, dim_feedforward)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(dim_feedforward, 1)
        
        # Add gradient checkpointing
        self.use_checkpointing = True
        
        # Initialize batch norm running statistics with correct dimensions
        self._initialize_bn_stats()
        
        # Print debug info
        print(f"Initialized TransformerDecoderLayer with seq_len={seq_len}, bn_size={bn_size}")
        print(f"Batch norm layer sizes: bn1={self.bn1.num_features}, bn2={self.bn2.num_features}, bn3={self.bn3.num_features}")

    def _initialize_bn_stats(self):
        """Initialize batch norm running statistics with proper dimensions"""
        # Initialize bn1 with correct sequence length
        bn_size = 2 * self.seq_len
        
        # Use register_buffer for running stats
        self.bn1.register_buffer('running_mean', torch.zeros(bn_size, device=self.bn1.running_mean.device))
        self.bn1.register_buffer('running_var', torch.ones(bn_size, device=self.bn1.running_var.device))
        self.bn1.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=self.bn1.num_batches_tracked.device))
        
        # Initialize bn2
        self.bn2.register_buffer('running_mean', torch.zeros(self.bn2.num_features, device=self.bn2.running_mean.device))
        self.bn2.register_buffer('running_var', torch.ones(self.bn2.num_features, device=self.bn2.running_var.device))
        self.bn2.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=self.bn2.num_batches_tracked.device))
        
        # Initialize bn3
        self.bn3.register_buffer('running_mean', torch.zeros(self.bn3.num_features, device=self.bn3.running_mean.device))
        self.bn3.register_buffer('running_var', torch.ones(self.bn3.num_features, device=self.bn3.running_var.device))
        self.bn3.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=self.bn3.num_batches_tracked.device))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        """Override to handle loading of batch norm statistics"""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                    missing_keys, unexpected_keys, error_msgs)
        
        # Handle batch norm statistics loading
        for name in ['bn1', 'bn2', 'bn3']:
            bn = getattr(self, name)
            if prefix + name + '.running_mean' in state_dict:
                bn.running_mean.copy_(state_dict[prefix + name + '.running_mean'])
            if prefix + name + '.running_var' in state_dict:
                bn.running_var.copy_(state_dict[prefix + name + '.running_var'])
            if prefix + name + '.num_batches_tracked' in state_dict:
                bn.num_batches_tracked.copy_(state_dict[prefix + name + '.num_batches_tracked'])

    def _compute_attention_chunk(self, query, key, chunk_size=32):
        """Compute attention scores in chunks to save memory"""
        batch_size = query.size(0)
        scores = []
        
        # Reshape query and key to spatial dimensions
        query = query.view(batch_size, self.h, self.w, -1)  # [batch, h, w, d]
        key = key.view(batch_size, self.h, self.w, -1)  # [batch, h, w, d]
        
        for i in range(0, batch_size, chunk_size):
            end_i = min(i + chunk_size, batch_size)
            query_chunk = query[i:end_i]  # [chunk_size, h, w, d]
            
            # Compute attention for this chunk
            chunk_scores = []
            for j in range(0, batch_size, chunk_size):
                end_j = min(j + chunk_size, batch_size)
                key_chunk = key[j:end_j]  # [chunk_size, h, w, d]
                
                # Reshape for attention computation
                q = query_chunk.view(end_i-i, self.seq_len, -1)  # [chunk_size, h*w, d]
                k = key_chunk.view(end_j-j, self.seq_len, -1)  # [chunk_size, h*w, d]
                
                # Compute attention scores for this chunk pair
                attn = torch.bmm(q, k.transpose(1, 2))  # [chunk_size, h*w, h*w]
                
                # Apply spatial score embedding
                attn = attn.view(end_i-i, self.h, self.w, self.h, self.w)
                attn = attn * self.score_embed.unsqueeze(0).sigmoid()
                attn = attn.view(end_i-i, self.seq_len, self.seq_len)
                
                chunk_scores.append(attn)
            
            # Concatenate chunk scores
            scores.append(torch.cat(chunk_scores, dim=1))
        
        return torch.cat(scores, dim=0)

    def _forward_without_checkpointing(self, tgt, memory):
        """Forward pass without gradient checkpointing"""
        # Process through layers
        query = self.fc1(tgt)
        key = self.fc1(memory)
        print(f"DEBUG: After fc1 - query: {query.shape}, key: {key.shape}")
        
        # Normalize query and key
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
        
        # Compute attention scores in chunks
        attention = self._compute_attention_chunk(query, key)
        print(f"DEBUG: After attention - shape: {attention.shape}")
        
        # Get max scores along both dimensions
        max_score1 = attention.max(dim=1)[0]  # [batch, seq_len]
        max_score2 = attention.max(dim=2)[0]  # [batch, seq_len]
        print(f"DEBUG: After max - max_score1: {max_score1.shape}, max_score2: {max_score2.shape}")
        
        # Concatenate max scores
        combined_scores = torch.cat([max_score1, max_score2], dim=1)  # [batch, 2*seq_len]
        print(f"DEBUG: Combined scores shape: {combined_scores.shape}, bn1 expected: {self.bn1.num_features}")
        
        # Process through remaining layers
        x = self.bn1(combined_scores)
        print(f"DEBUG: After bn1 - shape: {x.shape}")
        x = self.fc2(x)
        print(f"DEBUG: After fc2 - shape: {x.shape}")
        x = self.bn2(x)
        print(f"DEBUG: After bn2 - shape: {x.shape}")
        x = self.relu(x)
        x = self.fc3(x)
        print(f"DEBUG: After fc3 - shape: {x.shape}")
        x = self.bn3(x)
        print(f"DEBUG: After bn3 - shape: {x.shape}")
        
        return x.squeeze(-1)  # [batch]

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """
        Forward pass for a single decoder layer with memory-efficient attention.
        """
        # Ensure inputs are on the same device and dtype as the model
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        tgt = tgt.to(device=device, dtype=dtype)
        memory = memory.to(device=device, dtype=dtype)
        
        # Get shapes and verify dimensions
        batch, h, w, d = tgt.size()
        actual_seq_len = h * w
        print(f"DEBUG: TransformerDecoderLayer input shapes - tgt: {tgt.shape}, memory: {memory.shape}")
        print(f"DEBUG: Expected seq_len: {self.seq_len}, actual: {actual_seq_len}")
        print(f"DEBUG: Expected d_model: {self.d_model}, actual: {d}")
        assert actual_seq_len == self.seq_len, f"Feature map size {h}x{w}={actual_seq_len} != seq_len {self.seq_len}"
        assert d == self.d_model, f"Feature dimension {d} != d_model {self.d_model}"
        
        # Reshape for processing
        tgt = tgt.reshape(batch, self.seq_len, self.d_model)  # [batch, seq_len, d]
        memory = memory.reshape(batch, self.seq_len, self.d_model)  # [batch, seq_len, d]
        print(f"DEBUG: After reshape - tgt: {tgt.shape}, memory: {memory.shape}")
        
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing for the entire forward pass
            return torch.utils.checkpoint.checkpoint(self._forward_without_checkpointing, tgt, memory)
        else:
            return self._forward_without_checkpointing(tgt, memory)


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
        >>> transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)
        >>> memory = torch.rand(10, 24, 8, 512)
        >>> tgt = torch.rand(20, 24, 8, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        """
        Forward pass for the decoder.
        
        Args:
            tgt: Query features [batch, h, w, d]
            memory: Gallery features [batch, h, w, d]
            
        Returns:
            Similarity scores [batch]
        """
        # Process through each layer sequentially
        score = None
        for i, mod in enumerate(self.layers):
            if i == 0:
                score = mod(tgt, memory)  # Process entire batch at once
            else:
                # Add scores from each layer
                score = score + mod(tgt, memory)  # Process entire batch at once

        if self.norm is not None:
            score = score.view(-1, 1)
            score = self.norm(score)
            score = score.view(-1)

        return score


class TransMatcher(nn.Module):

    def __init__(self, seq_len, d_model=512, num_decoder_layers=3, dim_feedforward=2048, num_classes=None, k_nearest=20):
        super().__init__()
        # Verify seq_len is correct
        h = w = int(seq_len ** 0.5)
        assert h * w == seq_len, f"seq_len {seq_len} must be a perfect square (got h={h}, w={w})"
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.k_nearest = k_nearest
        self.num_classes = num_classes
        self.class_embed = None
        if num_classes is not None:
            self.class_embed = nn.Parameter(torch.randn(num_classes, d_model, h, w, dtype=torch.float32) / d_model**0.5)
        
        # Add memory-efficient settings
        self.chunk_size = 32  # Adjust based on available GPU memory
        self.use_checkpointing = True
        
        # Initialize decoder with verified seq_len
        self.decoder_layer = TransformerDecoderLayer(seq_len, d_model, dim_feedforward)
        self.decoder_layer.use_checkpointing = self.use_checkpointing
        decoder_norm = nn.BatchNorm1d(1)
        self.decoder = TransformerDecoder(self.decoder_layer, num_decoder_layers, decoder_norm)
        self.memory = None
        self.class_neighbors = None
        self.reset_parameters()

    def to(self, *args, **kwargs):
        """Override to() to ensure all parameters and buffers are moved to the same device and dtype"""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None or dtype is not None:
            # Move all parameters and buffers to the specified device and dtype
            for module in self.modules():
                for param in module.parameters():
                    if device is not None:
                        param.data = param.data.to(device, non_blocking=non_blocking)
                    if dtype is not None:
                        param.data = param.data.to(dtype)
                for buffer in module.buffers():
                    if device is not None:
                        buffer.data = buffer.data.to(device, non_blocking=non_blocking)
                    if dtype is not None:
                        buffer.data = buffer.data.to(dtype)
            if self.class_embed is not None:
                if device is not None:
                    self.class_embed.data = self.class_embed.data.to(device, non_blocking=non_blocking)
                if dtype is not None:
                    self.class_embed.data = self.class_embed.data.to(dtype)
            if self.memory is not None:
                if device is not None:
                    self.memory = self.memory.to(device, non_blocking=non_blocking)
                if dtype is not None:
                    self.memory = self.memory.to(dtype)
        return super().to(*args, **kwargs)

    def reset_running_stats(self):
        # For future use: reset batchnorm stats if needed
        pass

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        if self.class_embed is not None:
            nn.init.normal_(self.class_embed, std=1. / self.d_model ** 0.5)
        self.class_neighbors = None

    def make_kernel(self, features):
        self.memory = features

    def compute_class_neighbors(self):
        """
        Compute k-nearest neighbor classes for each class based on class embeddings.
        This is similar to QAConv's and graph_sampler.py's logic.
        Returns a [num_classes, k_nearest] tensor of neighbor indices.
        """
        if self.class_embed is None:
            return None
        with torch.no_grad():
            num_classes = self.class_embed.size(0)
            device = self.class_embed.device
            # Flatten and normalize class embeddings
            class_embeds_flat = self.class_embed.view(num_classes, -1)
            class_embeds_flat = nn.functional.normalize(class_embeds_flat, p=2, dim=1)
            # Compute similarity matrix (cosine similarity)
            similarity_matrix = torch.mm(class_embeds_flat, class_embeds_flat.t())
            # For graph sampler, we want distances (1 - similarity)
            distance_matrix = 1.0 - similarity_matrix
            # Mask diagonal (self-similarity) to large value so it's not selected as neighbor
            distance_matrix += torch.eye(num_classes, device=device) * 1e6
            # Get top-k nearest (smallest distance) neighbors for each class
            _, indices = torch.topk(distance_matrix, k=self.k_nearest, largest=False, dim=1)
            self.class_neighbors = indices  # [num_classes, k_nearest]
            return self.class_neighbors

    def graph_sampler_simulate(self):
        """
        Simulate the graph sampler: for each class, return its k-nearest neighbor class indices.
        This can be used to guide batch sampling or for analysis.
        Returns a [num_classes, k_nearest] tensor of neighbor indices.
        """
        neighbors = self.compute_class_neighbors()
        # Optionally, return as a list of lists for easier use
        if neighbors is not None:
            return neighbors.cpu().tolist()
        else:
            return None

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor = None, labels: torch.Tensor = None):
        """
        Forward pass for TransMatcher with memory-efficient processing.
        
        Args:
            tgt: Query features [batch_q, h, w, d]
            memory: Gallery features [batch_g, h, w, d]
            labels: Optional labels for class-based matching
            
        Returns:
            Similarity scores between query and gallery features [batch_q, batch_g]
        """
        # Ensure inputs are on the same device and dtype as the model
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        tgt = tgt.to(device=device, dtype=dtype)
        if memory is not None:
            memory = memory.to(device=device, dtype=dtype)
        if labels is not None:
            labels = labels.to(device=device, dtype=dtype)

        # Handle NaNs in input tensors
        if torch.isnan(tgt).any():
            print("WARNING: Query features contain NaNs. Replacing with zeros.")
            tgt = torch.nan_to_num(tgt, nan=0.0)
            # Add small epsilon to prevent all zeros
            tgt = tgt + 1e-6
        if memory is not None and torch.isnan(memory).any():
            print("WARNING: Gallery features contain NaNs. Replacing with zeros.")
            memory = torch.nan_to_num(memory, nan=0.0)
            # Add small epsilon to prevent all zeros
            memory = memory + 1e-6

        # Normalize input features
        tgt = F.normalize(tgt.reshape(tgt.size(0), -1), p=2, dim=1).reshape_as(tgt)
        if memory is not None:
            memory = F.normalize(memory.reshape(memory.size(0), -1), p=2, dim=1).reshape_as(memory)

        if self.training and memory is None and self.class_embed is not None:
            # Class-based matching during training with chunked processing
            batch = tgt.size(0)
            h, w, d = tgt.size(1), tgt.size(2), tgt.size(3)
            class_embed = self.class_embed.permute(0, 2, 3, 1).contiguous()
            class_embed = F.normalize(class_embed.reshape(class_embed.size(0), -1), p=2, dim=1).reshape_as(class_embed)
            
            # Process in chunks to save memory
            scores = []
            for i in range(0, batch, self.chunk_size):
                end_i = min(i + self.chunk_size, batch)
                tgt_chunk = tgt[i:end_i]
                
                chunk_scores = []
                for j in range(0, self.num_classes, self.chunk_size):
                    end_j = min(j + self.chunk_size, self.num_classes)
                    class_chunk = class_embed[j:end_j]
                    
                    # Expand dimensions for matching
                    tgt_exp = tgt_chunk.unsqueeze(1).expand(end_i-i, end_j-j, h, w, d)
                    class_exp = class_chunk.unsqueeze(0).expand(end_i-i, end_j-j, h, w, d)
                    
                    # Flatten for decoder
                    tgt_flat = tgt_exp.reshape(-1, h, w, d)
                    class_flat = class_exp.reshape(-1, h, w, d)
                    
                    # Get scores for this chunk
                    chunk_score = self.decoder(tgt_flat, class_flat)
                    chunk_scores.append(chunk_score.view(end_i-i, end_j-j))
                
                # Combine chunk scores
                scores.append(torch.cat(chunk_scores, dim=1))
            
            return torch.cat(scores, dim=0)
            
        elif memory is not None:
            # Regular matching between query and gallery with chunked processing
            batch_q = tgt.size(0)
            batch_g = memory.size(0)
            
            if self.training:
                # Process in chunks during training
                scores = []
                for i in range(0, batch_q, self.chunk_size):
                    end_i = min(i + self.chunk_size, batch_q)
                    query_chunk = tgt[i:end_i]
                    
                    chunk_scores = []
                    for j in range(0, batch_g, self.chunk_size):
                        end_j = min(j + self.chunk_size, batch_g)
                        gallery_chunk = memory[j:end_j]
                        
                        # Expand dimensions for matching
                        query_exp = query_chunk.unsqueeze(1).expand(end_i-i, end_j-j, *tgt.size()[1:])
                        gallery_exp = gallery_chunk.unsqueeze(0).expand(end_i-i, end_j-j, *memory.size()[1:])
                        
                        # Flatten for decoder
                        query_flat = query_exp.reshape(-1, *tgt.size()[1:])
                        gallery_flat = gallery_exp.reshape(-1, *memory.size()[1:])
                        
                        # Get scores for this chunk
                        chunk_score = self.decoder(query_flat, gallery_flat)
                        chunk_scores.append(chunk_score.view(end_i-i, end_j-j))
                    
                    # Combine chunk scores
                    scores.append(torch.cat(chunk_scores, dim=1))
                
                return torch.cat(scores, dim=0)
            else:
                # For inference, use fast cosine similarity with chunked processing
                query_flat = tgt.reshape(batch_q, -1)
                gallery_flat = memory.reshape(batch_g, -1)
                
                # Normalize to unit vectors
                query_norm = F.normalize(query_flat, p=2, dim=1)
                gallery_norm = F.normalize(gallery_flat, p=2, dim=1)
                
                # Compute cosine similarity in chunks
                scores = []
                for i in range(0, batch_q, self.chunk_size):
                    end_i = min(i + self.chunk_size, batch_q)
                    query_chunk = query_norm[i:end_i]
                    
                    chunk_scores = []
                    for j in range(0, batch_g, self.chunk_size):
                        end_j = min(j + self.chunk_size, batch_g)
                        gallery_chunk = gallery_norm[j:end_j]
                        
                        # Compute similarity for this chunk
                        chunk_score = torch.matmul(query_chunk, gallery_chunk.transpose(0, 1))
                        chunk_scores.append(chunk_score)
                    
                    # Combine chunk scores
                    scores.append(torch.cat(chunk_scores, dim=1))
                
                return torch.cat(scores, dim=0)
        else:
            raise ValueError("TransMatcher: Either memory or class_embed must be provided for matching.")

    # Optionally, add a method to get class neighbors for a given class
    def get_class_neighbors(self, class_idx):
        if self.class_neighbors is None:
            self.compute_class_neighbors()
        return self.class_neighbors[class_idx] if self.class_neighbors is not None else None

    # Optionally, add a method to update class embeddings externally
    def update_class_embed(self, new_embed):
        assert new_embed.shape == self.class_embed.shape
        with torch.no_grad():
            self.class_embed.copy_(new_embed)


if __name__ == "__main__":
    import time
    model = TransMatcher(24*8, 512, 3).eval()
    gallery = torch.rand((32, 24, 8, 512*3))
    probe = torch.rand((16, 24, 8, 512*3))

    start = time.time()
    model.make_kernel(gallery)
    out = model(probe)
    print(out.size())
    end = time.time()
    print('Time: %.3f seconds.' % (end - start))

    start = time.time()
    model.make_kernel(probe)
    out2 = model(gallery)
    print(out2.size())
    end = time.time()
    print('Time: %.3f seconds.' % (end - start))
    out2 = out2.t()
    print((out2 == out).all())
    print((out2 - out).abs().mean())
    print(out[:4, :4])
    print(out2[:4, :4])
