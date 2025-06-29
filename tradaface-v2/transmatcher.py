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
import torch.nn.functional as F


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
        score_embed = torch.randn(seq_len, seq_len)
        score_embed = score_embed + score_embed.t()
        self.score_embed = nn.Parameter(score_embed.view(1, 1, seq_len, seq_len))
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
        self.bn2 = nn.BatchNorm1d(dim_feedforward)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(dim_feedforward, 1)
        self.bn3 = nn.BatchNorm1d(1)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        r"""Pass the inputs through the decoder layer in turn.

        Args:
            tgt: [q, h, w, d], where q is the query length, d is d_model, and (h, w) is feature map size
            memory: [k, h, w, d], where k is the memory length
        """

        q, h, w, d = tgt.size()
        # Check that spatial dimensions match seq_len, but allow variable channel size
        assert(h * w == self.seq_len)
        k, h, w, d = memory.size()
        assert(h * w == self.seq_len)

        tgt = tgt.view(q, -1, d)
        memory = memory.view(k, -1, d)
        query = self.fc1(tgt)
        key = self.fc1(memory)
        score = einsum('q t d, k s d -> q k s t', query, key) * self.score_embed.sigmoid()
        score = score.reshape(q * k, self.seq_len, self.seq_len)
        score = torch.cat((score.max(dim=1)[0], score.max(dim=2)[0]), dim=-1)
        score = score.view(-1, 1, self.seq_len)
        score = self.bn1(score).view(-1, self.seq_len)

        score = self.fc2(score)
        score = self.bn2(score)
        score = self.relu(score)
        score = self.fc3(score)
        score = score.view(-1, 2).sum(dim=-1, keepdim=True)
        score = self.bn3(score)
        score = score.view(q, k)
        return score


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

    def __init__(self, decoder_layers, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        # Use the provided decoder layers instead of copying
        if isinstance(decoder_layers, nn.ModuleList):
            self.layers = decoder_layers
        elif isinstance(decoder_layers, list):
            self.layers = ModuleList(decoder_layers)
        else:
            # Fallback: copy the single decoder layer
            self.layers = ModuleList([copy.deepcopy(decoder_layers) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        r"""Pass the inputs through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).

        Shape:
            tgt: [q, h, w, d*n], where q is the query length, d is d_model, n is num_layers, and (h, w) is feature map size
            memory: [k, h, w, d*n], where k is the memory length
        """

        # Handle chunking when d_model is not perfectly divisible by num_layers
        total_channels = tgt.size(-1)
        channels_per_layer = total_channels // self.num_layers
        remainder = total_channels % self.num_layers
        
        # Create chunks with proper distribution of remainder
        tgt_chunks = []
        memory_chunks = []
        start_idx = 0
        
        for i in range(self.num_layers):
            # Add one extra channel to early layers if there's a remainder
            extra_channel = 1 if i < remainder else 0
            chunk_size = channels_per_layer + extra_channel
            
            tgt_chunk = tgt[..., start_idx:start_idx + chunk_size]
            memory_chunk = memory[..., start_idx:start_idx + chunk_size]
            
            tgt_chunks.append(tgt_chunk)
            memory_chunks.append(memory_chunk)
            start_idx += chunk_size
        
        for i, mod in enumerate(self.layers):
            if i == 0:
                score = mod(tgt_chunks[i], memory_chunks[i])
            else:
                score = score + mod(tgt_chunks[i], memory_chunks[i])

        if self.norm is not None:
            q, k = score.size()
            score = score.view(-1, 1)
            score = self.norm(score)
            score = score.view(q, k)

        return score


class TransMatcher(nn.Module):

    def __init__(self, seq_len, d_model=512, num_decoder_layers=3, dim_feedforward=2048):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers

        # Calculate chunk sizes for each layer
        channels_per_layer = d_model // num_decoder_layers
        remainder = d_model % num_decoder_layers
        
        self.chunk_sizes = []
        for i in range(num_decoder_layers):
            extra_channel = 1 if i < remainder else 0
            chunk_size = channels_per_layer + extra_channel
            self.chunk_sizes.append(chunk_size)
        
        print(f"TransMatcher: d_model={d_model}, num_layers={num_decoder_layers}")
        print(f"Chunk sizes: {self.chunk_sizes}")

        # Create separate decoder layers for each chunk size
        self.decoder_layers = nn.ModuleList()
        for chunk_size in self.chunk_sizes:
            decoder_layer = TransformerDecoderLayer(seq_len, chunk_size, dim_feedforward)
            self.decoder_layers.append(decoder_layer)
        
        decoder_norm = nn.BatchNorm1d(1)
        # Pass the list of decoder layers to TransformerDecoder
        self.decoder = TransformerDecoder(self.decoder_layers, num_decoder_layers, decoder_norm)
        self.memory = None
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def make_kernel(self, features):
        # Convert from [batch, channels, height, width] to [batch, height, width, channels]
        # This is the standard format expected by the transformer layers
        if features.dim() == 4:
            features = features.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        
        self.memory = features

    def forward(self, features):
        # Only permute if memory is not set (first call) or if features are in wrong format
        if self.memory is None or features.dim() == 4:
            # Convert from [batch, channels, height, width] to [batch, height, width, channels]
            # This is the standard format expected by the transformer layers
            if features.dim() == 4:
                features = features.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        
        # FIXED: Use original argument order - memory first, features second
        score = self.decoder(self.memory, features)
        return score

    def match(self, gallery, query):
        self.make_kernel(gallery)
        return self.forward(query)

    def match_pairs(self, probe_features, gallery_features):
        """
        Match probe-gallery pairs directly during validation.
        This is a memory-efficient version specifically for validation.
        
        Args:
            probe_features: Features from probe images [batch_size, num_features, height, width]
            gallery_features: Features from gallery images [batch_size, num_features, height, width]
            
        Returns:
            Similarity scores between probe and gallery pairs [batch_size]
        """
        batch_size = probe_features.size(0)
        device = probe_features.device
        dtype = probe_features.dtype
        
        # Process in smaller chunks for memory efficiency
        chunk_size = 32  # Adjust based on available memory
        pair_scores = torch.zeros(batch_size, device=device, dtype=dtype)
        
        with torch.no_grad():
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                
                # Get current chunk
                probe_chunk = probe_features[start:end]
                gallery_chunk = gallery_features[start:end]
                
                # Normalize features
                probe_chunk = F.normalize(probe_chunk, p=2, dim=1)
                gallery_chunk = F.normalize(gallery_chunk, p=2, dim=1)
                
                # Convert from [batch, channels, height, width] to [batch, height, width, channels]
                probe_chunk = probe_chunk.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
                gallery_chunk = gallery_chunk.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
                
                # Set memory directly (already in correct format)
                self.memory = gallery_chunk
                
                # Call forward with already-permuted features
                scores = self.decoder(self.memory, probe_chunk)
                
                # Extract diagonal scores (matching pairs)
                for i in range(end - start):
                    pair_scores[start + i] = scores[i, i]
                
                # Clear memory
                torch.cuda.empty_cache()
        
        return pair_scores

    def reset_running_stats(self):
        """Reset running statistics for batch normalization layers"""
        for layer in self.decoder.layers:
            layer.bn1.reset_running_stats()
            layer.bn2.reset_running_stats()
            layer.bn3.reset_running_stats()
        if self.decoder.norm is not None:
            self.decoder.norm.reset_running_stats()

    def clone(self):
        """Create a fresh clone of this module"""
        device = next(self.parameters()).device
        
        clone = TransMatcher(
            self.seq_len, 
            self.d_model, 
            self.num_decoder_layers, 
            self.dim_feedforward
        ).to(device)
        
        clone.load_state_dict(self.state_dict())
        return clone


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
