import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
from torch.nn import CrossEntropyLoss
import evaluate_utils
import head
import net
import numpy as np
import utils
import pairwise_matching_loss
import pytorch_lightning as pl 
import torch.nn.functional as F
import time
import traceback
import os
import wandb


class Trainer(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        self.automatic_optimization = False

        # --- LOSS WEIGHTS ---
        # Set these values to control the loss contribution
        self.adaface_loss_weight = 0.5
        self.transmatcher_loss_weight = 0.5
        # --- END LOSS WEIGHTS ---

        self.class_num = utils.get_num_class(self.hparams)

        # Initialize weights with better defaults from QAConv
        self.warmup_epochs = 5

        # Build the backbone model (which now includes TransMatcher)
        backbone = net.build_model(model_name=self.hparams.arch)
        
        # The TransMatcher is now integrated into the backbone
        # We can access it directly from the backbone
        self.model = backbone
        self.pairwise_matching_loss = pairwise_matching_loss.PairwiseMatchingLoss(self.model.transmatcher)

        self.head = head.build_head(head_type=self.hparams.head,
                                  embedding_size=512,
                                  class_num=self.class_num,
                                  m=self.hparams.m,
                                  h=self.hparams.h,
                                  t_alpha=self.hparams.t_alpha,
                                  s=self.hparams.s)

        self.cross_entropy_loss = CrossEntropyLoss()

        if self.hparams.start_from_model_statedict:
            ckpt = torch.load(self.hparams.start_from_model_statedict)
            self.model.load_state_dict({key.replace('model.', ''):val
                                      for key,val in ckpt['state_dict'].items() if 'model.' in key})

    def on_train_epoch_start(self):
        # Loss weights (equal for AdaFace and TransMatcher)
        self.adaface_loss_weight = 0.5
        self.transmatcher_loss_weight = 0.5

    def get_current_lr(self):
        scheduler = None
        if scheduler is None:
            try:
                # pytorch lightning >= 1.8
                scheduler = self.trainer.lr_scheduler_configs[0].scheduler
            except:
                pass

        if scheduler is None:
            # pytorch lightning <=1.7
            try:
                scheduler = self.trainer.lr_schedulers[0]['scheduler']
            except:
                pass

        if scheduler is None:
            raise ValueError('lr calculation not successful')

        if isinstance(scheduler, lr_scheduler._LRScheduler):
            lr = scheduler.get_last_lr()[0]
        else:
            lr = scheduler.get_epoch_values(self.current_epoch)[0]
        return lr

    def forward(self, images, labels):
        # Forward pass through model (includes TransMatcher)
        embeddings, norms, feature_map, transmatcher = self.model(images, return_feature_map=True)
        
        # Update the matcher in pairwise loss
        self.pairwise_matching_loss.matcher = transmatcher
        
        # AdaFace head
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index
        
        return cos_thetas, norms, embeddings, labels, feature_map

    def training_step(self, batch, batch_idx):
        img, label = batch
        
        # Debug: Check input data for NaNs
        if torch.isnan(img).any():
            print(f"[TRAIN_STEP] CRITICAL: Input images contain NaNs at batch {batch_idx}!")
            print(f"  - Image stats: min={img.min().item():.6f}, max={img.max().item():.6f}, mean={img.mean().item():.6f}")
            img = torch.nan_to_num(img, nan=0.0)
        
        if torch.isnan(label.float()).any():
            print(f"[TRAIN_STEP] CRITICAL: Labels contain NaNs at batch {batch_idx}!")
            label = torch.nan_to_num(label.float(), nan=0.0).long()
        
        device = img.device
        
        # Forward pass through the model to get embeddings, norms, feature_maps, and transmatcher
        try:
            embedding, norm, feature_maps, transmatcher = self.model(img, return_feature_map=True)
        except Exception as e:
            print(f"[TRAIN_STEP] ERROR in model forward pass: {e}")
            raise

        # Debug: Check model outputs for NaNs
        if torch.isnan(embedding).any():
            print(f"[TRAIN_STEP] CRITICAL: Embeddings contain NaNs after model forward pass!")
            print(f"  - Embeddings shape: {embedding.shape}")
            print(f"  - Embeddings stats: min={embedding.min().item():.6f}, max={embedding.max().item():.6f}, mean={embedding.mean().item():.6f}")
            print(f"  - Input image stats: min={img.min().item():.6f}, max={img.max().item():.6f}, mean={img.mean().item():.6f}")
        
        if torch.isnan(norm).any():
            print(f"[TRAIN_STEP] CRITICAL: Norms contain NaNs after model forward pass!")
            print(f"  - Norms shape: {norm.shape}")
            print(f"  - Norms stats: min={norm.min().item():.6f}, max={norm.max().item():.6f}, mean={norm.mean().item():.6f}")

        if torch.isnan(feature_maps).any():
            print(f"[TRAIN_STEP] CRITICAL: Feature maps contain NaNs after model forward pass!")
            print(f"  - Feature maps shape: {feature_maps.shape}")
            print(f"  - Feature maps stats: min={feature_maps.min().item():.6f}, max={feature_maps.max().item():.6f}, mean={feature_maps.mean().item():.6f}")

        # Only check for NaN in feature maps occasionally to save performance
        if batch_idx % 50 == 0 and torch.isnan(feature_maps).any():
            print(f"WARNING: Feature maps contain NaNs. Replacing with zeros.")
            feature_maps = torch.nan_to_num(feature_maps, nan=0.0)
            # Add small epsilon to non-zero values to prevent division issues
            feature_maps = feature_maps + 1e-8 * (feature_maps.abs() > 0).float()
            
        # Normalize feature maps for TransMatcher
        fm_flat = feature_maps.view(feature_maps.size(0), -1)
        norms_fm = torch.norm(fm_flat, p=2, dim=1)
        
        # Prevent division by zero
        norms_fm = torch.clamp(norms_fm, min=1e-8)
        
        if ((norms_fm < 0.99) | (norms_fm > 1.01)).any():
            fm_flat = torch.nn.functional.normalize(fm_flat, p=2, dim=1)
            feature_maps = fm_flat.view_as(feature_maps)

        # Only check for NaN in embeddings occasionally to save performance
        if batch_idx % 50 == 0 and torch.isnan(embedding).any():
            print(f"WARNING: Embeddings contain NaNs. Replacing with zeros.")
            embedding = torch.nan_to_num(embedding, nan=0.0)
            # Add small epsilon to non-zero values to prevent division issues
            embedding = embedding + 1e-8 * (embedding.abs() > 0).float()
            
        # Normalize embeddings
        emb_norms = torch.norm(embedding, 2, 1, True).clamp(min=1e-6)
        embedding = torch.div(embedding, emb_norms)
        
        # Only check for NaN occasionally after normalization
        if batch_idx % 50 == 0:
            if torch.isnan(embedding).any():
                embedding = torch.nan_to_num(embedding, nan=0.0)
            if torch.isnan(emb_norms).any():
                emb_norms = torch.nan_to_num(emb_norms, nan=1.0)

        # AdaFace head
        cos_thetas = self.head(embedding, norm, label)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            label[bad_grad.squeeze(-1)] = -100 # ignore_index
        
        # Debug: Check head outputs for NaNs
        if torch.isnan(cos_thetas).any():
            print(f"[TRAIN_STEP] CRITICAL: Cos thetas contain NaNs after head!")
            print(f"  - Cos thetas shape: {cos_thetas.shape}")
            print(f"  - Cos thetas stats: min={cos_thetas.min().item():.6f}, max={cos_thetas.max().item():.6f}, mean={cos_thetas.mean().item():.6f}")
            print(f"  - Embeddings stats: min={embedding.min().item():.6f}, max={embedding.max().item():.6f}, mean={embedding.mean().item():.6f}")
            print(f"  - Norms stats: min={norm.min().item():.6f}, max={norm.max().item():.6f}, mean={norm.mean().item():.6f}")
        
        # AdaFace loss
        adaface_loss = self.cross_entropy_loss(cos_thetas, label)
        
        # Debug: Check loss for NaNs
        if torch.isnan(adaface_loss):
            print(f"[TRAIN_STEP] CRITICAL: AdaFace loss is NaN!")
            print(f"  - Cos thetas stats: min={cos_thetas.min().item():.6f}, max={cos_thetas.max().item():.6f}, mean={cos_thetas.mean().item():.6f}")
            print(f"  - Labels: {label}")
            print(f"  - Cos thetas has NaN: {torch.isnan(cos_thetas).any()}")
            print(f"  - Labels has NaN: {torch.isnan(label.float()).any()}")
            adaface_loss = torch.tensor(0.0, device=device)
        
        # TransMatcher loss
        transmatcher_loss = torch.tensor(0.0, device=device)
        transmatcher_acc = torch.tensor(0.0, device=device)
        
        if self.pairwise_matching_loss is not None:
            try:
                # Update the matcher in pairwise loss
                self.pairwise_matching_loss.matcher = transmatcher
                
                # Debug: Check feature maps before TransMatcher
                print(f"[TRAIN_STEP] Feature maps shape: {feature_maps.shape}")
                print(f"[TRAIN_STEP] Feature maps stats: min={feature_maps.min().item():.6f}, max={feature_maps.max().item():.6f}, mean={feature_maps.mean().item():.6f}")
                print(f"[TRAIN_STEP] Labels: {label}")
                
                # Compute TransMatcher loss
                transmatcher_loss, transmatcher_acc = self.pairwise_matching_loss(feature_maps, label)
                
                # Check for NaN in TransMatcher loss
                if torch.isnan(transmatcher_loss).any():
                    print(f"[TRAIN_STEP] CRITICAL: TransMatcher loss is NaN!")
                    transmatcher_loss = torch.tensor(0.0, device=device)
                
                if torch.isnan(transmatcher_acc).any():
                    print(f"[TRAIN_STEP] CRITICAL: TransMatcher accuracy is NaN!")
                    transmatcher_acc = torch.tensor(0.0, device=device)
                    
            except Exception as e:
                print(f"[TRAIN_STEP] ERROR in TransMatcher loss computation: {e}")
                import traceback
                traceback.print_exc()
                transmatcher_loss = torch.tensor(0.0, device=device)
                transmatcher_acc = torch.tensor(0.0, device=device)
        
        # Combined loss (weighted)
        total_loss = adaface_loss + 0.5 * transmatcher_loss.mean()
        
        # Log metrics
        lr = self.get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('adaface_loss', adaface_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('transmatcher_loss', transmatcher_loss.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('transmatcher_acc', transmatcher_acc.mean(), on_step=True, on_epoch=True, logger=True)
        
        # Log to wandb
        wandb.log({
            "adaface_loss": adaface_loss.item(),
            "transmatcher_loss": transmatcher_loss.mean().item(),
            "transmatcher_acc": transmatcher_acc.mean().item(),
            "total_loss": total_loss.item(),
            "learning_rate": lr,
        })

        # Manual backward pass for manual optimization
        self.manual_backward(total_loss)
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
        
        # Aggressive memory cleanup
        del embedding, norm, feature_maps, cos_thetas, adaface_loss, transmatcher_loss, transmatcher
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        return total_loss

    def training_epoch_end(self, outputs):
        return None

    def on_after_backward(self):
        """Called after backward pass to handle NaN gradients."""
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Check for NaN gradients and handle them silently
        has_nan_grad = False
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad.data = torch.zeros_like(param.grad.data)
                has_nan_grad = True
        
        if has_nan_grad:
            print("[TRAIN_STEP] WARNING: NaN gradients detected and zeroed out")

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """Called before optimizer step to handle learning rate scheduling."""
        # Step the learning rate scheduler on the last batch of each epoch
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

    def validation_step(self, batch, batch_idx):
        images, labels, dataname, image_index = batch
        device = images.device

        # Only check for NaN in images occasionally to save performance
        if batch_idx % 50 == 0 and torch.isnan(images).any():
            images = torch.nan_to_num(images, nan=0.0)

        with torch.no_grad():
            embedding, norm, feature_maps, _ = self.model(images, return_feature_map=True)
            
            # Only check for NaN occasionally to save performance
            if batch_idx % 50 == 0:
                if torch.isnan(embedding).any():
                    embedding = torch.nan_to_num(embedding, nan=0.0)
                
                if torch.isnan(feature_maps).any():
                    feature_maps = torch.nan_to_num(feature_maps, nan=0.0)
                    # Force normalization again
                    fm_flat = feature_maps.view(feature_maps.size(0), -1)
                    feature_maps = torch.nn.functional.normalize(fm_flat, p=2, dim=1).view_as(feature_maps)
                
                # Final check for NaNs in norms
                if torch.isnan(norm).any():
                    norm = torch.nan_to_num(norm, nan=1.0)

        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            return {
                'adaface_output': embedding.to('cpu'),
                'norm': norm.to('cpu'),
                'qaconv_output': feature_maps.to('cpu'),  # Keep standard format for compatibility
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            return {
                'adaface_output': embedding,
                'norm': norm,
                'qaconv_output': feature_maps,  # Keep standard format for compatibility
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }

    def validation_epoch_end(self, outputs):
        # Group outputs by dataname to process one validation set at a time
        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        val_logs = {}
        
        for dataname_idx in outputs[0]['dataname'].unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            
            # Filter outputs for this dataset
            list_of_outputs = [out for out in outputs if out['dataname'] == dataname_idx]
            
            # Concatenate all tensors for the current dataset
            all_embeds = torch.cat([out['adaface_output'] for out in list_of_outputs], axis=0).numpy()
            all_norms = torch.cat([out['norm'] for out in list_of_outputs], axis=0).numpy()
            all_feature_maps = torch.cat([out['qaconv_output'] for out in list_of_outputs], axis=0)
            all_labels = torch.cat([out['target'] for out in list_of_outputs], axis=0).numpy().astype(bool)

            # Check for NaN in AdaFace embeddings (only occasionally to save performance)
            if np.isnan(all_embeds).any():
                all_embeds = np.nan_to_num(all_embeds, nan=0.0)
            
            if np.isnan(all_norms).any():
                all_norms = np.nan_to_num(all_norms, nan=1.0)

            # --- Evaluate AdaFace ---
            # Reconstruct the flat embedding array that evaluate() expects
            num_pairs = all_embeds.shape[0]
            adaface_embeddings = np.concatenate([all_embeds, all_embeds]).reshape(num_pairs * 2, -1)
            
            # Final check for NaN in AdaFace embeddings before evaluation
            if np.isnan(adaface_embeddings).any():
                adaface_embeddings = np.nan_to_num(adaface_embeddings, nan=0.0)
            
            tpr, fpr, accuracy, _ = evaluate_utils.evaluate(adaface_embeddings, all_labels, nrof_folds=10)
            adaface_acc = accuracy.mean()
            val_logs[f'{dataname}_adaface_acc'] = adaface_acc

            # --- Evaluate TransMatcher ---
            # For TransMatcher evaluation, we need to compute similarity scores
            # Convert feature maps to the format expected by TransMatcher
            feature_maps_perm = all_feature_maps.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
            
            # Check for NaN in feature maps before TransMatcher processing (only occasionally)
            if torch.isnan(feature_maps_perm).any():
                feature_maps_perm = torch.nan_to_num(feature_maps_perm, nan=0.0)
                # Force normalization again
                fm_flat = feature_maps_perm.view(feature_maps_perm.size(0), -1)
                feature_maps_perm = torch.nn.functional.normalize(fm_flat, p=2, dim=1).view_as(feature_maps_perm)
            
            # Compute TransMatcher scores for positive and negative pairs
            device = feature_maps_perm.device
            num_pairs = len(feature_maps_perm) // 2
            
            # Get gallery and query features
            gallery_features = feature_maps_perm[0::2]  # Even indices
            query_features = feature_maps_perm[1::2]    # Odd indices
            
            # Compute positive pair scores (same identity)
            positive_scores = self.model.transmatcher.match_pairs(query_features, gallery_features)
            
            # Check for NaN in positive scores (only occasionally)
            if torch.isnan(positive_scores).any():
                positive_scores = torch.nan_to_num(positive_scores, nan=0.0)
            
            # Compute negative pair scores (different identity) - use shifted indices
            negative_scores = torch.zeros(num_pairs, device=device)
            for i in range(0, num_pairs, 32):  # Process in batches
                end_idx = min(i + 32, num_pairs)
                batch_size = end_idx - i
                
                # Shift indices to get different identities
                shifted_indices = (torch.arange(batch_size, device=device) + num_pairs // 2) % num_pairs
                shifted_queries = query_features[shifted_indices]
                batch_galleries = gallery_features[i:end_idx]
                
                for j in range(batch_size):
                    score = self.model.transmatcher(batch_galleries[j:j+1], shifted_queries[j:j+1])
                    negative_scores[i + j] = score.view(-1)[0]
            
            # Check for NaN in negative scores (only occasionally)
            if torch.isnan(negative_scores).any():
                negative_scores = torch.nan_to_num(negative_scores, nan=0.0)
            
            # Combine scores and convert to distances
            all_scores = torch.cat([positive_scores, negative_scores])
            
            # Final check for NaN in all scores (only occasionally)
            if torch.isnan(all_scores).any():
                all_scores = torch.nan_to_num(all_scores, nan=0.0)
            
            distances = -all_scores.cpu().numpy()  # Convert similarity to distance
            
            # Check for NaN in distances after numpy conversion (only occasionally)
            if np.isnan(distances).any():
                distances = np.nan_to_num(distances, nan=0.0)
            
            # Evaluate TransMatcher
            tpr, fpr, accuracy, _ = evaluate_utils.evaluate(distances.reshape(-1, 1), all_labels, nrof_folds=10)
            trans_acc = accuracy.mean()
            val_logs[f'{dataname}_trans_acc'] = trans_acc

            val_logs[f'{dataname}_combined_acc'] = (adaface_acc + trans_acc) / 2.0
            
            # Explicitly free memory
            del all_embeds, all_norms, all_feature_maps, feature_maps_perm, gallery_features, query_features
            del positive_scores, negative_scores, all_scores, distances
            import gc
            gc.collect()

        # average accuracies across datasets
        val_logs['val_adaface_acc'] = np.mean([
            val_logs[f'{dataname}_adaface_acc'] for dataname in dataname_to_idx.keys() 
            if f'{dataname}_adaface_acc' in val_logs
        ])
        
        val_logs['val_trans_acc'] = np.mean([
            val_logs[f'{dataname}_trans_acc'] for dataname in dataname_to_idx.keys() 
            if f'{dataname}_trans_acc' in val_logs
        ])
        
        val_logs['val_combined_acc'] = np.mean([
            val_logs[f'{dataname}_combined_acc'] for dataname in dataname_to_idx.keys() 
            if f'{dataname}_combined_acc' in val_logs
        ])
        
        # Add val_acc for ModelCheckpoint to monitor (use the combined accuracy)
        val_logs['val_acc'] = val_logs['val_combined_acc']

        # Log all metrics in one place
        self.log('val_adaface_acc_epoch', val_logs['val_adaface_acc'], on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_trans_acc_epoch', val_logs['val_trans_acc'], on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_combined_acc_epoch', val_logs['val_combined_acc'], on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_combined_acc', val_logs['val_combined_acc'], on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

        return None
    
    def test_epoch_end(self, outputs):
        # Re-use the same memory-efficient logic for testing
        return self.validation_epoch_end(outputs)
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def split_parameters(self, module):
        """ Split parameters into with and without weight decay.
        Special handling for TransMatcher parameters to ensure proper optimization.
        """
        params_decay = []
        params_no_decay = []
        
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        
        # Verify all parameters are accounted for
        all_params = set(module.parameters())
        decay_params = set(params_decay)
        no_decay_params = set(params_no_decay)
        
        # Check for missing or duplicate parameters
        missing_params = all_params - (decay_params | no_decay_params)
        duplicate_params = decay_params & no_decay_params
        
        if missing_params:
            print("WARNING: Some parameters were not assigned to decay/no-decay groups:")
            for p in missing_params:
                params_decay.append(p)
                print(f"- Adding parameter of shape {p.shape} to decay group")
        
        if duplicate_params:
            print("WARNING: Some parameters were assigned to both decay and no-decay groups:")
            for p in duplicate_params:
                params_no_decay.remove(p)
                print(f"- Removing duplicate parameter of shape {p.shape} from no-decay group")
        
        return params_decay, params_no_decay

    def configure_optimizers(self):
        # Collect all parameters
        backbone_params = []
        transmatcher_params = []
        bn_params = []
        head_params = []
        
        # Collect backbone parameters (excluding TransMatcher)
        for name, param in self.model.named_parameters():
            if not name.startswith('transmatcher.'):
                backbone_params.append(param)
        
        # Collect TransMatcher parameters separately
        for name, param in self.model.transmatcher.named_parameters():
            if 'bn' in name.lower():
                bn_params.append(param)
            else:
                transmatcher_params.append(param)
        
        # Collect head parameters
        for name, param in self.head.named_parameters():
            head_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.SGD([
            {'params': backbone_params + head_params, 'lr': self.hparams.lr, 'weight_decay': 5e-4},
            {'params': transmatcher_params, 'lr': self.hparams.lr * 0.0001, 'weight_decay': 5e-4},
            {'params': bn_params, 'lr': self.hparams.lr, 'weight_decay': 0}
        ])
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=self.hparams.lr_milestones, 
            gamma=self.hparams.lr_gamma
        )
        
        return [optimizer], [scheduler]
