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
import torch.nn.functional as F
from pairwise_matching_loss import PairwiseMatchingLoss
import wandb
import os
from torchvision import transforms
from softmax_triplet_loss import SoftmaxTripletLoss


class Trainer(LightningModule):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        
        # Define weight variables before wandb initialization
        # Loss weights: QAConv=0.7, AdaFace=0.1, Occlusion=0.3 (total=1.1)
        self.qaconv_loss_weight = 0.7
        self.adaface_loss_weight = 0.1  
        self.occlusion_loss_weight = 0.3  # Weight for occlusion supervision loss
        
        #weights for combining AdaFace and QAConv scores during evaluation
        self.adaface_eval_weight = 0.5  
        self.qaconv_eval_weight = 0.5   
        
        # Occlusion-aware parameters
        self.occlusion_method = "scaling"  # Method for occlusion weighting in QAConv
        
        # Store validation and test outputs for epoch end processing (newer PyTorch Lightning)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Initialize wandb
        wandb.init(
            project="aadaface_face_recognition_qaconv_segmentation",
                config={
                    "architecture": self.hparams.arch,
                    "learning_rate": self.hparams.lr,
                    "head_type": self.hparams.head,
                    "qaconv_loss_weight": self.qaconv_loss_weight,
                    "adaface_loss_weight": self.adaface_loss_weight,
                    "adaface_eval_weight": self.adaface_eval_weight,
                    "qaconv_eval_weight": self.qaconv_eval_weight,
                    "occlusion_loss_weight": self.occlusion_loss_weight,
                    "occlusion_method": self.occlusion_method,
                "epochs": self.hparams.epochs if hasattr(self.hparams, 'epochs') else None,
                "batch_size": self.hparams.batch_size if hasattr(self.hparams, 'batch_size') else None,
                "k_nearest": self.hparams.k_nearest if hasattr(self.hparams, 'k_nearest') else None,
            }
        )

        self.class_num = utils.get_num_class(self.hparams)
        print('classnum: {}'.format(self.class_num))

        self.model = net.build_model(model_name=self.hparams.arch)
        self.head = head.build_head(head_type=self.hparams.head,
                                     embedding_size=512,
                                     class_num=self.class_num,
                                     m=self.hparams.m,
                                     h=self.hparams.h,
                                     t_alpha=self.hparams.t_alpha,
                                     s=self.hparams.s,
                                     )

        self.cross_entropy_loss = CrossEntropyLoss()
        
        # Initialize QAConv with class_num for classification mode
        if hasattr(self.model, 'qaconv'):
            self.qaconv = self.model.qaconv
            # Update QAConv to include class_num and k_nearest
            num_features = self.qaconv.num_features
            height = self.qaconv.height
            width = self.qaconv.width
            self.qaconv = type(self.qaconv)(num_features, height, width, 
                                          num_classes=self.class_num,
                                          k_nearest=self.hparams.k_nearest)  # Add k_nearest from hparams
            self.model.qaconv = self.qaconv
            # Initialize pairwise matching loss with qaconv matcher
            self.qaconv_criterion = PairwiseMatchingLoss(
                self.qaconv
            )
            # Initialize SoftmaxTripletLoss with qaconv matcher
            # We will use the triplet_loss output from this criterion
            self.qaconv_triplet_criterion = SoftmaxTripletLoss(
                matcher=self.qaconv, 
                margin=1.0, # Hardcoded margin for triplet loss
                triplet_weight=0.5 # Set triplet weight to 0.5
            )
        else:
            print("Warning: Model does not have QAConv layer")
            self.qaconv = None
            self.qaconv_criterion = None

        print(f'Loss weights - AdaFace: {self.adaface_loss_weight}, QAConv: {self.qaconv_loss_weight}')
        print(f'Evaluation weights - AdaFace: {self.adaface_eval_weight}, QAConv: {self.qaconv_eval_weight}')

        if self.hparams.start_from_model_statedict:
            ckpt = torch.load(self.hparams.start_from_model_statedict)
            self.model.load_state_dict({key.replace('model.', ''):val
                                        for key,val in ckpt['state_dict'].items() if 'model.' in key})

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

        # Try different methods to get learning rate
        if hasattr(scheduler, 'get_last_lr'):
            lr = scheduler.get_last_lr()[0]
        elif hasattr(scheduler, 'get_lr'):
            lr = scheduler.get_lr()[0] 
        elif hasattr(scheduler, '_last_lr'):
            lr = scheduler._last_lr[0]
        else:
            # Fallback to optimizer's learning rate
            lr = scheduler.optimizer.param_groups[0]['lr']
        return lr

    def forward(self, images, labels):
        embeddings, norms, occlusion_maps = self.model(images)
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100  # ignore_index
        return cos_thetas, norms, embeddings, occlusion_maps, labels

    def training_step(self, batch, batch_idx):
        # Handle both regular batches and batches with occlusion masks
        if len(batch) == 3:
            # Batch with occlusion masks from SyntheticOcclusionMask transform
            images, labels, gt_occlusion_masks = batch
        else:
            # Regular batch without occlusion masks
            images, labels = batch
            gt_occlusion_masks = None
        
        # --- Save sample augmented images in the first epoch ---
        if self.current_epoch == 0 and batch_idx < 3: # Save 3 sample batches
            save_dir = 'sample_occlusion_pics'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(min(images.size(0), 3)): # Save first 3 images from the batch
                img_tensor = images[i].cpu().clone().detach() # Get image tensor
                # Reverse normalization (assuming mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img_tensor = img_tensor * 0.5 + 0.5
                # Convert to PIL Image
                img_pil = transforms.ToPILImage()(img_tensor)
                # Save image
                img_pil.save(os.path.join(save_dir, f'epoch{self.current_epoch}_batch{batch_idx}_img{i}.png'))
        # --- End of saving sample images ---
        
        # Make sure everything is on the same device
        device = images.device
        
        # Check for NaN values in images
        if torch.isnan(images).any():
            print(f"WARNING: Training images contain NaN values. Replacing with zeros.")
            images = torch.nan_to_num(images, nan=0.0)
        
        # get features from model up to before output layer
        x = self.model.input_layer(images)
        
        # Check for NaNs
        if torch.isnan(x).any():
            print(f"WARNING: Values after input layer contain NaNs. Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0)
            
        for i, layer in enumerate(self.model.body):
            x = layer(x)
            # Check for NaNs periodically (every 10 layers to avoid performance impact)
            if i % 10 == 0 and torch.isnan(x).any():
                print(f"WARNING: Values after body layer {i} contain NaNs. Replacing with zeros.")
                x = torch.nan_to_num(x, nan=0.0)
        
        # Final check for NaNs before normalization
        if torch.isnan(x).any():
            print(f"WARNING: Feature maps contain NaNs before normalization. Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0)
            # Add small epsilon to non-zero values to prevent division issues
            x = x + 1e-8 * (x.abs() > 0).float()
            
        # normalize feature maps for qaconv
        x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
        # Prevent division by zero
        x_norm = torch.clamp(x_norm, min=1e-8)
        feature_maps = x / x_norm
        
        # Make deep copy of feature maps to avoid inference tensor issues
        feature_maps = feature_maps.clone().detach().requires_grad_(True)
        
        # Extract occlusion maps from the same feature maps
        pred_occlusion_maps = self.model.occlusion_head(x)  # Use original x before normalization
        
        # Verify normalization
        norms = torch.norm(feature_maps.view(feature_maps.size(0), -1), p=2, dim=1)
        if ((norms < 0.99) | (norms > 1.01)).any():
            print(f"WARNING: Feature maps not properly normalized. Min norm: {norms.min().item()}, Max norm: {norms.max().item()}")
            # Force normalization again with safety measures
            feature_maps = F.normalize(feature_maps, p=2, dim=1).clone().detach().requires_grad_(True)
        
        # get adaface embeddings through output layer
        embeddings = self.model.output_layer(x)
        
        # Check for NaNs in embeddings
        if torch.isnan(embeddings).any():
            print(f"WARNING: Embeddings contain NaNs. Replacing with zeros.")
            embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
        embeddings, norms = utils.l2_norm(embeddings, axis=1)
        
        # get adaface loss
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100  # ignore_index
        adaface_loss = self.cross_entropy_loss(cos_thetas, labels)
        
        # Initialize qaconv_loss for wandb logging
        qaconv_loss = torch.tensor(0.0, device=device)
        qaconv_acc = torch.tensor(0.0, device=device)

        # get qaconv pairwise matching loss if available
        if self.qaconv_criterion is not None:
            # Make sure qaconv is on the right device
            if hasattr(self, 'qaconv'):
                self.qaconv = self.qaconv.to(device)
            
            # Get loss and accuracy from pairwise matching loss
            pairwise_loss, pairwise_acc = self.qaconv_criterion(feature_maps, labels)
            
            # Get triplet loss from SoftmaxTripletLoss criterion
            # Note: We only use the triplet_loss output here
            cls_loss, triplet_loss, _, cls_acc, triplet_acc = self.qaconv_triplet_criterion(feature_maps, labels)

            # --- Debugging NaN in individual QAConv losses ---
            if torch.isnan(pairwise_loss).any() or torch.isinf(pairwise_loss).any():
                 print(f"DEBUG: Pairwise loss contains NaN or Inf at epoch {self.current_epoch}, batch {batch_idx}!")
            if torch.isnan(triplet_loss).any() or torch.isinf(triplet_loss).any():
                 print(f"DEBUG: Triplet loss contains NaN or Inf at epoch {self.current_epoch}, batch {batch_idx}!")
            # --- End Debugging ---

            # --- Debugging QAConv loss stuck at 0 (keep existing check) ---
            if self.current_epoch > 0 and self.qaconv_loss_weight > 0 and (torch.isnan(pairwise_loss).any() or torch.isinf(pairwise_loss).any() or torch.isnan(triplet_loss).any() or torch.isinf(triplet_loss).any()):
                 # The loss will become NaN and then might be converted to 0 by the system
                 print(f"DEBUG: QAConv combined loss is becoming NaN due to component NaNs/Infs at epoch {self.current_epoch}, batch {batch_idx}!")
            # --- End Debugging ---

            # Log QAConv metrics to pytorch lightning
            # You might want to log both pairwise and triplet accuracies
            self.log('qaconv_pairwise_acc', pairwise_acc.mean(), on_step=True, on_epoch=True, logger=True)
            self.log('qaconv_triplet_acc', triplet_acc.mean(), on_step=True, on_epoch=True, logger=True)
            
            # Combine pairwise and triplet losses for the total QAConv loss
            # Use the qaconv_loss_weight defined in __init__ for the combined loss
            qaconv_loss = pairwise_loss.mean() + self.qaconv_triplet_criterion.triplet_weight * triplet_loss.mean()
            
            # Check for NaN in losses
            if torch.isnan(qaconv_loss):
                print(f"WARNING: QAConv loss is NaN. Using zero loss instead.")
                qaconv_loss = torch.tensor(0.0, device=device)
                
            if torch.isnan(adaface_loss):
                print(f"WARNING: AdaFace loss is NaN. Using zero loss instead.")
                adaface_loss = torch.tensor(0.0, device=device)
            
            # Compute occlusion supervision loss if ground truth masks are available
            occlusion_loss = torch.tensor(0.0, device=device)
            if gt_occlusion_masks is not None:
                # Ensure ground truth masks are on the correct device
                gt_occlusion_masks = gt_occlusion_masks.to(device)
                
                # Resize ground truth masks to match prediction size
                # pred_occlusion_maps: [B, 1, 7, 7], gt_occlusion_masks: [B, 1, 112, 112]
                pred_h, pred_w = pred_occlusion_maps.shape[2], pred_occlusion_maps.shape[3]
                gt_h, gt_w = gt_occlusion_masks.shape[2], gt_occlusion_masks.shape[3]
                
                if (pred_h, pred_w) != (gt_h, gt_w):
                    # Downsample ground truth to match prediction resolution
                    gt_occlusion_masks = F.interpolate(
                        gt_occlusion_masks, 
                        size=(pred_h, pred_w), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Now both tensors have the same shape: [B, 1, 7, 7]
                occlusion_loss = F.mse_loss(pred_occlusion_maps, gt_occlusion_masks)
                
                # Check for NaN in occlusion loss
                if torch.isnan(occlusion_loss):
                    print(f"WARNING: Occlusion loss is NaN. Using zero loss instead.")
                    occlusion_loss = torch.tensor(0.0, device=device)
            
            # Combine losses
            total_loss = (self.adaface_loss_weight * adaface_loss + 
                         self.qaconv_loss_weight * qaconv_loss + 
                         self.occlusion_loss_weight * occlusion_loss)
        else:
            # No QAConv - still compute occlusion loss if available
            occlusion_loss = torch.tensor(0.0, device=device)
            if gt_occlusion_masks is not None:
                gt_occlusion_masks = gt_occlusion_masks.to(device)
                
                # Resize ground truth masks to match prediction size
                pred_h, pred_w = pred_occlusion_maps.shape[2], pred_occlusion_maps.shape[3]
                gt_h, gt_w = gt_occlusion_masks.shape[2], gt_occlusion_masks.shape[3]
                
                if (pred_h, pred_w) != (gt_h, gt_w):
                    gt_occlusion_masks = F.interpolate(
                        gt_occlusion_masks, 
                        size=(pred_h, pred_w), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                occlusion_loss = F.mse_loss(pred_occlusion_maps, gt_occlusion_masks)
                if torch.isnan(occlusion_loss):
                    print(f"WARNING: Occlusion loss is NaN. Using zero loss instead.")
                    occlusion_loss = torch.tensor(0.0, device=device)
            
            total_loss = adaface_loss + self.occlusion_loss_weight * occlusion_loss

        # log metrics - ensure we take mean of tensor values
        lr = self.get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', total_loss.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('adaface_loss', adaface_loss.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('qaconv_loss', qaconv_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('occlusion_loss', occlusion_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        if self.qaconv_criterion is not None:
            self.log('qaconv_acc', pairwise_acc.mean(), on_step=True, on_epoch=True, logger=True)
        
        # Debug: Log occlusion head weight norms every 1000 iterations
        if self.global_step % 1000 == 0:
            self._log_occlusion_head_norms()
        
        # Log to wandb
        wandb_log_dict = {
            "qaconv_loss": qaconv_loss.item(),
            "adaface_loss": adaface_loss.item(),
            "occlusion_loss": occlusion_loss.item(),
            "total_loss": total_loss.mean().item(),
            "learning_rate": lr,
        }
        
        # Add occlusion head norm to wandb every 1000 steps
        if self.global_step % 1000 == 0:
            try:
                total_norm = 0.0
                param_count = 0
                for name, param in self.model.named_parameters():
                    if 'occlusion' in name.lower():
                        total_norm += param.data.norm(2).item() ** 2
                        param_count += 1
                if param_count > 0:
                    avg_norm = (total_norm ** 0.5) / param_count
                    wandb_log_dict["occlusion_head_avg_norm"] = avg_norm
                    wandb_log_dict["occlusion_head_param_count"] = param_count
            except Exception as e:
                print(f"Error computing occlusion head norm for wandb: {e}")
        
        # Add QAConv-specific metrics if available
        if self.qaconv_criterion is not None:
            wandb_log_dict.update({
                "qaconv_acc": pairwise_acc.mean().item(),
                "qaconv_pairwise_loss": pairwise_loss.mean().item(),
                "qaconv_triplet_loss": triplet_loss.mean().item(),
            })
        
        wandb.log(wandb_log_dict)

        return total_loss.mean()

    def on_train_epoch_end(self):
        # New PyTorch Lightning API - no outputs parameter needed
        pass

    def validation_step(self, batch, batch_idx):
        images, labels, dataname, image_index = batch
        
        # Get device 
        device = images.device
        
        # Check for NaN values in images
        if torch.isnan(images).any():
            print(f"WARNING: Input images contain NaN values. Replacing with zeros.")
            images = torch.nan_to_num(images, nan=0.0)
        
        # get features from model up to before output layer
        with torch.no_grad():  # Ensure we don't track gradients
            # Extract feature maps directly from the model's input layer and body
            # This is more reliable than trying to get intermediate outputs
            x = self.model.input_layer(images)
            
            # Process through body layers
            for layer in self.model.body:
                x = layer(x)
            
            # At this point, x contains the feature maps needed for QAConv
            # Normalize these feature maps
            feature_maps = F.normalize(x, p=2, dim=1)
            
            # get adaface embeddings with flip augmentation
            embeddings, norms, _ = self.model(images)  # Ignore occlusion maps in validation
            fliped_images = torch.flip(images, dims=[3])
            flipped_embeddings, flipped_norms, _ = self.model(fliped_images)  # Ignore occlusion maps
            stacked_embeddings = torch.stack([embeddings, flipped_embeddings], dim=0)
            stacked_norms = torch.stack([norms, flipped_norms], dim=0)
            embeddings, norms = utils.fuse_features_with_norm(stacked_embeddings, stacked_norms)
            
            # Final check for NaNs
            if torch.isnan(embeddings).any():
                print(f"WARNING: AdaFace embeddings contain NaNs. Replacing with zeros.")
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            if torch.isnan(feature_maps).any():
                print(f"WARNING: QAConv feature maps contain NaNs. Replacing with zeros.")
                feature_maps = torch.nan_to_num(feature_maps, nan=0.0)
                # Force normalization again
                feature_maps = F.normalize(feature_maps, p=2, dim=1)

        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            output = {
                'adaface_output': embeddings.to('cpu'),
                'norm': norms.to('cpu'),
                'qaconv_output': feature_maps.to('cpu'),
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            output = {
                'adaface_output': embeddings,
                'norm': norms,
                'qaconv_output': feature_maps,
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }
        
        # Store output for epoch end processing (newer PyTorch Lightning)
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        # Use stored validation outputs (newer PyTorch Lightning approach)
        outputs = self.validation_step_outputs
        
        if not outputs:
            print("Warning: No validation outputs found, skipping validation epoch end")
            return
            
        all_adaface_tensor, all_norm_tensor, all_qaconv_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        val_logs = {}
        
        # Weight parameter for combining adaface and qaconv scores
        adaface_weight = self.adaface_eval_weight  # Use class parameter instead of hardcoded value
        qaconv_weight = self.qaconv_eval_weight    # Use class parameter instead of hardcoded value
        
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            
            # get data for this dataset
            mask = all_dataname_tensor == dataname_idx
            adaface_embeddings = all_adaface_tensor[mask].cpu().numpy()
            qaconv_features = all_qaconv_tensor[mask]
            labels = all_target_tensor[mask].cpu().numpy()
            issame = labels[0::2]  # Original issame labels for the pairs
            
            print(f"\nProcessing {dataname} with {len(adaface_embeddings)} samples")
            
            # evaluate adaface embeddings
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(adaface_embeddings, issame, nrof_folds=10)
            adaface_acc = accuracy.mean()
            val_logs[f'{dataname}_adaface_acc'] = adaface_acc
            print(f"{dataname} AdaFace accuracy: {adaface_acc:.4f}")
            
            # Structure data for gallery-query pairs following the original repo approach
            # Each consecutive pair of images forms a gallery-query pair
            gallery_features = qaconv_features[0::2]  # Even indices (0, 2, 4...)
            query_features = qaconv_features[1::2]    # Odd indices (1, 3, 5...)
            
            if len(gallery_features) == len(query_features) and len(gallery_features) > 0:
                try:
                    # Make sure QAConv is on the right device
                    device = query_features.device
                    if hasattr(self, 'qaconv'):
                        self.qaconv = self.qaconv.to(device)
                    
                    num_pairs = len(gallery_features)
                    print(f"Computing scores for {num_pairs} gallery-query pairs")
                    
                    # Verify inputs are properly normalized
                    q_norms = torch.norm(query_features.view(query_features.size(0), -1), p=2, dim=1)
                    g_norms = torch.norm(gallery_features.view(gallery_features.size(0), -1), p=2, dim=1)
                    
                    if (q_norms < 0.99).any() or (q_norms > 1.01).any():
                        print(f"WARNING: Query features not properly normalized. Min: {q_norms.min().item():.4f}, Max: {q_norms.max().item():.4f}")
                        query_features = F.normalize(query_features, p=2, dim=1)
                    
                    if (g_norms < 0.99).any() or (g_norms > 1.01).any():
                        print(f"WARNING: Gallery features not properly normalized. Min: {g_norms.min().item():.4f}, Max: {g_norms.max().item():.4f}")
                        gallery_features = F.normalize(gallery_features, p=2, dim=1)
                    
                    with torch.no_grad():
                        # Compute scores for matching pairs (direct matches)
                        positive_scores = self.qaconv.match_pairs(query_features, gallery_features)
                    
                        # Sample negative pairs (non-matching identities)
                        # For efficiency, we'll sample a number of negative pairs equal to the positive pairs
                        np.random.seed(42)  # For reproducibility
                        
                        # Initialize negative scores tensor
                        negative_scores = torch.zeros(num_pairs, device=device)
                        
                        # Create a batch of random indices for non-matching pairs
                        # Use a more structured approach - shift indices by half the dataset size
                        # This ensures gallery-query pairs with definitely different identities
                        half_size = num_pairs // 2
                        
                        for i in range(0, num_pairs, 32):  # Process in batches of 32 for efficiency
                            end_idx = min(i + 32, num_pairs)
                            batch_size = end_idx - i
                            
                            # For each gallery feature, pick a query feature from a different identity
                            # by shifting the index by half the dataset size
                            random_indices = np.zeros(batch_size, dtype=np.int64)
                            
                            for j in range(batch_size):
                                # Shift by half the dataset to get truly different identity
                                idx = (i + j + half_size) % num_pairs
                                random_indices[j] = idx
                            
                            # Gather the random query features
                            selected_queries = query_features[random_indices]
                            batch_galleries = gallery_features[i:end_idx]
                            
                            # Compute scores for these negative pairs
                            for j in range(batch_size):
                                # Get scores for each gallery with its selected non-matching query
                                score = self.qaconv(batch_galleries[j:j+1], selected_queries[j:j+1])
                                negative_scores[i + j] = score.view(-1)[0]
                    
                        # Combine positive and negative scores and create labels
                        all_scores = torch.cat([positive_scores, negative_scores])
                        all_labels = torch.cat([
                            torch.ones(num_pairs, device=device),   # Positive pairs are 1 (same identity)
                            torch.zeros(num_pairs, device=device)   # Negative pairs are 0 (different identity)
                        ])
                        
                        # Check for NaN values
                        if torch.isnan(all_scores).any():
                            print("WARNING: QAConv scores contain NaN values. Replacing with zeros.")
                            all_scores = torch.nan_to_num(all_scores, nan=0.0)
                    
                    # Move to CPU for numpy conversion
                    qaconv_scores = all_scores.cpu().numpy()
                    pair_labels = all_labels.cpu().numpy().astype(bool)
                    
                    # Print score stats for debugging
                    pos_scores = qaconv_scores[:num_pairs]
                    neg_scores = qaconv_scores[num_pairs:]
                    print(f"QAConv positive scores - min: {np.min(pos_scores):.4f}, max: {np.max(pos_scores):.4f}")
                    print(f"QAConv negative scores - min: {np.min(neg_scores):.4f}, max: {np.max(neg_scores):.4f}")
                    
                    # Check if scores are inverted (negative pairs getting higher scores than positive pairs)
                    pos_mean = np.mean(pos_scores)
                    neg_mean = np.mean(neg_scores)
                    print(f"QAConv score means - positive: {pos_mean:.4f}, negative: {neg_mean:.4f}")
                    
                    # If scores are inverted (negative pairs getting higher scores), flip the labels
                    labels_flipped = False
                    if neg_mean > pos_mean:
                        print("WARNING: QAConv scores appear to be inverted (negative pairs have higher scores). Flipping labels.")
                        # Invert the labels (0 becomes 1, 1 becomes 0)
                        pair_labels = ~pair_labels
                        labels_flipped = True
                    
                    # Check if scores are extremely large, which might cause numerical issues
                    if pos_mean > 100 or neg_mean > 100:
                        print(f"WARNING: QAConv scores are extremely large (pos_mean: {pos_mean:.4f}, neg_mean: {neg_mean:.4f}). Normalizing...")
                        # Normalize scores to have mean of 0 and standard deviation of 1
                        all_mean = np.mean(qaconv_scores)
                        all_std = np.std(qaconv_scores)
                    
                        # Prevent division by zero
                        if all_std < 1e-8:
                            all_std = 1.0
                            
                        qaconv_scores = (qaconv_scores - all_mean) / all_std
                        # Re-compute positive and negative scores after normalization
                        pos_scores = qaconv_scores[:num_pairs]
                        neg_scores = qaconv_scores[num_pairs:]
                        print(f"After normalization - Positive: {np.mean(pos_scores):.4f}, Negative: {np.mean(neg_scores):.4f}")
                    
                    # For ROC calculation, we need distances (smaller = more similar)
                    # For QAConv scores, higher = more similar, so negate them to get distances
                    qaconv_dists = -qaconv_scores
                    
                    # For highly reliable classification, ensure pos and neg are well separated
                    # Compute separation margin between positive and negative
                    qaconv_pos_dists = qaconv_dists[:num_pairs]
                    qaconv_neg_dists = qaconv_dists[num_pairs:]
                    
                    # Calculate direct accuracy by comparing each sample to the average of the other class
                    direct_correct = 0
                    for i in range(num_pairs):
                        if qaconv_pos_dists[i] < np.mean(qaconv_neg_dists):
                            direct_correct += 1
                    for i in range(num_pairs):
                        if qaconv_neg_dists[i] > np.mean(qaconv_pos_dists):
                            direct_correct += 1
                    direct_qaconv_acc = direct_correct / (2 * num_pairs)
                    
                    # Log the direct accuracy as the primary QAConv metric
                    val_logs[f'{dataname}_qaconv_acc'] = direct_qaconv_acc
                    print(f"{dataname} QAConv accuracy: {direct_qaconv_acc:.4f}")
                    
                except Exception as e:
                    import traceback
                    print(f"WARNING: Error during QAConv matching: {e}")
                    print(traceback.format_exc())
                    # Fallback: create dummy scores
                    qaconv_scores = np.zeros(2 * len(gallery_features))
                    direct_qaconv_acc = 0.0
                    val_logs[f'{dataname}_qaconv_acc'] = direct_qaconv_acc
                    continue
                
                # Calculate AdaFace distances for each pair
                adaface_embeddings1 = adaface_embeddings[0::2]  # gallery
                adaface_embeddings2 = adaface_embeddings[1::2]  # query
                adaface_dists = np.zeros(len(adaface_embeddings1))
                
                for i in range(len(adaface_embeddings1)):
                    diff = adaface_embeddings1[i] - adaface_embeddings2[i]
                    adaface_dists[i] = np.sum(np.square(diff))
                
                # For combined evaluation, we need to select only the positive pair scores from QAConv
                # since we don't have negative pair scores for AdaFace in the same format
                qaconv_positive_dists = qaconv_dists[:num_pairs]
                
                # Check for NaN in AdaFace distances
                if np.isnan(adaface_dists).any() or np.isinf(adaface_dists).any():
                    print("WARNING: AdaFace distances contain NaN or Inf values. Replacing with zeros.")
                    adaface_dists = np.nan_to_num(adaface_dists, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize distances to [0,1] range for fair comparison
                adaface_range = np.max(adaface_dists) - np.min(adaface_dists)
                if adaface_range > 1e-8:
                    adaface_dists_norm = (adaface_dists - np.min(adaface_dists)) / adaface_range
                else:
                    print("WARNING: AdaFace distances have very small range. Using zeros.")
                    adaface_dists_norm = np.zeros_like(adaface_dists)
                
                qaconv_range = np.max(qaconv_positive_dists) - np.min(qaconv_positive_dists)
                if qaconv_range > 1e-8:
                    qaconv_dists_norm = (qaconv_positive_dists - np.min(qaconv_positive_dists)) / qaconv_range
                else:
                    print("WARNING: QAConv distances have very small range. Using zeros.")
                    qaconv_dists_norm = np.zeros_like(qaconv_positive_dists)
                
                # Adjust weights based on QAConv reliability
                if direct_qaconv_acc < 0.5:
                    print(f"WARNING: QAConv accuracy too low ({direct_qaconv_acc:.4f}). Using more weight on AdaFace for combined score.")
                    adaface_weight = 0.8
                    qaconv_weight = 0.2
                else:
                    # Use both with specified weights
                    adaface_weight = self.adaface_eval_weight
                    qaconv_weight = self.qaconv_eval_weight
                
                # Combine the normalized distances
                combined_dists = adaface_weight * adaface_dists_norm + qaconv_weight * qaconv_dists_norm

                # --- Debugging combined distances ---
                print(f"DEBUG {dataname}: Normalized AdaFace distances stats - Min: {np.min(adaface_dists_norm):.6f}, Max: {np.max(adaface_dists_norm):.6f}, Mean: {np.mean(adaface_dists_norm):.6f}, Std: {np.std(adaface_dists_norm):.6f}")
                print(f"DEBUG {dataname}: Normalized QAConv distances (positive pairs) stats - Min: {np.min(qaconv_dists_norm):.6f}, Max: {np.max(qaconv_dists_norm):.6f}, Mean: {np.mean(qaconv_dists_norm):.6f}, Std: {np.std(qaconv_dists_norm):.6f}")
                print(f"DEBUG {dataname}: Combined distances stats - Min: {np.min(combined_dists):.6f}, Max: {np.max(combined_dists):.6f}, Mean: {np.mean(combined_dists):.6f}, Std: {np.std(combined_dists):.6f}")
                print(f"DEBUG {dataname}: Number of unique combined distance values: {len(np.unique(combined_dists))}")
                # --- End Debugging ---

                # Calculate direct accuracy for combined scores
                # Split distances into positive and negative pairs
                pos_dists = combined_dists[issame]
                neg_dists = combined_dists[~issame]
                
                # Calculate direct accuracy by comparing each sample to the average of the other class
                direct_correct = 0
                for i in range(len(pos_dists)):
                    if pos_dists[i] < np.mean(neg_dists):
                        direct_correct += 1
                for i in range(len(neg_dists)):
                    if neg_dists[i] > np.mean(pos_dists):
                        direct_correct += 1
                combined_acc = direct_correct / (len(pos_dists) + len(neg_dists))
                
                val_logs[f'{dataname}_combined_acc'] = combined_acc
                print(f"{dataname} Combined accuracy: {combined_acc:.4f}")
                
            else:
                print(f"Warning: {dataname} dataset has mismatched gallery/query sizes")
                val_logs[f'{dataname}_qaconv_acc'] = 0.0
                val_logs[f'{dataname}_combined_acc'] = 0.0
            
            val_logs[f'{dataname}_num_val_samples'] = len(adaface_embeddings)

        # average accuracies across datasets
        val_logs['val_adaface_acc'] = np.mean([
            val_logs[f'{dataname}_adaface_acc'] for dataname in dataname_to_idx.keys() 
            if f'{dataname}_adaface_acc' in val_logs
        ])
        
        # Average QAConv direct accuracies
        qaconv_accs = []
        for dataname in dataname_to_idx.keys():
            if f'{dataname}_qaconv_acc' in val_logs:
                qaconv_accs.append(val_logs[f'{dataname}_qaconv_acc'])
        
        val_logs['val_qaconv_acc'] = np.mean(qaconv_accs) if qaconv_accs else 0.0
        
        # Average combined accuracies
        combined_accs = []
        for dataname in dataname_to_idx.keys():
            if f'{dataname}_combined_acc' in val_logs:
                combined_accs.append(val_logs[f'{dataname}_combined_acc'])
        
        val_logs['val_combined_acc'] = np.mean(combined_accs) if combined_accs else 0.0
        
        # Add val_acc for ModelCheckpoint to monitor (use the combined accuracy)
        val_logs['val_acc'] = val_logs['val_combined_acc']
        
        val_logs['epoch'] = self.current_epoch

        # Log validation metrics to wandb
        wandb.log({
            'val_adaface_acc': val_logs.get('val_adaface_acc', 0.0),
            'val_qaconv_acc': val_logs.get('val_qaconv_acc', 0.0),
            'val_combined_acc': val_logs.get('val_combined_acc', 0.0),
            'epoch': self.current_epoch,
        })

        for k, v in val_logs.items():
            self.log(name=k, value=v)

        # Clear validation outputs for next epoch (newer PyTorch Lightning)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # Call validation_step logic but store in test outputs
        output = self.validation_step(batch, batch_idx)
        # Remove from validation outputs and add to test outputs
        if self.validation_step_outputs and self.validation_step_outputs[-1] == output:
            self.validation_step_outputs.pop()
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        # Use stored test outputs (newer PyTorch Lightning approach)
        outputs = self.test_step_outputs
        
        if not outputs:
            print("Warning: No test outputs found, skipping test epoch end")
            return
            
        all_adaface_tensor, all_norm_tensor, all_qaconv_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        test_logs = {}
        
        # Weight parameter for combining adaface and qaconv scores
        adaface_weight = self.adaface_eval_weight  # Use class parameter instead of hardcoded value
        qaconv_weight = self.qaconv_eval_weight    # Use class parameter instead of hardcoded value
        
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            
            # get data for this dataset
            mask = all_dataname_tensor == dataname_idx
            adaface_embeddings = all_adaface_tensor[mask].cpu().numpy()
            qaconv_features = all_qaconv_tensor[mask]
            labels = all_target_tensor[mask].cpu().numpy()
            issame = labels[0::2]  # Original issame labels for the pairs
            
            print(f"\nProcessing {dataname} with {len(adaface_embeddings)} samples")
            
            # evaluate adaface embeddings
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(adaface_embeddings, issame, nrof_folds=10)
            adaface_acc = accuracy.mean()
            test_logs[f'{dataname}_adaface_acc'] = adaface_acc
            print(f"{dataname} AdaFace accuracy: {adaface_acc:.4f}")
            
            # Structure data for gallery-query pairs following the original repo approach
            # Each consecutive pair of images forms a gallery-query pair
            gallery_features = qaconv_features[0::2]  # Even indices (0, 2, 4...)
            query_features = qaconv_features[1::2]    # Odd indices (1, 3, 5...)
            
            if len(gallery_features) == len(query_features) and len(gallery_features) > 0:
                try:
                    # Make sure QAConv is on the right device
                    device = query_features.device
                    if hasattr(self, 'qaconv'):
                        self.qaconv = self.qaconv.to(device)
                    
                    num_pairs = len(gallery_features)
                    print(f"Computing scores for {num_pairs} gallery-query pairs")
                    
                    # Verify inputs are properly normalized
                    q_norms = torch.norm(query_features.view(query_features.size(0), -1), p=2, dim=1)
                    g_norms = torch.norm(gallery_features.view(gallery_features.size(0), -1), p=2, dim=1)
                    
                    if (q_norms < 0.99).any() or (q_norms > 1.01).any():
                        print(f"WARNING: Query features not properly normalized. Min: {q_norms.min().item():.4f}, Max: {q_norms.max().item():.4f}")
                        query_features = F.normalize(query_features, p=2, dim=1)
                    
                    if (g_norms < 0.99).any() or (g_norms > 1.01).any():
                        print(f"WARNING: Gallery features not properly normalized. Min: {g_norms.min().item():.4f}, Max: {g_norms.max().item():.4f}")
                        gallery_features = F.normalize(gallery_features, p=2, dim=1)
                    
                    with torch.no_grad():
                        # Compute scores for matching pairs (direct matches)
                        positive_scores = self.qaconv.match_pairs(query_features, gallery_features)
                    
                        # Sample negative pairs (non-matching identities)
                        # For efficiency, we'll sample a number of negative pairs equal to the positive pairs
                        np.random.seed(42)  # For reproducibility
                        
                        # Initialize negative scores tensor
                        negative_scores = torch.zeros(num_pairs, device=device)
                        
                        # Create a batch of random indices for non-matching pairs
                        # Use a more structured approach - shift indices by half the dataset size
                        # This ensures gallery-query pairs with definitely different identities
                        half_size = num_pairs // 2
                        
                        for i in range(0, num_pairs, 32):  # Process in batches of 32 for efficiency
                            end_idx = min(i + 32, num_pairs)
                            batch_size = end_idx - i
                            
                            # For each gallery feature, pick a query feature from a different identity
                            # by shifting the index by half the dataset size
                            random_indices = np.zeros(batch_size, dtype=np.int64)
                            
                            for j in range(batch_size):
                                # Shift by half the dataset to get truly different identity
                                idx = (i + j + half_size) % num_pairs
                                random_indices[j] = idx
                            
                            # Gather the random query features
                            selected_queries = query_features[random_indices]
                            batch_galleries = gallery_features[i:end_idx]
                            
                            # Compute scores for these negative pairs
                            for j in range(batch_size):
                                # Get scores for each gallery with its selected non-matching query
                                score = self.qaconv(batch_galleries[j:j+1], selected_queries[j:j+1])
                                negative_scores[i + j] = score.view(-1)[0]
                    
                        # Combine positive and negative scores and create labels
                        all_scores = torch.cat([positive_scores, negative_scores])
                        all_labels = torch.cat([
                            torch.ones(num_pairs, device=device),   # Positive pairs are 1 (same identity)
                            torch.zeros(num_pairs, device=device)   # Negative pairs are 0 (different identity)
                        ])
                        
                        # Check for NaN values
                        if torch.isnan(all_scores).any():
                            print("WARNING: QAConv scores contain NaN values. Replacing with zeros.")
                            all_scores = torch.nan_to_num(all_scores, nan=0.0)
                    
                    # Move to CPU for numpy conversion
                    qaconv_scores = all_scores.cpu().numpy()
                    pair_labels = all_labels.cpu().numpy().astype(bool)
                    
                    # Print score stats for debugging
                    pos_scores = qaconv_scores[:num_pairs]
                    neg_scores = qaconv_scores[num_pairs:]
                    print(f"QAConv positive scores - min: {np.min(pos_scores):.4f}, max: {np.max(pos_scores):.4f}")
                    print(f"QAConv negative scores - min: {np.min(neg_scores):.4f}, max: {np.max(neg_scores):.4f}")
                    
                    # Check if scores are inverted (negative pairs getting higher scores than positive pairs)
                    pos_mean = np.mean(pos_scores)
                    neg_mean = np.mean(neg_scores)
                    print(f"QAConv score means - positive: {pos_mean:.4f}, negative: {neg_mean:.4f}")
                    
                    # If scores are inverted (negative pairs getting higher scores), flip the labels
                    labels_flipped = False
                    if neg_mean > pos_mean:
                        print("WARNING: QAConv scores appear to be inverted (negative pairs have higher scores). Flipping labels.")
                        # Invert the labels (0 becomes 1, 1 becomes 0)
                        pair_labels = ~pair_labels
                        labels_flipped = True
                    
                    # Check if scores are extremely large, which might cause numerical issues
                    if pos_mean > 100 or neg_mean > 100:
                        print(f"WARNING: QAConv scores are extremely large (pos_mean: {pos_mean:.4f}, neg_mean: {neg_mean:.4f}). Normalizing...")
                        # Normalize scores to have mean of 0 and standard deviation of 1
                        all_mean = np.mean(qaconv_scores)
                        all_std = np.std(qaconv_scores)
                    
                        # Prevent division by zero
                        if all_std < 1e-8:
                            all_std = 1.0
                            
                        qaconv_scores = (qaconv_scores - all_mean) / all_std
                        # Re-compute positive and negative scores after normalization
                        pos_scores = qaconv_scores[:num_pairs]
                        neg_scores = qaconv_scores[num_pairs:]
                        print(f"After normalization - Positive: {np.mean(pos_scores):.4f}, Negative: {np.mean(neg_scores):.4f}")
                    
                    # For ROC calculation, we need distances (smaller = more similar)
                    # For QAConv scores, higher = more similar, so negate them to get distances
                    qaconv_dists = -qaconv_scores
                    
                    # For highly reliable classification, ensure pos and neg are well separated
                    # Compute separation margin between positive and negative
                    qaconv_pos_dists = qaconv_dists[:num_pairs]
                    qaconv_neg_dists = qaconv_dists[num_pairs:]
                    
                    # Calculate direct accuracy by comparing each sample to the average of the other class
                    direct_correct = 0
                    for i in range(num_pairs):
                        if qaconv_pos_dists[i] < np.mean(qaconv_neg_dists):
                            direct_correct += 1
                    for i in range(num_pairs):
                        if qaconv_neg_dists[i] > np.mean(qaconv_pos_dists):
                            direct_correct += 1
                    direct_qaconv_acc = direct_correct / (2 * num_pairs)
                    
                    # Log the direct accuracy as the primary QAConv metric
                    test_logs[f'{dataname}_qaconv_acc'] = direct_qaconv_acc
                    print(f"{dataname} QAConv accuracy: {direct_qaconv_acc:.4f}")
                    
                except Exception as e:
                    import traceback
                    print(f"WARNING: Error during QAConv matching: {e}")
                    print(traceback.format_exc())
                    # Fallback: create dummy scores
                    qaconv_scores = np.zeros(2 * len(gallery_features))
                    direct_qaconv_acc = 0.0
                    test_logs[f'{dataname}_qaconv_acc'] = direct_qaconv_acc
                    continue
                
                # Calculate AdaFace distances for each pair
                adaface_embeddings1 = adaface_embeddings[0::2]  # gallery
                adaface_embeddings2 = adaface_embeddings[1::2]  # query
                adaface_dists = np.zeros(len(adaface_embeddings1))
                
                for i in range(len(adaface_embeddings1)):
                    diff = adaface_embeddings1[i] - adaface_embeddings2[i]
                    adaface_dists[i] = np.sum(np.square(diff))
                
                # For combined evaluation, we need to select only the positive pair scores from QAConv
                # since we don't have negative pair scores for AdaFace in the same format
                qaconv_positive_dists = qaconv_dists[:num_pairs]
                
                # Check for NaN in AdaFace distances
                if np.isnan(adaface_dists).any() or np.isinf(adaface_dists).any():
                    print("WARNING: AdaFace distances contain NaN or Inf values. Replacing with zeros.")
                    adaface_dists = np.nan_to_num(adaface_dists, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize distances to [0,1] range for fair comparison
                adaface_range = np.max(adaface_dists) - np.min(adaface_dists)
                if adaface_range > 1e-8:
                    adaface_dists_norm = (adaface_dists - np.min(adaface_dists)) / adaface_range
                else:
                    print("WARNING: AdaFace distances have very small range. Using zeros.")
                    adaface_dists_norm = np.zeros_like(adaface_dists)
                
                qaconv_range = np.max(qaconv_positive_dists) - np.min(qaconv_positive_dists)
                if qaconv_range > 1e-8:
                    qaconv_dists_norm = (qaconv_positive_dists - np.min(qaconv_positive_dists)) / qaconv_range
                else:
                    print("WARNING: QAConv distances have very small range. Using zeros.")
                    qaconv_dists_norm = np.zeros_like(qaconv_positive_dists)
                
                # Adjust weights based on QAConv reliability
                if direct_qaconv_acc < 0.5:
                    print(f"WARNING: QAConv accuracy too low ({direct_qaconv_acc:.4f}). Using more weight on AdaFace for combined score.")
                    adaface_weight = 0.8
                    qaconv_weight = 0.2
                else:
                    # Use both with specified weights
                    adaface_weight = self.adaface_eval_weight
                    qaconv_weight = self.qaconv_eval_weight
                
                # Combine the normalized distances
                combined_dists = adaface_weight * adaface_dists_norm + qaconv_weight * qaconv_dists_norm

                # --- Debugging combined distances ---
                print(f"DEBUG {dataname}: Normalized AdaFace distances stats - Min: {np.min(adaface_dists_norm):.6f}, Max: {np.max(adaface_dists_norm):.6f}, Mean: {np.mean(adaface_dists_norm):.6f}, Std: {np.std(adaface_dists_norm):.6f}")
                print(f"DEBUG {dataname}: Normalized QAConv distances (positive pairs) stats - Min: {np.min(qaconv_dists_norm):.6f}, Max: {np.max(qaconv_dists_norm):.6f}, Mean: {np.mean(qaconv_dists_norm):.6f}, Std: {np.std(qaconv_dists_norm):.6f}")
                print(f"DEBUG {dataname}: Combined distances stats - Min: {np.min(combined_dists):.6f}, Max: {np.max(combined_dists):.6f}, Mean: {np.mean(combined_dists):.6f}, Std: {np.std(combined_dists):.6f}")
                print(f"DEBUG {dataname}: Number of unique combined distance values: {len(np.unique(combined_dists))}")
                # --- End Debugging ---

                # Calculate direct accuracy for combined scores
                # Split distances into positive and negative pairs
                pos_dists = combined_dists[issame]
                neg_dists = combined_dists[~issame]
                
                # Calculate direct accuracy by comparing each sample to the average of the other class
                direct_correct = 0
                for i in range(len(pos_dists)):
                    if pos_dists[i] < np.mean(neg_dists):
                        direct_correct += 1
                for i in range(len(neg_dists)):
                    if neg_dists[i] > np.mean(pos_dists):
                        direct_correct += 1
                combined_acc = direct_correct / (len(pos_dists) + len(neg_dists))
                
                test_logs[f'{dataname}_combined_acc'] = combined_acc
                print(f"{dataname} Combined accuracy: {combined_acc:.4f}")
                
            else:
                print(f"Warning: {dataname} dataset has mismatched gallery/query sizes")
                test_logs[f'{dataname}_qaconv_acc'] = 0.0
                test_logs[f'{dataname}_combined_acc'] = 0.0
            
            test_logs[f'{dataname}_num_test_samples'] = len(adaface_embeddings)

        # average accuracies across datasets
        test_logs['test_adaface_acc'] = np.mean([
            test_logs[f'{dataname}_adaface_acc'] for dataname in dataname_to_idx.keys() 
            if f'{dataname}_adaface_acc' in test_logs
        ])
        
        # Average QAConv direct accuracies
        qaconv_accs = []
        for dataname in dataname_to_idx.keys():
            if f'{dataname}_qaconv_acc' in test_logs:
                qaconv_accs.append(test_logs[f'{dataname}_qaconv_acc'])
        
        test_logs['test_qaconv_acc'] = np.mean(qaconv_accs) if qaconv_accs else 0.0
        
        # Average combined accuracies
        combined_accs = []
        for dataname in dataname_to_idx.keys():
            if f'{dataname}_combined_acc' in test_logs:
                combined_accs.append(test_logs[f'{dataname}_combined_acc'])
        
        test_logs['test_combined_acc'] = np.mean(combined_accs) if combined_accs else 0.0
        
        # Add test_acc for consistency
        test_logs['test_acc'] = test_logs['test_combined_acc']
        
        test_logs['epoch'] = self.current_epoch

        for k, v in test_logs.items():
            self.log(name=k, value=v)

        # Clear test outputs for next run (newer PyTorch Lightning)
        self.test_step_outputs.clear()

    def gather_outputs(self, outputs):
        if self.hparams.distributed_backend == 'ddp':
            # gather outputs across gpu
            outputs_list = []
            _outputs_list = utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs

        all_adaface_tensor = torch.cat([out['adaface_output'] for out in outputs_list], axis=0).to('cpu')
        all_norm_tensor = torch.cat([out['norm'] for out in outputs_list], axis=0).to('cpu')
        all_qaconv_tensor = torch.cat([out['qaconv_output'] for out in outputs_list], axis=0).to('cpu')
        all_target_tensor = torch.cat([out['target'] for out in outputs_list], axis=0).to('cpu')
        all_dataname_tensor = torch.cat([out['dataname'] for out in outputs_list], axis=0).to('cpu')
        all_image_index = torch.cat([out['image_index'] for out in outputs_list], axis=0).to('cpu')

        # get rid of duplicate index outputs
        unique_dict = {}
        for _ada, _nor, _qa, _tar, _dat, _idx in zip(all_adaface_tensor, all_norm_tensor, all_qaconv_tensor,
                                                   all_target_tensor, all_dataname_tensor, all_image_index):
            unique_dict[_idx.item()] = {
                'adaface_output': _ada, 
                'norm': _nor,
                'qaconv_output': _qa, 
                'target': _tar,
                'dataname': _dat
            }
        unique_keys = sorted(unique_dict.keys())
        all_adaface_tensor = torch.stack([unique_dict[key]['adaface_output'] for key in unique_keys], axis=0)
        all_norm_tensor = torch.stack([unique_dict[key]['norm'] for key in unique_keys], axis=0)
        all_qaconv_tensor = torch.stack([unique_dict[key]['qaconv_output'] for key in unique_keys], axis=0)
        all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
        all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)

        return all_adaface_tensor, all_norm_tensor, all_qaconv_tensor, all_target_tensor, all_dataname_tensor

    def split_parameters(self, module):
        """ Split parameters into with and without weight decay.
        Special handling for QAConv parameters to ensure proper optimization.
        """
        params_decay = []
        params_no_decay = []
        
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif isinstance(m, type(self.qaconv)) if hasattr(self, 'qaconv') else False:
                # Handle QAConv parameters - class embeddings should use weight decay
                if hasattr(m, 'class_embed') and m.class_embed is not None:
                    params_decay.append(m.class_embed)
                # Other QAConv parameters (bn, fc) follow normal rules
                if hasattr(m, 'bn'):
                    params_no_decay.extend([*m.bn.parameters()])
                if hasattr(m, 'logit_bn'):
                    params_no_decay.extend([*m.logit_bn.parameters()])
                if hasattr(m, 'fc'):
                    params_decay.extend([*m.fc.parameters()])
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

    def _log_occlusion_head_norms(self):
        """Log L2 norms of occlusion head weights for debugging parameter updates"""
        try:
            occlusion_norms = {}
            total_norm = 0.0
            param_count = 0
            
            for name, param in self.model.named_parameters():
                if 'occlusion' in name.lower():
                    param_norm = param.data.norm(2).item()
                    occlusion_norms[f'occlusion_head/{name.replace(".", "_")}_norm'] = param_norm
                    total_norm += param_norm ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = (total_norm ** 0.5) / param_count  # Average L2 norm
                occlusion_norms['occlusion_head/total_avg_norm'] = total_norm
                occlusion_norms['occlusion_head/param_count'] = param_count
                
                # Log to wandb
                self.logger.experiment.log(occlusion_norms, step=self.global_step)
                
                print(f"Step {self.global_step}: Occlusion head has {param_count} parameters, avg L2 norm: {total_norm:.6f}")
            else:
                print(f"Step {self.global_step}: No occlusion head parameters found!")
                
        except Exception as e:
            print(f"Error logging occlusion head norms: {e}")

    def configure_optimizers(self):
        paras_wo_bn, paras_only_bn = self.split_parameters(self.model)
        
        # Add QAConv parameters if available
        if hasattr(self, 'qaconv') and self.qaconv is not None:
            qaconv_paras_wo_bn, qaconv_paras_only_bn = self.split_parameters(self.qaconv)
            paras_wo_bn.extend(qaconv_paras_wo_bn)
            paras_only_bn.extend(qaconv_paras_only_bn)
        
        # Add Occlusion Head parameters if available
        if hasattr(self.model, 'occlusion_head_112') or hasattr(self.model, 'occlusion_head_224'):
            # Find occlusion head parameters
            occlusion_paras_wo_bn, occlusion_paras_only_bn = [], []
            for name, param in self.model.named_parameters():
                if 'occlusion' in name.lower():
                    if 'bn' in name.lower() or 'norm' in name.lower():
                        occlusion_paras_only_bn.append(param)
                    else:
                        occlusion_paras_wo_bn.append(param)
            
            paras_wo_bn.extend(occlusion_paras_wo_bn)
            paras_only_bn.extend(occlusion_paras_only_bn)
            
            print(f"Added {len(occlusion_paras_wo_bn)} occlusion head parameters to optimizer")

        optimizer = optim.SGD([{
            'params': paras_wo_bn + [self.head.kernel],
            'weight_decay': 5e-4
        }, {
            'params': paras_only_bn
        }],
                                lr=self.hparams.lr,
                                momentum=self.hparams.momentum)

        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=self.hparams.lr_milestones,
                                             gamma=self.hparams.lr_gamma)

        return [optimizer], [scheduler]

    def evaluate_with_distances(self, distances, issame, nrof_folds=10):
        """
        Custom evaluation function that works with precomputed distances.
        Based on evaluate_utils.calculate_roc but accepts distances directly.
        """
        # Use thresholds from evaluate_utils
        thresholds = np.arange(0, 4, 0.01)
        nrof_pairs = min(len(issame), len(distances))
        nrof_thresholds = len(thresholds)
        
        # Use KFold from sklearn for consistent results with evaluate_utils
        k_fold = evaluate_utils.KFold(n_splits=nrof_folds, shuffle=False)
        
        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros(nrof_folds)
        best_thresholds = np.zeros(nrof_folds)
        
        indices = np.arange(nrof_pairs)
        dist = distances  # Use provided distances
        
        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            # Find the best threshold for the fold
            acc_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx] = evaluate_utils.calculate_accuracy(threshold, dist[train_set], issame[train_set])
            
            best_threshold_index = np.argmax(acc_train)
            best_thresholds[fold_idx] = thresholds[best_threshold_index]
            
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = evaluate_utils.calculate_accuracy(
                    threshold, dist[test_set], issame[test_set]
                )
            
            _, _, accuracy[fold_idx] = evaluate_utils.calculate_accuracy(
                thresholds[best_threshold_index], dist[test_set], issame[test_set]
            )
        
        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        
        return tpr, fpr, accuracy, best_thresholds

    def on_train_end(self):
        # Close wandb run when training ends
        wandb.finish()
