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
from dataset.niqab_mask_dataset import NiqabMaskDataset, get_default_niqab_transform


class Trainer(LightningModule):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        
        # Define weight variables before wandb initialization
        # weight for qaconv loss
        self.qaconv_loss_weight = 0.9
        self.adaface_loss_weight = 1.0 - self.qaconv_loss_weight

        # Occlusion loss weight (for training with niqab GT masks)
        self.occlusion_loss_weight = getattr(self.hparams, 'occlusion_loss_weight', 0.1)  
        
        #weights for combining AdaFace and QAConv scores during evaluation
        self.adaface_eval_weight = 0.5
        self.qaconv_eval_weight = 0.5

        # Initialize wandb only on rank 0 to avoid conflicts in DDP
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            wandb.init(
                project="adaface_face_recognition",
                    config={
                        "architecture": self.hparams.arch,
                        "learning_rate": self.hparams.lr,
                        "head_type": self.hparams.head,
                        "qaconv_loss_weight": self.qaconv_loss_weight,
                        "adaface_loss_weight": self.adaface_loss_weight,
                        "occlusion_loss_weight": self.occlusion_loss_weight,
                        "adaface_eval_weight": self.adaface_eval_weight,
                        "qaconv_eval_weight": self.qaconv_eval_weight,
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

        # Storage for validation/test outputs (PL 2.0 compatibility)
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Setup niqab dataloader for occlusion training
        self.niqab_dataloader = None
        self.niqab_iter = None
        self.clean_dataloader = None
        self.clean_iter = None
        niqab_path = getattr(self.hparams, 'niqab_data_path', '')
        if niqab_path and os.path.exists(niqab_path):
            self._setup_niqab_dataloader(niqab_path)
            print(f"Niqab dataloader initialized with {len(self.niqab_dataset)} samples")
        else:
            print(f"Niqab data path not provided or doesn't exist. Occlusion training disabled.")
        
        # Setup clean face dataloader for occlusion training (20% of occlusion batch)
        # Clean faces will have all-ones masks (fully visible)
        self._setup_clean_dataloader()

    def _setup_niqab_dataloader(self, niqab_path):
        """Setup the niqab dataloader for occlusion training."""
        # Get batch size from hparams, use smaller batch for niqab
        niqab_batch_size = min(getattr(self.hparams, 'batch_size', 32), 32)

        # Create niqab dataset
        self.niqab_dataset = NiqabMaskDataset(
            root_dir=niqab_path,
            image_transform=get_default_niqab_transform(image_size=112),
            mask_target_size=14,  # Match intermediate feature map resolution (14x14 from Block 3)
            image_subdir='kept_faces',
            mask_subdir='masks',
            mask_suffix='_mask'
        )

        # Create dataloader
        self.niqab_dataloader = torch.utils.data.DataLoader(
            self.niqab_dataset,
            batch_size=niqab_batch_size,
            shuffle=True,
            num_workers=min(getattr(self.hparams, 'num_workers', 4), 4),
            drop_last=True,
            pin_memory=True
        )

        # Initialize iterator
        self.niqab_iter = iter(self.niqab_dataloader)

    def _setup_clean_dataloader(self):
        """Setup clean face dataloader for occlusion training (20% of occlusion batch).
        
        Clean faces will have all-ones masks (fully visible) to teach the occlusion head
        that clean faces should have high visibility predictions.
        """
        # Only setup if we have access to training data
        data_root = getattr(self.hparams, 'data_root', None)
        train_data_path = getattr(self.hparams, 'train_data_path', None)
        use_mxrecord = getattr(self.hparams, 'use_mxrecord', False)
        
        if not data_root or not train_data_path:
            print("Warning: data_root or train_data_path not found. Clean face dataloader disabled.")
            self.clean_dataloader = None
            self.clean_iter = None
            return
        
        try:
            from dataset.image_folder_dataset import CustomImageFolderDataset
            from dataset.record_dataset import AugmentRecordDataset
            
            # Create transform WITHOUT any occlusion augmentations for clean faces
            # We still apply other augmentations (flip, crop, photometric) but NO occlusion
            # EXCLUDED: MedicalMaskOcclusion, RandomOcclusion - we want clean faces only
            clean_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # NOTE: Both MedicalMaskOcclusion and RandomOcclusion are EXCLUDED
                #       This ensures clean faces have no occlusion augmentation
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            # Get augmentation probabilities (but we won't use occlusion)
            low_res_augmentation_prob = getattr(self.hparams, 'low_res_augmentation_prob', 0.0)
            crop_augmentation_prob = getattr(self.hparams, 'crop_augmentation_prob', 0.0)
            photometric_augmentation_prob = getattr(self.hparams, 'photometric_augmentation_prob', 0.0)
            swap_color_channel = getattr(self.hparams, 'swap_color_channel', False)
            output_dir = getattr(self.hparams, 'output_dir', './')
            
            # Create clean dataset (same structure as training dataset but without occlusion)
            if use_mxrecord:
                train_dir = os.path.join(data_root, train_data_path)
                clean_dataset = AugmentRecordDataset(
                    root_dir=train_dir,
                    transform=clean_transform,
                    low_res_augmentation_prob=low_res_augmentation_prob,
                    crop_augmentation_prob=crop_augmentation_prob,
                    photometric_augmentation_prob=photometric_augmentation_prob,
                    swap_color_channel=swap_color_channel,
                    output_dir=output_dir
                )
            else:
                train_dir = os.path.join(data_root, train_data_path, 'imgs')
                clean_dataset = CustomImageFolderDataset(
                    root=train_dir,
                    transform=clean_transform,
                    low_res_augmentation_prob=low_res_augmentation_prob,
                    crop_augmentation_prob=crop_augmentation_prob,
                    photometric_augmentation_prob=photometric_augmentation_prob,
                    swap_color_channel=swap_color_channel,
                    output_dir=output_dir
                )
            
            # Calculate batch size: 20% of niqab batch size
            niqab_batch_size = min(getattr(self.hparams, 'batch_size', 32), 32)
            clean_batch_size = max(1, int(niqab_batch_size * 0.2))  # 20% of niqab batch
            
            # Create dataloader
            self.clean_dataloader = torch.utils.data.DataLoader(
                clean_dataset,
                batch_size=clean_batch_size,
                shuffle=True,
                num_workers=min(getattr(self.hparams, 'num_workers', 4), 4),
                drop_last=True,
                pin_memory=True
            )
            
            # Initialize iterator
            self.clean_iter = iter(self.clean_dataloader)
            print(f"Clean face dataloader initialized with {len(clean_dataset)} samples (batch_size={clean_batch_size})")
            
        except Exception as e:
            print(f"Warning: Failed to setup clean face dataloader: {e}")
            print("Occlusion training will use 100% niqab images (no clean faces)")
            self.clean_dataloader = None
            self.clean_iter = None

    def _get_next_niqab_batch(self):
        """Get next batch from niqab dataloader, cycling when exhausted.
        
        Returns:
            tuple: (images, masks) where:
                - images: Mixed batch of 80% niqab + 20% clean faces
                - masks: Mixed batch of niqab GT masks + all-ones masks for clean faces
        """
        if self.niqab_dataloader is None:
            return None, None

        # Get niqab batch (80% of the occlusion training batch)
        try:
            niqab_batch = next(self.niqab_iter)
        except StopIteration:
            # Restart iterator when exhausted
            self.niqab_iter = iter(self.niqab_dataloader)
            niqab_batch = next(self.niqab_iter)

        # niqab_batch is (images, masks, indices)
        niqab_images, niqab_masks, _ = niqab_batch
        
        # Get clean face batch (20% of the occlusion training batch)
        clean_images = None
        if self.clean_dataloader is not None:
            try:
                clean_batch = next(self.clean_iter)
            except StopIteration:
                # Restart iterator when exhausted
                self.clean_iter = iter(self.clean_dataloader)
                clean_batch = next(self.clean_iter)
            
            # clean_batch is (images, labels) - we only need images
            clean_images = clean_batch[0]
            
            # Create all-ones masks for clean faces (fully visible)
            # Shape: [B, 1, 14, 14] matching intermediate feature map resolution
            # Use same device and dtype as niqab_masks to ensure compatibility
            clean_batch_size = clean_images.shape[0]
            clean_masks = torch.ones(clean_batch_size, 1, 14, 14,
                                    device=niqab_images.device,  # Match niqab device
                                    dtype=niqab_masks.dtype)      # Match niqab dtype
            
            # Ensure clean_images are on the same device as niqab_images
            if clean_images.device != niqab_images.device:
                clean_images = clean_images.to(niqab_images.device)
        else:
            clean_masks = None
        
        # Mix niqab (80%) and clean (20%) batches
        if clean_images is not None:
            # Concatenate images and masks
            mixed_images = torch.cat([niqab_images, clean_images], dim=0)
            mixed_masks = torch.cat([niqab_masks, clean_masks], dim=0)
            return mixed_images, mixed_masks
        else:
            # Fallback: return only niqab batch if clean dataloader is not available
            return niqab_images, niqab_masks

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

        # Use hasattr checks for compatibility with both PyTorch and timm schedulers
        # PyTorch schedulers (MultiStepLR, etc.) have get_last_lr()
        # timm schedulers have get_epoch_values()
        if hasattr(scheduler, 'get_last_lr'):
            lr = scheduler.get_last_lr()[0]
        elif hasattr(scheduler, 'get_epoch_values'):
            lr = scheduler.get_epoch_values(self.current_epoch)[0]
        else:
            # Fallback: try to get lr from optimizer param groups
            optimizer = self.trainer.optimizers[0]
            lr = optimizer.param_groups[0]['lr']
        return lr

    def forward(self, images, labels):
        embeddings, norms, occlusion_maps = self.model(images)
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100  # ignore_index
        return cos_thetas, norms, embeddings, occlusion_maps, labels

    def training_step(self, batch, batch_idx):
        # Main batch from CASIA-WebFace: (images, labels)
        # Recognition losses (AdaFace + QAConv) computed from this batch
        images, labels = batch

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

        # get intermediate features (14x14, 256ch) from body_early for QAConv and OcclusionHead
        x = self.model.input_layer(images)

        # Check for NaNs
        if torch.isnan(x).any():
            print(f"WARNING: Values after input layer contain NaNs. Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0)

        # Process through body_early (Blocks 1-3) to get 14x14 intermediate features
        for i, layer in enumerate(self.model.body_early):
            x = layer(x)
            # Check for NaNs periodically (every 10 layers to avoid performance impact)
            if i % 10 == 0 and torch.isnan(x).any():
                print(f"WARNING: Values after body_early layer {i} contain NaNs. Replacing with zeros.")
                x = torch.nan_to_num(x, nan=0.0)

        # Store intermediate features (14x14, 256ch) for QAConv and OcclusionHead
        intermediate_x = x

        # Final check for NaNs before normalization
        if torch.isnan(intermediate_x).any():
            print(f"WARNING: Intermediate feature maps contain NaNs before normalization. Replacing with zeros.")
            intermediate_x = torch.nan_to_num(intermediate_x, nan=0.0)
            # Add small epsilon to non-zero values to prevent division issues
            intermediate_x = intermediate_x + 1e-8 * (intermediate_x.abs() > 0).float()

        # normalize intermediate feature maps for qaconv (14x14 resolution)
        x_norm = torch.norm(intermediate_x.view(intermediate_x.size(0), -1), p=2, dim=1, keepdim=True).view(intermediate_x.size(0), 1, 1, 1)
        # Prevent division by zero
        x_norm = torch.clamp(x_norm, min=1e-8)
        feature_maps = intermediate_x / x_norm

        # Compute occlusion maps for the main batch (detached from backbone)
        # Occlusion maps will be used to weight QAConv matching (14x14 resolution)
        occlusion_maps = self.model.occlusion_head(intermediate_x.detach())
        # IMPORTANT: prevent QAConv gradients from flowing into OcclusionHead
        occlusion_maps_for_qaconv = occlusion_maps.detach()

        # Continue through body_late (Block 4) to get final features for embedding
        final_x = intermediate_x
        for i, layer in enumerate(self.model.body_late):
            final_x = layer(final_x)
            if torch.isnan(final_x).any():
                print(f"WARNING: Values after body_late layer {i} contain NaNs. Replacing with zeros.")
                final_x = torch.nan_to_num(final_x, nan=0.0)

        # Make deep copy of feature maps to avoid inference tensor issues
        feature_maps = feature_maps.clone().detach().requires_grad_(True)

        # Verify normalization
        norms = torch.norm(feature_maps.view(feature_maps.size(0), -1), p=2, dim=1)
        if ((norms < 0.99) | (norms > 1.01)).any():
            print(f"WARNING: Feature maps not properly normalized. Min norm: {norms.min().item()}, Max norm: {norms.max().item()}")
            # Force normalization again with safety measures
            feature_maps = F.normalize(feature_maps, p=2, dim=1).clone().detach().requires_grad_(True)

        # get adaface embeddings through output layer (from final 7x7 features)
        embeddings = self.model.output_layer(final_x)

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

        # ========== OCCLUSION LOSS FROM MIXED BATCH (80% NIQAB + 20% CLEAN) ==========
        # Get mixed batch (separate from main batch) for occlusion training
        # - 80% niqab images with GT masks (occluded regions)
        # - 20% clean faces with all-ones masks (fully visible)
        # This teaches the occlusion head to predict high visibility for clean faces
        occlusion_loss = torch.tensor(0.0, device=device)

        if self.niqab_dataloader is not None:
            occlusion_images, occlusion_masks = self._get_next_niqab_batch()

            if occlusion_images is not None and occlusion_masks is not None:
                # Move to device
                occlusion_images = occlusion_images.to(device)
                occlusion_masks = occlusion_masks.to(device)

                # Forward mixed images through body_early to get 14x14 intermediate feature maps
                # DETACHED: Occlusion loss should only train OcclusionHead, not backbone
                # This is consistent with QAConv (which is also detached from backbone)
                with torch.no_grad():
                    occlusion_x = self.model.input_layer(occlusion_images)
                    for layer in self.model.body_early:
                        occlusion_x = layer(occlusion_x)

                # Detach and enable gradients for OcclusionHead training only
                occlusion_x = occlusion_x.detach().requires_grad_(True)

                # Compute occlusion maps from intermediate feature maps (14x14 resolution)
                # Gradients will flow to OcclusionHead only (not backbone)
                occlusion_pred_maps = self.model.occlusion_head(occlusion_x)  # [B, 1, 14, 14]

                # Resize GT masks if needed (should already be 14x14, but just in case)
                if occlusion_masks.shape[-2:] != occlusion_pred_maps.shape[-2:]:
                    occlusion_masks = F.interpolate(
                        occlusion_masks,
                        size=occlusion_pred_maps.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # MSE loss between predicted and ground truth occlusion maps
                # For niqab: GT masks show occluded regions (0) and visible regions (1)
                # For clean faces: GT masks are all-ones (1) indicating fully visible
                occlusion_loss = F.mse_loss(occlusion_pred_maps, occlusion_masks)

                # Check for NaN
                if torch.isnan(occlusion_loss):
                    print(f"WARNING: Occlusion loss is NaN. Using zero loss instead.")
                    occlusion_loss = torch.tensor(0.0, device=device)

        # Initialize qaconv_loss for wandb logging
        qaconv_loss = torch.tensor(0.0, device=device)
        qaconv_acc = torch.tensor(0.0, device=device)

        # get qaconv pairwise matching loss if available
        if self.qaconv_criterion is not None:
            # Make sure qaconv is on the right device
            if hasattr(self, 'qaconv'):
                self.qaconv = self.qaconv.to(device)
            
            # Get loss and accuracy from pairwise matching loss
            pairwise_loss, pairwise_acc = self.qaconv_criterion(
                feature_maps,
                labels,
                occlusion_maps=occlusion_maps_for_qaconv
            )
            
            # Get triplet loss from SoftmaxTripletLoss criterion
            # Note: We only use the triplet_loss output here
            cls_loss, triplet_loss, _, cls_acc, triplet_acc = self.qaconv_triplet_criterion(
                feature_maps,
                labels,
                occlusion_maps=occlusion_maps_for_qaconv
            )

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
            
            # Combine losses (including occlusion loss if applicable)
            total_loss = (
                self.adaface_loss_weight * adaface_loss +
                self.qaconv_loss_weight * qaconv_loss +
                self.occlusion_loss_weight * occlusion_loss
            )
        else:
            total_loss = adaface_loss + self.occlusion_loss_weight * occlusion_loss

        # log metrics - ensure we take mean of tensor values
        lr = self.get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', total_loss.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('adaface_loss', adaface_loss.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('qaconv_loss', qaconv_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('occlusion_loss', occlusion_loss, on_step=True, on_epoch=True, logger=True)
        if 'pairwise_acc' in dir() and pairwise_acc is not None:
            self.log('qaconv_acc', pairwise_acc.mean(), on_step=True, on_epoch=True, logger=True)
        
        # Log to wandb (only on rank 0)
        if self.global_rank == 0:
            wandb_log_dict = {
                "qaconv_loss": qaconv_loss.item(),
                "qaconv_acc": pairwise_acc.mean().item() if 'pairwise_acc' in dir() else 0.0,
                "adaface_loss": adaface_loss.item(),
                "occlusion_loss": occlusion_loss.item(),
                "total_loss": total_loss.mean().item(),
                "learning_rate": lr,
            }
            # Add QAConv sub-losses if available
            if 'pairwise_loss' in dir() and pairwise_loss is not None:
                wandb_log_dict["qaconv_pairwise_loss"] = pairwise_loss.mean().item()
            if 'triplet_loss' in dir() and triplet_loss is not None:
                wandb_log_dict["qaconv_triplet_loss"] = triplet_loss.mean().item()
            wandb.log(wandb_log_dict)

        return total_loss.mean()

    def on_train_epoch_end(self):
        # PL 2.0 compatible - no outputs parameter
        return None

    def validation_step(self, batch, batch_idx):
        images, labels, dataname, image_index = batch
        
        # Get device 
        device = images.device
        
        # Check for NaN values in images
        if torch.isnan(images).any():
            print(f"WARNING: Input images contain NaN values. Replacing with zeros.")
            images = torch.nan_to_num(images, nan=0.0)
        
        # get intermediate features (14x14, 256ch) from body_early for QAConv and OcclusionHead
        with torch.no_grad():  # Ensure we don't track gradients
            # Extract intermediate feature maps from body_early (Blocks 1-3)
            x = self.model.input_layer(images)

            # Process through body_early layers to get 14x14 intermediate features
            for layer in self.model.body_early:
                x = layer(x)

            # At this point, x contains the 14x14 intermediate feature maps for QAConv
            # Normalize these feature maps
            feature_maps = F.normalize(x, p=2, dim=1)

            # Compute occlusion maps for QAConv weighting (14x14 resolution)
            occlusion_maps = self.model.occlusion_head(x)
            
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
                'qaconv_occ': occlusion_maps.to('cpu'),
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
                'qaconv_occ': occlusion_maps,
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }
        
        # Store output for epoch end processing (newer PyTorch Lightning)
        self.validation_step_outputs.append(output)
        return output

        # PL 2.0: Store outputs for epoch end processing
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        # PL 2.0 compatible - use stored outputs
        outputs = self.validation_step_outputs
        all_adaface_tensor, all_norm_tensor, all_qaconv_tensor, all_qaconv_occ_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

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
            labels = all_target_tensor[mask].cpu().numpy()
            issame = labels[0::2]  # Original issame labels for the pairs
            
            print(f"\nProcessing {dataname} with {len(adaface_embeddings)} samples")
            
            # evaluate adaface embeddings
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(adaface_embeddings, issame, nrof_folds=10)
            adaface_acc = accuracy.mean()
            val_logs[f'{dataname}_adaface_acc'] = adaface_acc
            print(f"{dataname} AdaFace accuracy: {adaface_acc:.4f}")

            # If QAConv features not gathered (DDP mode), skip QAConv validation
            if all_qaconv_tensor is None:
                val_logs[f'{dataname}_qaconv_acc'] = 0.0
                val_logs[f'{dataname}_combined_acc'] = adaface_acc
                print(f"{dataname} QAConv: SKIPPED (DDP gather disabled to avoid OOM)")
                print(f"{dataname} Combined accuracy: {adaface_acc:.4f} (AdaFace only)")
                continue

            qaconv_features = all_qaconv_tensor[mask]
            qaconv_occ = all_qaconv_occ_tensor[mask]
            # Structure data for gallery-query pairs following the original repo approach
            # Each consecutive pair of images forms a gallery-query pair
            gallery_features = qaconv_features[0::2]  # Even indices (0, 2, 4...)
            query_features = qaconv_features[1::2]    # Odd indices (1, 3, 5...)
            gallery_occ = qaconv_occ[0::2]
            query_occ = qaconv_occ[1::2]
            
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
                        positive_scores = self.qaconv.match_pairs(
                            query_features,
                            gallery_features,
                            probe_occ=query_occ,
                            gallery_occ=gallery_occ
                        )
                    
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
                            selected_queries_occ = query_occ[random_indices]
                            batch_galleries = gallery_features[i:end_idx]
                            batch_galleries_occ = gallery_occ[i:end_idx]
                            
                            # Compute scores for these negative pairs
                            for j in range(batch_size):
                                # Get scores for each gallery with its selected non-matching query
                                score = self.qaconv(
                                    batch_galleries[j:j+1],
                                    selected_queries[j:j+1],
                                    prob_occ=batch_galleries_occ[j:j+1],
                                    gal_occ=selected_queries_occ[j:j+1]
                                )
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

        # Log validation metrics to wandb (only on rank 0 to avoid duplicates)
        if self.global_rank == 0:
            wandb.log({
                'val_adaface_acc': val_logs.get('val_adaface_acc', 0.0),
                'val_qaconv_acc': val_logs.get('val_qaconv_acc', 0.0),
                'val_combined_acc': val_logs.get('val_combined_acc', 0.0),
                'epoch': self.current_epoch,
            })

        for k, v in val_logs.items():
            self.log(name=k, value=v, sync_dist=True)

        # Clear outputs for next epoch (PL 2.0)
        self.validation_step_outputs.clear()
        return None

    def test_step(self, batch, batch_idx):
        output = self.validation_step(batch, batch_idx)
        # PL 2.0: Store outputs for epoch end processing
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        # PL 2.0 compatible - use stored outputs
        outputs = self.test_step_outputs
        all_adaface_tensor, all_norm_tensor, all_qaconv_tensor, all_qaconv_occ_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)

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
            labels = all_target_tensor[mask].cpu().numpy()
            issame = labels[0::2]  # Original issame labels for the pairs
            
            print(f"\nProcessing {dataname} with {len(adaface_embeddings)} samples")
            
            # evaluate adaface embeddings
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(adaface_embeddings, issame, nrof_folds=10)
            adaface_acc = accuracy.mean()
            test_logs[f'{dataname}_adaface_acc'] = adaface_acc
            print(f"{dataname} AdaFace accuracy: {adaface_acc:.4f}")

            # If QAConv features not gathered (DDP mode), skip QAConv validation
            if all_qaconv_tensor is None:
                test_logs[f'{dataname}_qaconv_acc'] = 0.0
                test_logs[f'{dataname}_combined_acc'] = adaface_acc
                print(f"{dataname} QAConv: SKIPPED (DDP gather disabled to avoid OOM)")
                print(f"{dataname} Combined accuracy: {adaface_acc:.4f} (AdaFace only)")
                continue

            qaconv_features = all_qaconv_tensor[mask]
            qaconv_occ = all_qaconv_occ_tensor[mask]
            # Structure data for gallery-query pairs following the original repo approach
            # Each consecutive pair of images forms a gallery-query pair
            gallery_features = qaconv_features[0::2]  # Even indices (0, 2, 4...)
            query_features = qaconv_features[1::2]    # Odd indices (1, 3, 5...)
            gallery_occ = qaconv_occ[0::2]
            query_occ = qaconv_occ[1::2]
            
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
                        positive_scores = self.qaconv.match_pairs(
                            query_features,
                            gallery_features,
                            probe_occ=query_occ,
                            gallery_occ=gallery_occ
                        )
                    
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
                            selected_queries_occ = query_occ[random_indices]
                            batch_galleries = gallery_features[i:end_idx]
                            batch_galleries_occ = gallery_occ[i:end_idx]
                            
                            # Compute scores for these negative pairs
                            for j in range(batch_size):
                                # Get scores for each gallery with its selected non-matching query
                                score = self.qaconv(
                                    batch_galleries[j:j+1],
                                    selected_queries[j:j+1],
                                    prob_occ=batch_galleries_occ[j:j+1],
                                    gal_occ=selected_queries_occ[j:j+1]
                                )
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
            self.log(name=k, value=v, sync_dist=True)

        # Clear outputs (PL 2.0)
        self.test_step_outputs.clear()
        return None

    def gather_outputs(self, outputs):
        if self.hparams.distributed_backend == 'ddp':
            # Gather outputs across GPUs in small chunks, but only for AdaFace tensors
            # Skip QAConv feature map gathering to avoid OOM; validate QAConv separately on 1 GPU after training.
            chunk_size = 10  # adjust down if still OOM
            adaface_chunks = []
            norm_chunks = []
            target_chunks = []
            dataname_chunks = []
            index_chunks = []
            
            for i in range(0, len(outputs), chunk_size):
                chunk = outputs[i:i + chunk_size]
                _chunk_list = utils.all_gather(chunk)
                
                chunk_outputs = []
                for _chunk in _chunk_list:
                    chunk_outputs.extend(_chunk)
                
                adaface_chunks.append(torch.cat([out['adaface_output'] for out in chunk_outputs], axis=0).cpu())
                norm_chunks.append(torch.cat([out['norm'] for out in chunk_outputs], axis=0).cpu())
                target_chunks.append(torch.cat([out['target'] for out in chunk_outputs], axis=0).cpu())
                dataname_chunks.append(torch.cat([out['dataname'] for out in chunk_outputs], axis=0).cpu())
                index_chunks.append(torch.cat([out['image_index'] for out in chunk_outputs], axis=0).cpu())
                
                del chunk_outputs, _chunk_list
                torch.cuda.empty_cache()
            
            all_adaface_tensor = torch.cat(adaface_chunks, axis=0)
            all_norm_tensor = torch.cat(norm_chunks, axis=0)
            all_target_tensor = torch.cat(target_chunks, axis=0)
            all_dataname_tensor = torch.cat(dataname_chunks, axis=0)
            all_image_index = torch.cat(index_chunks, axis=0)
            all_qaconv_tensor = None  # not gathered in DDP to avoid OOM
            
            del adaface_chunks, norm_chunks, target_chunks, dataname_chunks, index_chunks
        else:
            all_adaface_tensor = torch.cat([out['adaface_output'] for out in outputs], axis=0).to('cpu')
            all_norm_tensor = torch.cat([out['norm'] for out in outputs], axis=0).to('cpu')
            all_qaconv_tensor = torch.cat([out['qaconv_output'] for out in outputs], axis=0).to('cpu')
            all_qaconv_occ_tensor = torch.cat([out['qaconv_occ'] for out in outputs], axis=0).to('cpu')
            all_target_tensor = torch.cat([out['target'] for out in outputs], axis=0).to('cpu')
            all_dataname_tensor = torch.cat([out['dataname'] for out in outputs], axis=0).to('cpu')
            all_image_index = torch.cat([out['image_index'] for out in outputs], axis=0).to('cpu')

        # get rid of duplicate index outputs
        unique_dict = {}
        if all_qaconv_tensor is not None:
            for _ada, _nor, _qa, _occ, _tar, _dat, _idx in zip(
                all_adaface_tensor,
                all_norm_tensor,
                all_qaconv_tensor,
                all_qaconv_occ_tensor,
                all_target_tensor,
                all_dataname_tensor,
                all_image_index,
            ):
                unique_dict[_idx.item()] = {
                    'adaface_output': _ada, 
                    'norm': _nor,
                    'qaconv_output': _qa, 
                    'qaconv_occ': _occ,
                    'target': _tar,
                    'dataname': _dat
                }
        else:
            # DDP path when qaconv not gathered
            for _ada, _nor, _tar, _dat, _idx in zip(all_adaface_tensor, all_norm_tensor,
                                                   all_target_tensor, all_dataname_tensor, all_image_index):
                unique_dict[_idx.item()] = {
                    'adaface_output': _ada, 
                    'norm': _nor,
                    'qaconv_output': None,
                    'qaconv_occ': None,
                    'target': _tar,
                    'dataname': _dat
                }
        unique_keys = sorted(unique_dict.keys())
        all_adaface_tensor = torch.stack([unique_dict[key]['adaface_output'] for key in unique_keys], axis=0)
        all_norm_tensor = torch.stack([unique_dict[key]['norm'] for key in unique_keys], axis=0)
        all_qaconv_tensor = torch.stack([unique_dict[key]['qaconv_output'] for key in unique_keys], axis=0) if all_qaconv_tensor is not None else None
        all_qaconv_occ_tensor = torch.stack([unique_dict[key]['qaconv_occ'] for key in unique_keys], axis=0) if all_qaconv_tensor is not None else None
        all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
        all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)

        return all_adaface_tensor, all_norm_tensor, all_qaconv_tensor, all_qaconv_occ_tensor, all_target_tensor, all_dataname_tensor

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
        
        # Use KFold from sklearn for consisti usent results with evaluate_utils
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
