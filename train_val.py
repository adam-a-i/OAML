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
# Import TransMatcher components directly to avoid module overwriting
from transmatcher import TransMatcher, TransformerDecoder, TransformerDecoderLayer

__all__ = ['Trainer']  # Explicitly declare what should be exposed

class Trainer(LightningModule):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        
        # Define weight variables for AdaFace and TransMatcher loss
        self.adaface_loss_weight = 0.1
        self.transmatcher_loss_weight = 0.9
        
        # Initialize wandb
        wandb.init(
            project="adaface_face_recognition",
                config={
                    "architecture": self.hparams.arch,
                    "learning_rate": self.hparams.lr,
                    "head_type": self.hparams.head,
                    "adaface_loss_weight": self.adaface_loss_weight,
                    "transmatcher_loss_weight": self.transmatcher_loss_weight,
                    "epochs": self.hparams.epochs if hasattr(self.hparams, 'epochs') else None,
                    "batch_size": self.hparams.batch_size if hasattr(self.hparams, 'batch_size') else None,
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

        if isinstance(scheduler, lr_scheduler._LRScheduler):
            lr = scheduler.get_last_lr()[0]
        else:
            lr = scheduler.get_epoch_values(self.current_epoch)[0]
        return lr

    def forward(self, images, labels):
        embeddings, norms = self.model(images)
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100  # ignore_index
        return cos_thetas, norms, embeddings, labels

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Create TransMatcher loss here instead of in __init__
        transmatcher_loss = PairwiseMatchingLoss(self.model.transmatcher)
        
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
        # --- TransMatcher Pairwise Matching Loss ---
        # Extract feature maps for TransMatcher: [B, C, H, W] -> [B, H, W, C]
        feature_maps = x.permute(0, 2, 3, 1).contiguous()
        transmatcher_loss_val, transmatcher_acc = transmatcher_loss(feature_maps, labels)
        total_loss = self.adaface_loss_weight * adaface_loss + self.transmatcher_loss_weight * transmatcher_loss_val.mean()
        # log metrics
        lr = self.get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', total_loss.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('adaface_loss', adaface_loss.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('transmatcher_loss', transmatcher_loss_val.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('transmatcher_acc', transmatcher_acc.mean(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        wandb.log({
            "adaface_loss": adaface_loss.item(),
            "transmatcher_loss": transmatcher_loss_val.mean().item(),
            "transmatcher_acc": transmatcher_acc.mean().item(),
            "total_loss": total_loss.mean().item(),
            "learning_rate": lr,
        })
        return total_loss.mean()

    def training_epoch_end(self, outputs):
        return None

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
            x = self.model.input_layer(images)
            for layer in self.model.body:
                x = layer(x)
            
            # Ensure feature maps are properly shaped for TransMatcher
            # TransMatcher expects [B, H, W, C] format where H*W = seq_len
            B, C, H, W = x.shape
            
            # Keep in [B, H, W, C] format that TransMatcher expects
            feature_maps = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            
            # Verify dimensions match TransMatcher's expectations
            assert H * W == self.model.transmatcher.seq_len, f"H*W ({H*W}) != seq_len ({self.model.transmatcher.seq_len})"
            assert C == self.model.transmatcher.d_model, f"C ({C}) != d_model ({self.model.transmatcher.d_model})"
            
            # get adaface embeddings with flip augmentation
            embeddings, norms = self.model(images)
            fliped_images = torch.flip(images, dims=[3])
            flipped_embeddings, flipped_norms = self.model(fliped_images)
            stacked_embeddings = torch.stack([embeddings, flipped_embeddings], dim=0)
            stacked_norms = torch.stack([norms, flipped_norms], dim=0)
            embeddings, norms = utils.fuse_features_with_norm(stacked_embeddings, stacked_norms)
            
            # Final check for NaNs
            if torch.isnan(embeddings).any():
                print(f"WARNING: AdaFace embeddings contain NaNs. Replacing with zeros.")
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
            if torch.isnan(feature_maps).any():
                print(f"WARNING: TransMatcher feature maps contain NaNs. Replacing with zeros.")
                feature_maps = torch.nan_to_num(feature_maps, nan=0.0)
            
        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            return {
                'adaface_output': embeddings.to('cpu'),
                'norm': norms.to('cpu'),
                'transmatcher_output': feature_maps.to('cpu'),
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            return {
                'adaface_output': embeddings,
                'norm': norms,
                'transmatcher_output': feature_maps,
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }

    def validation_epoch_end(self, outputs):
        all_adaface_tensor, all_norm_tensor, all_transmatcher_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)
        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        val_logs = {}
        adaface_accs = []
        transmatcher_accs = []
        combined_accs = []
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            mask = all_dataname_tensor == dataname_idx
            adaface_embeddings = all_adaface_tensor[mask].cpu().numpy()
            transmatcher_features = all_transmatcher_tensor[mask]
            labels = all_target_tensor[mask].cpu().numpy()
            print(f"\nProcessing {dataname} with {len(adaface_embeddings)} samples")
            
            # --- AdaFace Similarity Matrix ---
            adaface_emb = torch.tensor(adaface_embeddings)
            adaface_emb = torch.nn.functional.normalize(adaface_emb, p=2, dim=1)
            adaface_sim = torch.matmul(adaface_emb, adaface_emb.t()).cpu().numpy()
            
            # --- TransMatcher Similarity Matrix ---
            N = transmatcher_features.shape[0]
            transmatcher_sim_tensor = torch.zeros((N, N), dtype=transmatcher_features.dtype, device=transmatcher_features.device)
            batch_size = 256
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    end_i = min(i + batch_size, N)
                    query_batch = transmatcher_features[i:end_i]
                    for j in range(0, N, batch_size):
                        end_j = min(j + batch_size, N)
                        gallery_batch = transmatcher_features[j:end_j]
                        scores = self.model.transmatcher(query_batch, gallery_batch)
                        transmatcher_sim_tensor[i:end_i, j:end_j] = scores
            transmatcher_sim = transmatcher_sim_tensor.cpu().numpy()
            
            # --- Normalize similarity matrices to [0, 1] ---
            adaface_sim_norm = (adaface_sim - adaface_sim.min()) / (adaface_sim.max() - adaface_sim.min() + 1e-8)
            transmatcher_sim_norm = (transmatcher_sim - transmatcher_sim.min()) / (transmatcher_sim.max() - transmatcher_sim.min() + 1e-8)
            
            # --- Combined Similarity Matrix ---
            combined_sim = self.adaface_loss_weight * adaface_sim_norm + self.transmatcher_loss_weight * transmatcher_sim_norm
            
            # --- Compute accuracy for each method ---
            n = len(labels)
            pids = labels
            label_matrix = pids[:, None] == pids[None, :]
            
            # Get upper triangle indices (excluding diagonal)
            indices = np.triu_indices(n, k=1)
            issame_pairs = label_matrix[indices]
            
            # Get scores for all pairs
            adaface_scores = adaface_sim[indices]
            transmatcher_scores = transmatcher_sim[indices]
            combined_scores = combined_sim[indices]
            
            # Calculate accuracy using ROC curve
            def calculate_accuracy(scores, issame):
                # Get thresholds
                thresholds = np.arange(0, 1, 0.01)
                accuracies = []
                
                for threshold in thresholds:
                    # Calculate true positives and true negatives
                    tp = np.sum((scores[issame] > threshold))
                    tn = np.sum((scores[~issame] <= threshold))
                    
                    # Calculate accuracy
                    accuracy = (tp + tn) / len(scores)
                    accuracies.append(accuracy)
                
                # Return best accuracy
                return np.max(accuracies)
            
            # Calculate accuracies
            adaface_acc = calculate_accuracy(adaface_scores, issame_pairs)
            transmatcher_acc = calculate_accuracy(transmatcher_scores, issame_pairs)
            combined_acc = calculate_accuracy(combined_scores, issame_pairs)
            
            val_logs[f'{dataname}_adaface_acc'] = adaface_acc
            adaface_accs.append(adaface_acc)
            val_logs[f'{dataname}_transmatcher_acc'] = transmatcher_acc
            transmatcher_accs.append(transmatcher_acc)
            val_logs[f'{dataname}_combined_acc'] = combined_acc
            combined_accs.append(combined_acc)
            
            print(f"{dataname} AdaFace accuracy: {adaface_acc:.4f}")
            print(f"{dataname} TransMatcher accuracy: {transmatcher_acc:.4f}")
            print(f"{dataname} Combined accuracy: {combined_acc:.4f}")
            val_logs[f'{dataname}_num_val_samples'] = len(adaface_embeddings)
            
        # Mean accuracy across datasets
        val_logs['val_adaface_acc'] = np.mean(adaface_accs) if adaface_accs else 0.0
        val_logs['val_transmatcher_acc'] = np.mean(transmatcher_accs) if transmatcher_accs else 0.0
        val_logs['val_combined_acc'] = np.mean(combined_accs) if combined_accs else 0.0
        val_logs['val_acc'] = val_logs['val_combined_acc']
        val_logs['epoch'] = self.current_epoch
        
        # Log to wandb
        wandb.log({
            'val_adaface_acc': val_logs.get('val_adaface_acc', 0.0),
            'val_transmatcher_acc': val_logs.get('val_transmatcher_acc', 0.0),
            'val_combined_acc': val_logs.get('val_combined_acc', 0.0),
            'epoch': self.current_epoch,
        })
        
        for k, v in val_logs.items():
            self.log(name=k, value=v)
        return None

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        # Same as validation_epoch_end, but for test set
        all_adaface_tensor, all_norm_tensor, all_transmatcher_tensor, all_target_tensor, all_dataname_tensor = self.gather_outputs(outputs)
        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
        test_logs = {}
        adaface_accs = []
        transmatcher_accs = []
        combined_accs = []
        for dataname_idx in all_dataname_tensor.unique():
            dataname = idx_to_dataname[dataname_idx.item()]
            mask = all_dataname_tensor == dataname_idx
            adaface_embeddings = all_adaface_tensor[mask].cpu().numpy()
            transmatcher_features = all_transmatcher_tensor[mask]
            labels = all_target_tensor[mask].cpu().numpy()
            issame = labels[0::2]  # Original issame labels for the pairs
            print(f"\nProcessing {dataname} with {len(adaface_embeddings)} samples")
            # --- AdaFace Similarity Matrix ---
            adaface_emb = torch.tensor(adaface_embeddings)
            adaface_emb = torch.nn.functional.normalize(adaface_emb, p=2, dim=1)
            adaface_sim = torch.matmul(adaface_emb, adaface_emb.t()).cpu().numpy()
            # --- TransMatcher Similarity Matrix ---
            N = transmatcher_features.shape[0]
            # Initialize on device to avoid costly CPU-GPU transfers in loop
            transmatcher_sim_tensor = torch.zeros((N, N), dtype=transmatcher_features.dtype, device=transmatcher_features.device)
            batch_size = 256 # Increased batch size for faster computation
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    end_i = min(i + batch_size, N)
                    query_batch = transmatcher_features[i:end_i]
                    for j in range(0, N, batch_size):
                        end_j = min(j + batch_size, N)
                        gallery_batch = transmatcher_features[j:end_j]
                        scores = self.model.transmatcher(query_batch, gallery_batch)
                        transmatcher_sim_tensor[i:end_i, j:end_j] = scores
            # Convert to numpy only once after all computations are done on device
            transmatcher_sim = transmatcher_sim_tensor.cpu().numpy()
            
            # --- Normalize both similarity matrices to [0, 1] ---
            adaface_sim_norm = (adaface_sim - adaface_sim.min()) / (adaface_sim.max() - adaface_sim.min() + 1e-8)
            transmatcher_sim_norm = (transmatcher_sim - transmatcher_sim.min()) / (transmatcher_sim.max() - transmatcher_sim.min() + 1e-8)
            # --- Combined Similarity Matrix ---
            combined_sim = self.adaface_loss_weight * adaface_sim_norm + self.transmatcher_loss_weight * transmatcher_sim_norm
            # --- Compute accuracy for each method ---
            n = len(labels)
            pids = labels
            label_matrix = pids[:, None] == pids[None, :]
            indices = np.triu_indices(n, k=1)
            issame_pairs = label_matrix[indices]
            # AdaFace accuracy
            adaface_scores = adaface_sim[indices]
            # Calculate accuracy for genuine and impostor pairs separately
            adaface_acc_genuine = np.mean(adaface_scores[issame_pairs] > adaface_scores[~issame_pairs].mean()) if issame_pairs.sum() > 0 else 0.0
            adaface_acc_impostor = np.mean(adaface_scores[~issame_pairs] < adaface_scores[issame_pairs].mean()) if (~issame_pairs).sum() > 0 else 0.0
            adaface_acc = (adaface_acc_genuine + adaface_acc_impostor) / 2.0
            test_logs[f'{dataname}_adaface_acc'] = adaface_acc
            adaface_accs.append(adaface_acc)
            # TransMatcher accuracy
            transmatcher_scores = transmatcher_sim[indices]
            # Calculate accuracy for genuine and impostor pairs separately
            transmatcher_acc_genuine = np.mean(transmatcher_scores[issame_pairs] > transmatcher_scores[~issame_pairs].mean()) if issame_pairs.sum() > 0 else 0.0
            transmatcher_acc_impostor = np.mean(transmatcher_scores[~issame_pairs] < transmatcher_scores[issame_pairs].mean()) if (~issame_pairs).sum() > 0 else 0.0
            transmatcher_acc = (transmatcher_acc_genuine + transmatcher_acc_impostor) / 2.0
            test_logs[f'{dataname}_transmatcher_acc'] = transmatcher_acc
            transmatcher_accs.append(transmatcher_acc)
            # Combined accuracy
            combined_scores = combined_sim[indices]
            # Calculate accuracy for genuine and impostor pairs separately
            combined_acc_genuine = np.mean(combined_scores[issame_pairs] > combined_scores[~issame_pairs].mean()) if issame_pairs.sum() > 0 else 0.0
            combined_acc_impostor = np.mean(combined_scores[~issame_pairs] < combined_scores[issame_pairs].mean()) if (~issame_pairs).sum() > 0 else 0.0
            combined_acc = (combined_acc_genuine + combined_acc_impostor) / 2.0
            test_logs[f'{dataname}_combined_acc'] = combined_acc
            combined_accs.append(combined_acc)
            print(f"{dataname} AdaFace accuracy: {adaface_acc:.4f}")
            print(f"{dataname} TransMatcher accuracy: {transmatcher_acc:.4f}")
            print(f"{dataname} Combined accuracy: {combined_acc:.4f}")
            test_logs[f'{dataname}_num_test_samples'] = len(adaface_embeddings)
        test_logs['test_adaface_acc'] = np.mean(adaface_accs) if adaface_accs else 0.0
        test_logs['test_transmatcher_acc'] = np.mean(transmatcher_accs) if transmatcher_accs else 0.0
        test_logs['test_combined_acc'] = np.mean(combined_accs) if combined_accs else 0.0
        test_logs['test_acc'] = test_logs['test_combined_acc']
        test_logs['epoch'] = self.current_epoch
        # Log to wandb after each test epoch
        wandb.log({
            'test_adaface_acc': test_logs.get('test_adaface_acc', 0.0),
            'test_transmatcher_acc': test_logs.get('test_transmatcher_acc', 0.0),
            'test_combined_acc': test_logs.get('test_combined_acc', 0.0),
            'epoch': self.current_epoch,
        })
        for k, v in test_logs.items():
            self.log(name=k, value=v)
        return None

    def gather_outputs(self, outputs):
        if self.hparams.distributed_backend == 'ddp':
            outputs_list = []
            _outputs_list = utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs
        all_adaface_tensor = torch.cat([out['adaface_output'] for out in outputs_list], axis=0).to('cpu')
        all_norm_tensor = torch.cat([out['norm'] for out in outputs_list], axis=0).to('cpu')
        all_transmatcher_tensor = torch.cat([out['transmatcher_output'] for out in outputs_list], axis=0).to('cpu')
        all_target_tensor = torch.cat([out['target'] for out in outputs_list], axis=0).to('cpu')
        all_dataname_tensor = torch.cat([out['dataname'] for out in outputs_list], axis=0).to('cpu')
        all_image_index = torch.cat([out['image_index'] for out in outputs_list], axis=0).to('cpu')
        unique_dict = {}
        for _ada, _nor, _tm, _tar, _dat, _idx in zip(all_adaface_tensor, all_norm_tensor, all_transmatcher_tensor,
                                                   all_target_tensor, all_dataname_tensor, all_image_index):
            unique_dict[_idx.item()] = {
                'adaface_output': _ada, 
                'norm': _nor,
                'transmatcher_output': _tm, 
                'target': _tar,
                'dataname': _dat
            }
        unique_keys = sorted(unique_dict.keys())
        all_adaface_tensor = torch.stack([unique_dict[key]['adaface_output'] for key in unique_keys], axis=0)
        all_norm_tensor = torch.stack([unique_dict[key]['norm'] for key in unique_keys], axis=0)
        all_transmatcher_tensor = torch.stack([unique_dict[key]['transmatcher_output'] for key in unique_keys], axis=0)
        all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
        all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)
        return all_adaface_tensor, all_norm_tensor, all_transmatcher_tensor, all_target_tensor, all_dataname_tensor

    def split_parameters(self, module):
        """ Split parameters into with and without weight decay.
        Handles AdaFace backbone, head, and TransMatcher parameters only.
        """
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        # Add head kernel (for AdaFace head)
        if hasattr(self, 'head') and hasattr(self.head, 'kernel'):
            params_decay.append(self.head.kernel)
        return params_decay, params_no_decay

    def configure_optimizers(self):
        # Only AdaFace backbone, head, and TransMatcher parameters are handled
        paras_wo_bn, paras_only_bn = self.split_parameters(self.model)
        # Add TransMatcher parameters (if not already included)
        if hasattr(self.model, 'transmatcher'):
            tm_decay, tm_no_decay = self.split_parameters(self.model.transmatcher)
            paras_wo_bn.extend(tm_decay)
            paras_only_bn.extend(tm_no_decay)
        optimizer = optim.SGD([{
            'params': paras_wo_bn,
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
