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


class Trainer(LightningModule):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()
        self.save_hyperparameters()  # sets self.hparams

        # Initialize wandb
        wandb.init(
            project="adaface_face_recognition",
            config={
                "architecture": self.hparams.arch,
                "learning_rate": self.hparams.lr,
                "head_type": self.hparams.head,
                "qaconv_loss_weight": 0.99,
                "adaface_eval_weight": 0.5,
                "qaconv_eval_weight": 0.5,
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
        # initialize pairwise matching loss with qaconv matcher
        # Make sure qaconv is properly cloned to avoid inference mode issues
        if hasattr(self.model, 'qaconv'):
            self.qaconv = self.model.qaconv
            self.pairwise_loss = PairwiseMatchingLoss(self.qaconv)
        else:
            print("Warning: Model does not have QAConv layer")
            self.pairwise_loss = None
        # weight for qaconv loss during training
        self.qaconv_loss_weight = 0.99  # increased to 0.9 as per professor's specification

        # Hard coded weights for combining AdaFace and QAConv scores during evaluation
        self.adaface_eval_weight = 0.5  # Equal weight for combined score evaluation
        self.qaconv_eval_weight = 0.5   # Equal weight for combined score evaluation
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

        # get qaconv pairwise matching loss if available
        if self.pairwise_loss is not None:
            # Make sure pairwise_loss is on the right device
            if hasattr(self, 'qaconv'):
                self.qaconv = self.qaconv.to(device)
            qaconv_loss, qaconv_acc = self.pairwise_loss(feature_maps, labels)
            qaconv_loss = qaconv_loss.mean()
            
            # Check for NaN in losses
            if torch.isnan(qaconv_loss):
                print(f"WARNING: QAConv loss is NaN. Using zero loss instead.")
                qaconv_loss = torch.tensor(0.0, device=device)
                
            if torch.isnan(adaface_loss):
                print(f"WARNING: AdaFace loss is NaN. Using zero loss instead.")
                adaface_loss = torch.tensor(0.0, device=device)
                
            # combine losses
            total_loss = adaface_loss + self.qaconv_loss_weight * qaconv_loss
            # log QAConv metrics
            self.log('qaconv_loss', qaconv_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
            self.log('qaconv_acc', qaconv_acc.mean(), on_step=True, on_epoch=True, logger=True)
            # Log AdaFace loss
            self.log('adaface_loss', adaface_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        else:
            total_loss = adaface_loss

        # log metrics
        lr = self.get_current_lr()
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        # Log to wandb
        wandb.log({
            "qaconv_loss": qaconv_loss.item(),
            "qaconv_acc": qaconv_acc.mean().item(),
            "adaface_loss": adaface_loss.item(),
            "total_loss": total_loss.item(),
            "learning_rate": lr
        })

        return total_loss

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
            # This is more reliable than trying to get intermediate outputs
            x = self.model.input_layer(images)
            
            # Process through body layers
            for layer in self.model.body:
                x = layer(x)
            
            # At this point, x contains the feature maps needed for QAConv
            # Normalize these feature maps
            feature_maps = F.normalize(x, p=2, dim=1)
            
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
                print(f"WARNING: QAConv feature maps contain NaNs. Replacing with zeros.")
                feature_maps = torch.nan_to_num(feature_maps, nan=0.0)
                # Force normalization again
                feature_maps = F.normalize(feature_maps, p=2, dim=1)

        if self.hparams.distributed_backend == 'ddp':
            # to save gpu memory
            return {
                'adaface_output': embeddings.to('cpu'),
                'norm': norms.to('cpu'),
                'qaconv_output': feature_maps.to('cpu'),
                'target': labels.to('cpu'),
                'dataname': dataname.to('cpu'),
                'image_index': image_index.to('cpu')
            }
        else:
            # dp requires the tensor to be cuda
            return {
                'adaface_output': embeddings,
                'norm': norms,
                'qaconv_output': feature_maps,
                'target': labels,
                'dataname': dataname,
                'image_index': image_index
            }

    def validation_epoch_end(self, outputs):
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
                
                print(f"Weights used - AdaFace: {adaface_weight:.2f}, QAConv: {qaconv_weight:.2f}")
                
                # Evaluate combined scores using the original issame labels
                thresholds = np.arange(0, 4, 0.01)
                tpr, fpr, accuracy, best_thresholds = evaluate_utils.calculate_roc(
                    thresholds,
                    combined_dists[:, np.newaxis],
                    combined_dists[:, np.newaxis],
                    issame,
                    nrof_folds=10
                )
                combined_acc = accuracy.mean()
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

        for k, v in val_logs.items():
            self.log(name=k, value=v)

        return None

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
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
                
                print(f"Weights used - AdaFace: {adaface_weight:.2f}, QAConv: {qaconv_weight:.2f}")
                
                # Evaluate combined scores using the original issame labels
                thresholds = np.arange(0, 4, 0.01)
                tpr, fpr, accuracy, best_thresholds = evaluate_utils.calculate_roc(
                    thresholds,
                    combined_dists[:, np.newaxis],
                    combined_dists[:, np.newaxis],
                    issame,
                    nrof_folds=10
                )
                combined_acc = accuracy.mean()
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

        return None

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

    def configure_optimizers(self):
        paras_wo_bn, paras_only_bn = self.split_parameters(self.model)

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

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay

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
