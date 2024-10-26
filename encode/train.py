import copy
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import traceback

# Import models and their components
from model.CGMN_dataset import CFGDataset
from model.DeepDXF_dataset import DXFDataset
from model.layers.DenseGraphMatching import HierarchicalGraphMatchNetwork, EarlyStopping
from model.transformer_encoder import DXFTransformer
from model.DeepDXF_loss import ContrastiveLoss
from config.CGMN_cfg_config import cfg_args
from utils.CGMN_utils import create_dir_if_not_exists, write_log_file, generate_epoch_pair

class CombinedLoss(torch.nn.Module):
    def __init__(self, cgmn_weight=0.5, dxf_weight=0.5, batch_size=32, temperature=0.5):
        super().__init__()
        self.cgmn_weight = cgmn_weight
        self.dxf_weight = dxf_weight
        self.dxf_loss_fn = ContrastiveLoss(
            batch_size=batch_size,
            temperature=temperature,
            loss_type='simclr'
        )

    def compute_cgmn_loss(self, cgmn_outputs, cgmn_model):
        feature_p1 = cgmn_outputs["feature_p1"]
        feature_p2 = cgmn_outputs["feature_p2"]
        feature_h1 = cgmn_outputs["feature_h1"]
        feature_h2 = cgmn_outputs["feature_h2"]
        feature_p0 = cgmn_outputs["feature_p0"]
        feature_h0 = cgmn_outputs["feature_h0"]

        # print(f"Feature p1 shape: {feature_p1.shape}")
        # print(f"Feature p2 shape: {feature_p2.shape}")
        # print(f"Feature h1 shape: {feature_h1.shape}")
        # print(f"Feature h2 shape: {feature_h2.shape}")
        # Node information fusion
        feature_p1, feature_p2 = cgmn_model.matching_layer(feature_p1, feature_p2)
        feature_h1, feature_h2 = cgmn_model.matching_layer(feature_h1, feature_h2)

        # print(f"Feature p1 shape: {feature_p1.shape}")
        # print(f"Feature h0 shape: {feature_h0.shape}")
        feature_p1, _ = cgmn_model.matching_layer(feature_p1, feature_h0)
        feature_p2, _ = cgmn_model.matching_layer(feature_p2, feature_h0)
        feature_h1, _ = cgmn_model.matching_layer(feature_h1, feature_p0)
        feature_h2, _ = cgmn_model.matching_layer(feature_h2, feature_p0)


        # Calculate contrastive loss
        loss_p = cgmn_model.loss(feature_p1, feature_p2)
        loss_h = cgmn_model.loss(feature_h1, feature_h2)

        return (loss_p + loss_h) * 0.5

    def forward(self, cgmn_outputs, dxf_outputs, cgmn_model):
        # Compute CGMN loss
        # print(f"CGMN outputs shape: {cgmn_outputs['feature_p1'].shape}")
        cgmn_loss = self.compute_cgmn_loss(cgmn_outputs, cgmn_model)

        # Compute DXF loss using the dedicated loss function
        dxf_loss = self.dxf_loss_fn(dxf_outputs["proj_z1"], dxf_outputs["proj_z2"])
        # print(f"CGMN loss shape: {cgmn_loss.shape}, DXF loss shape: {dxf_loss.shape}")
        return {
            "cgmn_loss": self.cgmn_weight * cgmn_loss,
            "dxf_loss": self.dxf_weight * dxf_loss,
            "total_loss": self.cgmn_weight * cgmn_loss + self.dxf_weight * dxf_loss
        }

class CombinedTrainer:
    def __init__(self, cfg_args, dxf_args):
        super(CombinedTrainer, self).__init__()
        # Move parameters initialization to the top
        self.max_epoch = cfg_args.epochs
        self.batch_size = cfg_args.batch_size  # Move this up
        self.lr = cfg_args.lr
        self.patience = cfg_args.patience

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Set up data directories
        self.cfg_data_dir = os.path.join(
            cfg_args.data_dir,
            f"{cfg_args.dataset}_{cfg_args.graph_init_dim}ACFG_min{cfg_args.graph_size_min}_max{cfg_args.graph_size_max}",
            f"acfgSSL_{cfg_args.graph_init_dim}"
        )
        print(f"CGMN data directory: {self.cfg_data_dir}")

        # Initialize models
        self.cgmn_model = HierarchicalGraphMatchNetwork(
            node_init_dims=cfg_args.graph_init_dim,
            arguments=cfg_args,
            device=self.device
        ).to(self.device)

        self.dxf_model = DXFTransformer().to(self.device)

        # Initialize combined loss with batch_size
        self.combined_loss = CombinedLoss(
            batch_size=self.batch_size,
            temperature=dxf_args.temperature
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.cgmn_model.parameters()},
            {'params': self.dxf_model.parameters()},
        ], lr=self.lr)

        self.earlystopping = EarlyStopping(self.patience, verbose=True)

        # Load datasets
        self.load_datasets(cfg_args, dxf_args)

    def load_datasets(self, cfg_args, dxf_args):
        # print("\nLoading datasets:")
        #
        # print(f"\nLoading CGMN dataset from: {self.cfg_data_dir}")
        cfg_dataset = CFGDataset(data_dir=self.cfg_data_dir, batch_size=self.batch_size)

        print(f"Training graphs: {len(cfg_dataset.graph_train)}")
        print(f"Training classes: {len(cfg_dataset.classes_train)}")
        print(f"Validation data: {len(cfg_dataset.valid_epoch) if cfg_dataset.valid_epoch else 0} batches")
        print(f"Test data: {len(cfg_dataset.test_epoch) if cfg_dataset.test_epoch else 0} batches")

        self.graph_train = cfg_dataset.graph_train
        self.classes_train = cfg_dataset.classes_train
        self.epoch_data_valid = cfg_dataset.valid_epoch
        self.epoch_data_test = cfg_dataset.test_epoch

        # # 打印第一个验证批次的数据
        # if self.epoch_data_valid:
        #     x1, x2, adj1, adj2, y = self.epoch_data_valid[0]
        #     print("\nFirst validation batch:")
        #     print(f"x1 shape: {x1.shape}")
        #     print(f"x2 shape: {x2.shape}")
        #     print(f"adj1 shape: {adj1.shape}")
        #     print(f"adj2 shape: {adj2.shape}")
        #     print(f"y values: {y}")

        # print(f"Loading DXF datasets from {dxf_args.data_dir}")
        dxf_datasets = []
        for filename in os.listdir(dxf_args.data_dir):
            if filename.endswith('.h5'):
                try:
                    dataset = DXFDataset(os.path.join(dxf_args.data_dir, filename))
                    dxf_datasets.append(dataset)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    continue

        if not dxf_datasets:
            raise ValueError("No valid DXF datasets found!")

        self.dxf_dataset = ConcatDataset(dxf_datasets)
        print(f"Total DXF samples: {len(self.dxf_dataset)}")

        self.dxf_dataloader = DataLoader(
            self.dxf_dataset,
            batch_size=self.batch_size,  # 使用相同的batch_size
            shuffle=True,
            drop_last=True
        )

    def train_one_epoch(self):
        """Train both models for one epoch"""
        self.cgmn_model.train()
        self.dxf_model.train()

        total_loss = 0.0
        num_batches = 0

        # Generate CGMN epoch data
        cgmn_epoch_data = generate_epoch_pair(
            self.graph_train,
            self.classes_train,
            self.batch_size
        )
        cgmn_perm = np.random.permutation(len(cgmn_epoch_data))

        # Iterate through both datasets
        for idx, dxf_batch in enumerate(self.dxf_dataloader):
            if idx >= len(cgmn_perm):
                break

            try:
                # Process CGMN batch
                cgmn_idx = cgmn_perm[idx]
                cgmn_batch = cgmn_epoch_data[cgmn_idx]

                x1, x2, adj1, adj2, y = cgmn_batch


                # Ensure consistent batch size
                current_batch_size = min(x1.shape[0], dxf_batch[0].size(0))
                if current_batch_size == 0:
                    continue

                # Truncate to consistent batch size
                x1 = x1[:current_batch_size]
                x2 = x2[:current_batch_size]
                adj1 = adj1[:current_batch_size]
                adj2 = adj2[:current_batch_size]
                y = y[:current_batch_size]

                # Process CGMN batch with cloned tensors
                feature_p_init = torch.FloatTensor(x1).to(self.device).clone()
                adj_p = torch.FloatTensor(adj1).to(self.device).clone()
                feature_h_init = torch.FloatTensor(x2).to(self.device).clone()
                adj_h = torch.FloatTensor(adj2).to(self.device).clone()

                # Get baseline features through GCN
                feature_p0 = self.cgmn_model(batch_x_p=feature_p_init, batch_adj_p=adj_p)
                feature_h0 = self.cgmn_model(batch_x_p=feature_h_init, batch_adj_p=adj_h)

                # Get augmented features with cloned inputs
                drop_feature_p1 = self.cgmn_model.drop_feature(feature_p_init.clone(), self.cgmn_model.args.drop_feature1)
                drop_feature_p2 = self.cgmn_model.drop_feature(feature_p_init.clone(), self.cgmn_model.args.drop_feature2)


                drop_edge_p1 = adj_p.clone()
                drop_edge_p2 = adj_p.clone()
                for i in range(adj_p.size()[0]):
                    drop_edge_p1[i] = self.cgmn_model.aug_random_edge(adj_p[i].cpu().numpy(), self.cgmn_model.args.drop_edge1)
                    drop_edge_p2[i] = self.cgmn_model.aug_random_edge(adj_p[i].cpu().numpy(), self.cgmn_model.args.drop_edge2)

                # Get model augmented outputs
                feature_p1 = self.cgmn_model(batch_x_p=drop_feature_p1, batch_adj_p=drop_edge_p1)  # [5,10,100]
                feature_p2 = self.cgmn_model(batch_x_p=drop_feature_p2, batch_adj_p=drop_edge_p2)  # [5,10,100]

                drop_feature_h1 = self.cgmn_model.drop_feature(feature_h_init, self.cgmn_model.args.drop_feature1)
                drop_feature_h2 = self.cgmn_model.drop_feature(feature_h_init, self.cgmn_model.args.drop_feature2)

                drop_edge_h1 = adj_h.clone()
                drop_edge_h2 = adj_h.clone()
                for i in range(adj_h.size()[0]):
                    drop_edge_h1[i] = self.cgmn_model.aug_random_edge(adj_h[i].cpu().numpy(), self.cgmn_model.args.drop_edge1)
                    drop_edge_h2[i] = self.cgmn_model.aug_random_edge(adj_h[i].cpu().numpy(), self.cgmn_model.args.drop_edge2)

                feature_h1 = self.cgmn_model(batch_x_p=drop_feature_h1, batch_adj_p=drop_edge_h1)  # [5,10,100]
                feature_h2 = self.cgmn_model(batch_x_p=drop_feature_h2, batch_adj_p=drop_edge_h2)  # [5,10,100]

                cgmn_outputs = {
                    "feature_p1": feature_p1,  # [5,10,100]
                    "feature_p2": feature_p2,  # [5,10,100]
                    "feature_h1": feature_h1,  # [5,10,100]
                    "feature_h2": feature_h2,  # [5,10,100]
                    "feature_p0": feature_p0,  # [5,10,100]
                    "feature_h0": feature_h0   # [5,10,100]
                }
                # Process DXF batch
                entity_type, entity_params = dxf_batch
                entity_type = entity_type.long().to(self.device)
                entity_params = entity_params.float().to(self.device)

                z, proj_z1, proj_z2 = self.dxf_model(entity_type, entity_params)

                dxf_outputs = {
                    "z": z,
                    "proj_z1": proj_z1,
                    "proj_z2": proj_z2
                }

                # Calculate combined loss and optimize
                self.optimizer.zero_grad()
                loss_dict = self.combined_loss(cgmn_outputs, dxf_outputs, self.cgmn_model)
                loss = loss_dict["total_loss"]

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 10 == 0:
                    print(
                        f'Batch {num_batches}: '
                        f'Total Loss = {loss.item():.4f} '
                        f'(CGMN: {loss_dict["cgmn_loss"].item():.4f}, '
                        f'DXF: {loss_dict["dxf_loss"].item():.4f})'
                    )

            except Exception as e:
                print(f"Error in batch {idx}: {str(e)}")
                traceback.print_exc()
                continue

        return total_loss / (num_batches if num_batches > 0 else 1)

    def validate(self):
        """Validate both models"""
        self.cgmn_model.eval()
        self.dxf_model.eval()

        with torch.no_grad():
            tot_diff = []
            tot_truth = []

            if self.epoch_data_valid:
                # print("Starting validation...")
                print(f"Number of validation batches: {len(self.epoch_data_valid)}")

                for batch_idx, valid_batch in enumerate(self.epoch_data_valid):
                    try:
                        # Process CGMN validation data
                        x1, x2, adj1, adj2, y = valid_batch
                        # print(f"\nValidation batch {batch_idx}:")
                        # print(f"x1 shape: {x1.shape}")
                        # print(f"x2 shape: {x2.shape}")
                        # print(f"adj1 shape: {adj1.shape}")
                        # print(f"adj2 shape: {adj2.shape}")
                        # print(f"y values: {y}")

                        feature_p_init = torch.FloatTensor(x1).to(self.device)
                        adj_p = torch.FloatTensor(adj1).to(self.device)
                        feature_h_init = torch.FloatTensor(x2).to(self.device)
                        adj_h = torch.FloatTensor(adj2).to(self.device)

                        # Get CGMN features
                        feature_p = self.cgmn_model(batch_x_p=feature_p_init, batch_adj_p=adj_p)
                        feature_h = self.cgmn_model(batch_x_p=feature_h_init, batch_adj_p=adj_h)

                        # print(f"feature_p shape: {feature_p.shape}")
                        # print(f"feature_h shape: {feature_h.shape}")

                        # Calculate similarity score
                        sim_score = self.cosine_similarity(feature_h, feature_p)
                        print(f"sim_score values: {sim_score}")
                        # print(f"sim_score shape: {sim_score.shape}")

                        tot_diff.extend(sim_score.data.cpu().numpy())
                        tot_truth.extend(y > 0)

                        # print(f"Current tot_diff length: {len(tot_diff)}")
                        # print(f"Current tot_truth length: {len(tot_truth)}")

                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {str(e)}")
                        traceback.print_exc()
                        continue

                # print("\nComputing final metrics:")
                # print(f"Final tot_diff length: {len(tot_diff)}")
                # print(f"Final tot_truth length: {len(tot_truth)}")

                if len(tot_diff) == 0:
                    print("Warning: No valid samples collected during validation")
                    return 0.0

                diff = np.array(tot_diff) * -1
                truth = np.array(tot_truth)

                # print(f"diff values: {diff}")
                # print(f"truth values: {truth}")

                try:
                    fpr, tpr, thresholds = roc_curve(truth, (1 - diff) / 2)
                    # print(f"FPR values: {fpr}")
                    # print(f"TPR values: {tpr}")
                    # print(f"Thresholds: {thresholds}")

                    model_auc = auc(fpr, tpr)
                    print(f"Computed AUC score: {model_auc:.4f}")
                    return model_auc
                except Exception as e:
                    print(f"Error computing ROC/AUC: {str(e)}")
                    traceback.print_exc()
                    return 0.0
            else:
                print("Warning: No validation data available")
                return 0.0

    def calculate_validation_metrics(self, feature_p, feature_h, z, y):
        """
        Calculate validation metrics for both models
        Returns:
            score: combined validation score
        """
        # CGMN similarity
        cgmn_sim = self.cosine_similarity(feature_h, feature_p)

        # Convert y to binary labels if needed
        if isinstance(y, np.ndarray):
            binary_y = (y > 0).astype(int)
        else:
            binary_y = [1 if val > 0 else 0 for val in y]
        binary_y = torch.tensor(binary_y).float().to(feature_p.device)

        # Calculate CGMN score
        cgmn_score = torch.mean((cgmn_sim > 0.5).float() == binary_y).item()

        # Calculate DXF score
        dxf_norm = F.normalize(z, dim=1)
        dxf_sim = torch.mm(dxf_norm, dxf_norm.t())
        dxf_score = torch.mean(torch.diagonal(dxf_sim)).item()

        # Combined score with equal weights
        return 0.5 * cgmn_score + 0.5 * dxf_score

    @staticmethod
    def cosine_similarity(feature_h, feature_p):
        """Calculate cosine similarity between two feature sets"""
        # print("\nIn cosine_similarity:")
        # print(f"feature_h shape: {feature_h.shape}")
        # print(f"feature_p shape: {feature_p.shape}")

        # Calculate mean node features
        agg_h = torch.mean(feature_h, dim=1)
        agg_p = torch.mean(feature_p, dim=1)

        # print(f"agg_h shape: {agg_h.shape}")
        # print(f"agg_p shape: {agg_p.shape}")
        # print(f"agg_h values: {agg_h}")
        # print(f"agg_p values: {agg_p}")

        # Calculate cosine similarity
        sim = torch.nn.functional.cosine_similarity(agg_p, agg_h, dim=1).clamp(min=-1, max=1)
        print(f"similarity values: {sim}")
        # print(f"similarity shape: {sim.shape}")

        return sim

    def fit(self, save_path):
        best_val_score = float('-inf')
        best_epoch = 0

        for epoch in range(1, self.max_epoch + 1):
            print(f'\nEpoch {epoch}/{self.max_epoch}:')

            # Training
            train_loss = self.train_one_epoch()
            print(f'Training Loss: {train_loss:.4f}')

            # Validation
            val_score = self.validate()
            print(f'Validation Score: {val_score:.4f}')

            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                self.save_models(save_path)
                print(f'New best model saved! Score: {best_val_score:.4f}')

            # Early stopping with patience
            if self.earlystopping(val_score, self.cgmn_model):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return best_val_score

    def save_models(self, save_path):
        """Save both models and optimizer state"""
        state = {
            'cgmn_model': self.cgmn_model.state_dict(),
            'dxf_model': self.dxf_model.state_dict(),
            'combined_loss': self.combined_loss.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, save_path)
        print(f"Models saved to {save_path}")

    def load_models(self, load_path):
        """Load both models and optimizer state"""
        if os.path.exists(load_path):
            state = torch.load(load_path)
            self.cgmn_model.load_state_dict(state['cgmn_model'])
            self.dxf_model.load_state_dict(state['dxf_model'])
            self.combined_loss.load_state_dict(state['combined_loss'])
            self.optimizer.load_state_dict(state['optimizer'])
            print(f"Models loaded from {load_path}")
        else:
            print(f"No saved model found at {load_path}")

def main():
    # 开启自动梯度检测
    torch.autograd.set_detect_anomaly(True)
    """Main function to run the training"""
    parser = argparse.ArgumentParser(description='Train Combined CGMN and DeepDXF model')

    # Use cfg_args for CGMN configuration
    print("\nCGMN Configuration:")
    print(cfg_args)

    # Add DXF specific arguments
    parser.add_argument('--data_dir', type=str,
                        default=r'/mnt/share/DeepDXF_CGMN/encode/data/DeepDXF/dxf_vec',
                        help='Directory containing DXF h5 files')
    # 在参数配置中统一批次大小
    parser.add_argument('--batch_size', type=int, default=5,  # 改为与CGMN相同
                        help='Batch size for DXF training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for DXF model')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for contrastive loss')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to load checkpoint from')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        # Initialize trainer
        trainer = CombinedTrainer(cfg_args=cfg_args, dxf_args=args)

        # Load checkpoint if specified
        if args.load_checkpoint:
            trainer.load_models(args.load_checkpoint)

        # Training
        save_path = os.path.join(args.save_dir, 'combined_model.pth')
        best_score = trainer.fit(save_path)

        print(f'\nTraining completed successfully!')
        print(f'Best validation score: {best_score:.4f}')
        print(f'Model saved to: {save_path}')

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()