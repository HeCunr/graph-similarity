import os
import time
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import torch
from datetime import datetime
from sklearn.model_selection import KFold
from pathlib import Path
from model.GF_dataset import GraphData
from tqdm import tqdm
class Graph:
    """Graph data structure with basic operations"""
    def __init__(self, node_num: int, name: str = None):
        self.node_num = node_num
        self.name = name
        self.features = np.zeros((node_num, 0))
        self.neighbors: Dict[int, List[int]] = {i: [] for i in range(node_num)}
        self.edge_list: List[Tuple[int, int]] = []

    def add_edge(self, u: int, v: int):
        """Add edge between nodes u and v"""
        self.neighbors[u].append(v)
        self.neighbors[v].append(u)
        self.edge_list.append((u, v))

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        adj = np.zeros((self.node_num, self.node_num))
        for u, v in self.edge_list:
            adj[u][v] = adj[v][u] = 1
        return adj

def setup_logger(args, model_name: str) -> Tuple[str, logging.Logger]:
    """
    Setup logging configuration
    
    Args:
        args: Configuration arguments
        model_name: Name of the model for logging
        
    Returns:
        Tuple of log directory path and logger object
    """
    # Create timestamp and log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_path) / f"{args.dataset}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logger
    logger = logging.getLogger('GF')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / 'train.log')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Log configuration
    logger.info(f"Starting {model_name} training with config:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    return str(log_dir), logger

def generate_batches(
        graphs: List[Graph],
        batch_size: int,
        shuffle: bool = True
) -> List[List[Graph]]:
    """
    Generate batches of graphs for training
    
    Args:
        graphs: List of all graphs
        batch_size: Size of each batch
        shuffle: Whether to shuffle graphs before batching
        
    Returns:
        List of batches, where each batch is a list of graphs
    """
    num_graphs = len(graphs)
    indices = list(range(num_graphs))

    if shuffle:
        random.shuffle(indices)

    batches = []
    for i in range(0, num_graphs - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [graphs[idx] for idx in batch_indices]
        batches.append(batch)

    return batches

def generate_pairs_from_batch(
        batch: List[Graph]
) -> List[Tuple[Graph, Graph]]:
    """
    Generate random pairs from a batch of graphs
    
    Args:
        batch: List of graphs in the batch
        
    Returns:
        List of graph pairs
    """
    indices = list(range(len(batch)))
    random.shuffle(indices)
    pairs = []

    # Generate pairs from shuffled indices
    for i in range(0, len(indices), 2):
        if i + 1 < len(indices):
            pairs.append((batch[indices[i]], batch[indices[i + 1]]))

    return pairs



def prepare_batch_data(pairs: List[Tuple[GraphData, GraphData]], device: torch.device) -> Tuple[torch.Tensor, ...]:
    try:
        features1, adj1, features2, adj2 = [], [], [], []

        for i, (g1, g2) in enumerate(pairs):
            if g1.matrices is None or g2.matrices is None:
                raise ValueError(f"Graph at index {i} has not been preprocessed")

            feat1, adj_mat1, _ = g1.matrices
            feat2, adj_mat2, _ = g2.matrices

            # Convert to double precision
            features1.append(torch.tensor(feat1, dtype=torch.float64))
            adj1.append(torch.tensor(adj_mat1, dtype=torch.float64))
            features2.append(torch.tensor(feat2, dtype=torch.float64))
            adj2.append(torch.tensor(adj_mat2, dtype=torch.float64))

        # Stack tensors
        features1 = torch.stack(features1).to(device)
        adj1 = torch.stack(adj1).to(device)
        features2 = torch.stack(features2).to(device)
        adj2 = torch.stack(adj2).to(device)

        return features1, adj1, features2, adj2

    except Exception as e:
        print(f"Error in prepare_batch_data: {str(e)}")
        raise

def cross_validation_split(
        graphs: List[Graph],
        n_folds: int,
        test_size: float,
        random_state: int = 42
) -> Tuple[List[Tuple[List[int], List[int]]], List[int]]:
    """
    Split graphs for cross validation and final testing
    
    Args:
        graphs: List of all graphs
        n_folds: Number of folds for cross validation
        test_size: Proportion of data for final testing
        random_state: Random seed
        
    Returns:
        Tuple of (cv_splits, test_indices)
    """
    num_graphs = len(graphs)
    indices = list(range(num_graphs))

    # Split into train+val and test
    test_size_abs = int(num_graphs * test_size)
    train_val_indices = indices[:-test_size_abs]
    test_indices = indices[-test_size_abs:]

    # Create cross-validation splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_splits = list(kf.split(train_val_indices))

    return cv_splits, test_indices

def save_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        path: str
):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: str
) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def get_device(args) -> torch.device:
    """Get appropriate device based on configuration"""
    if torch.cuda.is_available() and int(args.gpu_index) >= 0:  # 转换为整数
        device = torch.device(f'cuda:{args.gpu_index}')
        torch.cuda.set_device(int(args.gpu_index))  # 显式设置 GPU 设备
    else:
        device = torch.device('cpu')
    return device

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False