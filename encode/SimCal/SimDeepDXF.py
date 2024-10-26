# !/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import h5py
import argparse
from model.transformer_encoder import DXFTransformer
from model.DeepDXF_dataset import DXFDataset
from torch.utils.data import DataLoader

class SimDeepDXF:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Initialize and load the model
        self.model = DXFTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def load_single_h5(self, h5_path):
        """Load and process a single h5 file"""
        dataset = DXFDataset(h5_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_embeddings = []

        with torch.no_grad():
            for entity_type, entity_params in dataloader:
                entity_type = entity_type.to(self.device)
                entity_params = entity_params.to(self.device)

                # Get embeddings using the model
                z, _, _ = self.model(entity_type, entity_params)
                all_embeddings.append(z)

        # Combine all embeddings if there are multiple entries in the h5 file
        final_embedding = torch.mean(torch.stack(all_embeddings), dim=0)
        return final_embedding

    def calculate_similarity(self, embedding1, embedding2, method='cosine'):
        """Calculate similarity between two embeddings"""
        if method == 'cosine':
            # Normalize embeddings
            embedding1_norm = F.normalize(embedding1, p=2, dim=1)
            embedding2_norm = F.normalize(embedding2, p=2, dim=1)

            # Calculate cosine similarity
            similarity = F.cosine_similarity(embedding1_norm, embedding2_norm)
            return similarity.item()

        elif method == 'euclidean':
            # Calculate Euclidean distance and convert to similarity
            distance = torch.cdist(embedding1, embedding2, p=2)
            # Convert distance to similarity (1 / (1 + distance))
            similarity = 1 / (1 + distance)
            return similarity.item()

        else:
            raise ValueError(f"Unsupported similarity method: {method}")

    def compare_h5_files(self, h5_path1, h5_path2, method='cosine'):
        """Compare two h5 files and return their similarity score"""
        try:
            # Get embeddings for both files
            embedding1 = self.load_single_h5(h5_path1)
            embedding2 = self.load_single_h5(h5_path2)

            # Calculate and return similarity
            similarity = self.calculate_similarity(embedding1, embedding2, method)
            return similarity

        except Exception as e:
            print(f"Error comparing files: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Calculate similarity between two DXF files')
    parser.add_argument('--model_path', type=str,default=r'C:\srtp\encode\pretrained\DeepDXF\dxf_transformer_model.pth', help='Path to the trained model weights')
    parser.add_argument('--file1', type=str, default=r'C:\srtp\encode\data\DeepDXF\dxf_vec\DFN6BU(NiPdAu)-437 Rev1_1.h5',help='Path to first h5 file')
    parser.add_argument('--file2', type=str,default=r'C:\srtp\encode\data\DeepDXF\dxf_vec\DFN6BU(NiPdAu)-437 Rev1_2.h5',help='Path to second h5 file')
    parser.add_argument('--method', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='Similarity calculation method')

    args = parser.parse_args()

    # Initialize SimDeepDXF
    sim_dxf = SimDeepDXF(args.model_path)

    # Calculate similarity
    similarity = sim_dxf.compare_h5_files(args.file1, args.file2, args.method)

    if similarity is not None:
        print(f"Similarity score ({args.method}): {similarity:.4f}")
        print(f"Files are {similarity*100:.2f}% similar")

if __name__ == "__main__":
    main()