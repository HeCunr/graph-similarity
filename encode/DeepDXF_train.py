
import torch
from torch.utils.data import DataLoader, ConcatDataset
from model.DeepDXF_dataset import DXFDataset
from model.transformer_encoder import DXFTransformer
from model.DeepDXF_loss import ContrastiveLoss
import os
import h5py
import argparse
import traceback

def load_h5_files(directory):
    datasets = []
    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            try:
                dataset = DXFDataset(file_path)
                datasets.append(dataset)
            except KeyError as e:
                print(f"Error loading {filename}: {e}")
                continue
    if not datasets:
        raise ValueError("No valid datasets found in the specified directory.")
    return ConcatDataset(datasets)

def main(args):
    h5_directory = args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        combined_dataset = load_h5_files(h5_directory)
        print(f"Dataset loaded, size: {len(combined_dataset)}")
        batch_size = args.batch_size
        dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print(f"Dataloader created, number of batches: {len(dataloader)}")

        model = DXFTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        contrastive_loss = ContrastiveLoss(batch_size=batch_size, temperature=args.temperature, loss_type=args.loss_type).to(device)

        num_epochs = args.epochs
        for epoch in range(num_epochs):
            total_loss = 0
            for i, (entity_type, entity_params) in enumerate(dataloader):
                entity_type = entity_type.long().to(device)
                entity_params = entity_params.float().to(device)

                print(f"Batch {i}: entity_type shape: {entity_type.shape}, dtype: {entity_type.dtype}")
                print(f"Batch {i}: entity_params shape: {entity_params.shape}, dtype: {entity_params.dtype}")

                optimizer.zero_grad()
                z, proj_z1, proj_z2 = model(entity_type, entity_params)

                print(f"z shape: {z.shape}")
                print(f"proj_z1 shape: {proj_z1.shape}")
                print(f"proj_z2 shape: {proj_z2.shape}")

                loss = contrastive_loss(proj_z1, proj_z2)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(dataloader)}, Loss: {loss.item()}")

            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(dataloader)}")

        # Save the model
        torch.save(model.state_dict(), 'dxf_transformer_model.pth')
        print("Model saved successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DXF Transformer with contrastive loss')
   # parser.add_argument('--data_dir', type=str, default=r'C:\srtp\encode\data\DeepDXF\dxf_vec', help='Directory containing h5 files')
    parser.add_argument('--data_dir', type=str, default=r'/mnt/share/DeepDXF_CGMN/encode/data/DeepDXF/dxf_vec', help='Directory containing h5 files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for contrastive loss')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--loss_type', type=str, default='simclr', choices=['simclr', 'infonce'], help='Type of contrastive loss to use')

    args = parser.parse_args()
    main(args)



