import argparse 
import os
import numpy as np
import pandas as pd
import torch 

from src.data.sd_dataloader import SpatialDataset
from src.utils.crop_list import save_cropped_cells, load_cropped_cells, get_dicts_ind_id
from src.utils.process_sdata import load_sdata
from src.models.hibou_st import HibouST
from torch.utils.data import DataLoader, random_split

from transformers import AutoImageProcessor, AutoModel

def main():
    parser = argparse.ArgumentParser(description='Train a model on spatial data')
    parser.add_argument('--data_path', type=str, default = 'notebooks/data/UC1_NI.zarr', help='Path to the data file')
    parser.add_argument('--output_dir', type=str, default= 'logs/', help='Directory to save the output')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--recompute_crops', type = int, default = 0, help='Recompute the crops')
    parser.add_argument('--crop_dir', type = str, default = 'image_crop/', help='Directory to save the crops')
    parser.add_argument('--linear_config', type = list, default = [1024, 512], help='Configuration for the linear layers')
    args = parser.parse_args()

    # Load data
    # list every thing in data path
    sdata = load_sdata(args.data_path)

    ## Get the list of gene concerned in crunch 1
    gene_name_list = sdata['anucleus'].var['gene_symbols'].values
    x_count = pd.DataFrame(sdata['anucleus'].layers['counts'], columns=gene_name_list) # raw count data
    x_norm = pd.DataFrame((sdata['anucleus'].X), columns=gene_name_list) # normalized data

    ## Selecting training cells
    cell_id_example = sdata['cell_id-group'].obs[sdata['cell_id-group'].obs['group'] == 'train']['cell_id'].to_numpy()
    cell_id_example = list(set(cell_id_example).intersection(set(sdata['anucleus'].obs['cell_id'].unique())))

    ## Get y from the anucleus data
    ground_truth_example = sdata['anucleus'].layers['counts'][sdata['anucleus'].obs['cell_id'].isin(cell_id_example),:]
    y = pd.DataFrame(ground_truth_example, columns= gene_name_list, index = cell_id_example)

    # for some reason the cell_id-group not equal to train have empty intersection with anucleus
    # so we split the training cells into train and validation and test sets

    if args.recompute_crops:
        # Save the cropped images
        crop_list = save_cropped_cells(sdata, y, out_dir=args.crop_dir)
    else:
        # Load the cropped images
        crop_list = load_cropped_cells(out_dir=args.crop_dir)
    
    # get the dictionary to convert cell_id to crop_list index
    cell_id_to_crop_list_ind_dict, crop_list_id_to_ind = get_dicts_ind_id(args.crop_dir)

    # Create the dataset
    dataset = SpatialDataset(crop_list, y, cell_id_to_crop_list_ind_dict=crop_list_id_to_ind)

    # split the dataset into train, validation and test sets and then make dataloaders for each
    # Define the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
    model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)

    # Initialize model
    hibou_st = HibouST(model, linear_config=[1024, 512, 256])

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(hibou_st.parameters(), lr=args.learning_rate)

    # Move the model to the device
    hibou_st.to(device)

    # Initialize lists to store loss history
    train_loss_history = []

    # Training loop
    for epoch in range(args.epochs):
        hibou_st.train()
        train_loss = 0
        for i, (image, expression) in enumerate(train_loader):
            image = {k: v.to(device) for k, v in image.items()}
            expression = expression.to(device)

            optimizer.zero_grad()
            output = hibou_st(image)

            loss = criterion(output, expression)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_train_loss}')

    # Save the loss history
    loss_history_path = os.path.join(args.output_dir, 'loss_history.npy')
    np.save(loss_history_path, np.array(train_loss_history))

    # Evaluate the model on the validation set
    hibou_st.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (image, expression) in enumerate(val_loader):
            image = {k: v.to(device) for k, v in image.items()}
            expression = expression.to(device)

            output = hibou_st(image)
            loss = criterion(output, expression)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss}')

    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))

    # Write the args to a file
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

if __name__ == '__main__':
    main()
