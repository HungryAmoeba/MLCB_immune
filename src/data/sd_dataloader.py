# make a dataloader that pairs the image crop with its spatial transcriptomics expression stored in y

import torch
from torch.utils.data import Dataset, DataLoader

class SpatialDataset(Dataset):
    def __init__(self, image_crops, expressions, transform=None, cell_id_to_crop_list_ind_dict=None):
        self.image_crops = image_crops
        self.expressions = expressions
        self.transform = transform 
        self.id_to_ind = cell_id_to_crop_list_ind_dict

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        # idx specifies the ROW number of the expression data
        cell_id = self.expressions.index[idx]
        image = self.image_crops[self.id_to_ind[cell_id]]
        expression = self.expressions.iloc[idx].values
        if self.transform:
            image = self.transform(image, return_tensors="pt")['pixel_values']
        image = image.squeeze(0)
        image = {'pixel_values': image}
        return image, expression

# example usage: 

# subset the crop_list and y to only include the training cells
#crop_list_train = [crop_list[crop_list_id_to_ind[cell_id]] for cell_id in cell_id_example]

# # Create the dataset
# MAKE SURE PROCESSER IS PASSED IN AS A PARAMETER
# dataset = SpatialDataset(crop_list, y, cell_id_to_crop_list_ind_dict=crop_list_id_to_ind)

# # Create the dataloader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)