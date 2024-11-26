import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import tifffile
import os

def save_cropped_cells(sdata, crop_size=128, out_dir='image_crop'):
    """
    Function to save cropped cell images from the original intensity image.

    Parameters:
    sdata: SpatialData object containing the image data.
    crop_size: Size of the square crop around each cell's centroid.
    out_dir: Directory where the cropped images will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Get regions from the nucleus image
    regions = regionprops(sdata['HE_nuc_original'][0, :, :].to_numpy())

    # Get the original intensity image
    intensity_image = sdata['HE_original'].to_numpy()

    # Half of the crop size to calculate boundaries
    half_crop = crop_size // 2

    crop_list = []

    # Loop through each region and extract the crop
    for props in tqdm(regions):
        cell_id = props.label
        centroid = props.centroid
        y_center, x_center = int(centroid[0]), int(centroid[1])

        # Calculate the crop boundaries
        minr, maxr = y_center - half_crop, y_center + half_crop
        minc, maxc = x_center - half_crop, x_center + half_crop

        # Ensure boundaries are within the image dimensions
        pad_top = max(0, -minr)
        minr = max(0, minr)

        pad_bottom = max(0, maxr - intensity_image.shape[1])
        maxr = min(maxr, intensity_image.shape[1])

        pad_left = max(0, -minc)
        minc = max(0, minc)

        pad_right = max(0, maxc - intensity_image.shape[2])
        maxc = min(maxc, intensity_image.shape[2])

        # Crop and pad the image if needed
        if pad_top + pad_bottom + pad_left + pad_right > 0:
            crop = np.pad(intensity_image[:, minr:maxr, minc:maxc],
                          ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='constant', constant_values=0)
        else:
            crop = intensity_image[:, minr:maxr, minc:maxc]


        crop_list.append(crop);
        # Save the crop as a TIFF file
        tifffile.imwrite(f"{out_dir}/{cell_id}.tif", crop.astype('uint8'), metadata={'axes': 'CYX'})

    return crop_list

def load_cropped_cells(out_dir='image_crop'):
    """
    Function to load the cropped cell images from the output directory.

    Parameters:
    out_dir: Directory where the cropped images are saved.

    Returns:
    crops: List of cropped cell images.
    """
    crops = []
    # sort the file names 
    files = os.listdir(out_dir)
    files.sort(key=lambda x: int(x.split('.')[0]))
    print(files[:10])

    for file in tqdm(files):
        if file.endswith(".tif"):
            crop = tifffile.imread(f"{out_dir}/{file}")
            crops.append(crop)
    
    return crops

def get_dicts_ind_id(out_dir='image_crop'):
    """
    Function to get the mapping between cell IDs and crop indices.

    Parameters:
    out_dir: Directory where the cropped images are saved.

    Returns:
    id_to_ind: Dictionary mapping cell IDs to crop indices.
    ind_to_id: Dictionary mapping crop indices to cell IDs.
    """
    id_to_ind = {}
    ind_to_id = {}
    files = os.listdir(out_dir)
    files.sort(key=lambda x: int(x.split('.')[0]))

    for i, file in enumerate(files):
        if file.endswith(".tif"):
            cell_id = int(file.split('.')[0])
            id_to_ind[cell_id] = i
            ind_to_id[i] = cell_id

    return id_to_ind, ind_to_id