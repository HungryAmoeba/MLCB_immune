from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spatialdata_plot
import spatialdata as sd
import scanpy as sc
import os


def load_sdata(path):
    """
    Function to load the spatial data from the specified path.
    Returns the spatial data object.

    Parameters:
    path: the path to the spatial data file.

    Returns:
    sdata: Spatial data object.
    """
    sdata = sd.read_zarr(path)
    return sdata


