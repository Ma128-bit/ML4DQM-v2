import pandas as pd
import numpy as np
np.random.seed(42)
import math, copy, os, time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import Utilities.ResNet as ResNet

def batch_generator(df, batch_size):
    n = len(df)
    for start in range(0, n, batch_size):
        yield df.iloc[start:start+batch_size]

def main(df, model, ring, ring_id, num_sectors, max_loss=0.4, min_loss=-0.25):
    df = df[df["max_loss"] < max_loss]
    df = df[df["min_loss"] > min_loss]
    df = df.sample(frac=min(round(5000./len(df), 2), 1.0), random_state=42)
    df_merged_list = []
    batches = list(batch_generator(df, int(len(df)/16) + 1))
    #for batch_df in batch_generator(df, int(len(df)/10) + 1 ):

    for batch_df in tqdm(batches, desc="Generating anomalies:", unit="batch"):
        df_merged = anomaly(batch_df, model, ring, ring_id, num_sectors, max_loss, min_loss)
        df_merged_list.append(df_merged)
    return pd.concat(df_merged_list, ignore_index=True)

def anomaly(df, model, ring, ring_id, num_sectors, max_loss=0.4, min_loss=-0.25):
    binary_matrix = (np.mean(df[ring], axis=0) != 0)
    
    df_anomalies = produce_new_images(df, ring, binary_matrix, ring_id)
    del df

    df_anomalies = df_anomalies.reset_index()

    df_anomalies['down'] = df_anomalies['down'].apply(lambda histo: np.vstack(histo).astype(np.float32))
    df_anomalies['up'] = df_anomalies['up'].apply(lambda histo: np.vstack(histo).astype(np.float32))

    df_an_down = ResNet.predictions(df_anomalies.copy(), model, 'down', num_sectors)
    del df_an_down["reco_img"]
    del df_an_down["loss_img"]
    df_an_up = ResNet.predictions(df_anomalies.copy(), model, 'up', num_sectors)
    del df_an_up["reco_img"]
    del df_an_up["loss_img"]

    df_an_down.loc[100:, ['rebinned_loss_img', 'down', 'up']] = np.nan
    df_an_up.loc[100:, ['rebinned_loss_img', 'down', 'up']] = np.nan

    for s in ["rebinned_loss_img", "max_loss","min_loss"]:
        df_an_down = df_an_down.rename(columns={s: s+"_down"})
        df_an_up = df_an_up.rename(columns={s: s+"_up"})
    del df_an_down["up"]
    del df_an_down["max_loss_down"]
    df_an_down = df_an_down.rename(columns={"min_loss_down": "min_loss"})
    del df_an_down["sf_up"]
    del df_an_up["down"]
    del df_an_up["min_loss_up"]
    df_an_up = df_an_up.rename(columns={"max_loss_up": "max_loss"})
    del df_an_up["sf_down"]
    df_merged = pd.merge(df_an_down, df_an_up, on=['index', 'dim'])
    del df_an_down
    del df_an_up

    #print(df_merged.columns)
    #df_merged.to_pickle(f"{job_label}/MEs_anomalies.pkl")

    return df_merged

def produce_new_images(df, ring, binary_matrix, ring_id):
    if ring_id==1:
        sectors = [1, 2, 3, 4, 5]
    else:
        sectors = [1, 2, 3, 4, 5, 10]
    down_ = []
    up_ = []
    dim = []
    sf_up = []
    sf_down = []
    images = df[ring].to_list()

    for s in sectors:
        for index in np.random.randint(1, s+1, 2):
            s2 = s*18
            scale_factors_up = np.random.uniform(2., 3.5, len(df))
            scale_factors_down = np.random.uniform(0.0, 0.4, len(df))
            
            scaled_up = batch_image_scaled(images, binary_matrix, num_sectors=s2, sector_idx=index, scale_factors=scale_factors_up)
            scaled_down = batch_image_scaled(images, binary_matrix, num_sectors=s2, sector_idx=index, scale_factors=scale_factors_down)

            up_.extend(scaled_up)
            down_.extend(scaled_down)
            sf_up.extend(scale_factors_up)
            sf_down.extend(scale_factors_down)
            dim.extend([s] * len(df))

    return pd.DataFrame({'down': down_, 'up': up_, 'dim': dim, 'sf_up': sf_up, 'sf_down': sf_down})

def batch_image_scaled(images, binary_matrix, num_sectors=18, sector_idx=1, scale_factors=None):
    r, c = images[0].shape
    center = (49.5, 49.5)

    # Creazione della griglia solo una volta
    y, x = np.indices((r, c))
    angles = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)
    
    # Calcolo di theta una volta sola
    theta = np.arctan2(y - center[0], x - center[1]) - np.radians(25) + np.pi / num_sectors
    theta[theta < 0] += 2 * np.pi  # Porta gli angoli nel range corretto
    rebinning = np.digitize(theta, angles) * binary_matrix  # Mappatura settori con maschera
    
    # Seleziona i pixel del settore
    bin_mask = (rebinning == sector_idx)

    # Applica il fattore di scala a tutto il batch in parallelo
    scaled_images = np.array(images).copy()
    
    scaled_images[:, bin_mask] *= scale_factors[:, None]

    return scaled_images.tolist()

def image_scaled(image, binary_matrix, num_sectors=18, sector_idx=1, scale_factor=1):
    angles = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)
    r, c = image.shape
    center = (49.5, 49.5)
    
    # Create polar coordinate grid
    y, x = np.indices((r, c))
    theta = np.arctan2(y - center[0], x - center[1]) -np.radians(25) + np.pi / num_sectors  # Polar angle
    theta[theta < 0] += 2 * np.pi  # Ensure all angles are positive

    # Sector mapping
    rebinning = np.digitize(theta, angles)
    rebinning = rebinning * binary_matrix  # Apply binary mask

    # Identify the bin to scale
    
    bin_mask = (rebinning == sector_idx)

    # Apply the scale factor only to the pixels of the specified sector
    try:
        image[bin_mask] *= scale_factor
    except:
        print(image, binary_matrix, num_sectors, sector_idx, scale_factor)
        exit()

    return image
