import numpy as np

import torch
import os
from scipy.io import loadmat

def load_Indian_pines():
    data_path = './Indian_pines/Indian_pines_corrected.mat'
    GT_path = './Indian_pines/Indian_pines_gt.mat'


    data_mat = loadmat(data_path)
    #print(data_mat)
    data_np = np.array([[element for element in upperElement] for upperElement in data_mat['indian_pines_corrected']])
    data = torch.from_numpy(data_np.astype('double'))
    gt_mat = loadmat(GT_path)
    raw_gt = torch.tensor([[element for element in upperElement] for upperElement in gt_mat['indian_pines_gt']])

    n_pix,_,n_bands = data.shape

    #training_set_path = './Indian_pines/IP_TraingSet/indianpines_ts_raw_classes.hdr'
    #import spectral as sp
    #hdr = sp.envi.open(training_set_path)
    #training_set_GRSS = torch.tensor(hdr.load())[:,:,0]


    labels_id_raw = [
        'Alfalfa',
        'Corn-notill',
        'Corn-mintill',
        'Corn',
        'Grass-pasture',
        'Grass-trees',
        'Grass-pasture-mowed',
        'Hay-windrowed',
        'Oats',
        'Soybean-notill',
        'Soybean-mintill',
        'Soybean-clean',
        'Wheat',
        'Woods',
        'Buildings-Grass-Trees-Drives',
        'Stone-Steel-Towers'
    ]

    n_classes_raw = len(labels_id_raw)

    #return data, raw_gt, training_set_GRSS, labels_id_raw, n_classes_raw, n_pix, n_bands
    return data, raw_gt, labels_id_raw, n_classes_raw, n_pix, n_bands
