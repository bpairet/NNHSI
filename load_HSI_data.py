import numpy as np

import torch
import os
from scipy.io import loadmat

def load_HSI_data(data_name):
    if data_name == 'indian_pines_corrected':
        path_to_data = './indian_pines/'
        gt_name = 'indian_pines_gt'
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

    elif data_name == 'paviaU':
        path_to_data = './paviaU/'
        gt_name = 'paviaU_gt'
        labels_id_raw = ['Asphalt','Meadows','Gravel','Trees',
                         'Painted metal sheets','Bare Soil','Bitumen',
                         'Self-Blocking' 'Bricks','Shadows']
    elif data_name == 'pavia':
        path_to_data = './pavia/'
        gt_name = 'pavia_gt'
        labels_id_raw = ['Water','Trees','Asphalt','Self-Blocking Bricks',
                         'Bitumen','Tiles','Shadows','Meadows','Bare Soil']
    else:
        print('data_name not recognized')

    data_path = path_to_data+data_name
    GT_path = path_to_data+gt_name
    data_mat = loadmat(data_path)
    #print(data_mat)
    data_np = np.array([[element for element in upperElement] for upperElement in data_mat[data_name]])
    data = torch.from_numpy(data_np.astype('double'))
    gt_mat = loadmat(GT_path)
    raw_gt = torch.tensor([[element for element in upperElement] for upperElement in gt_mat[gt_name]])

    n_pix,_,n_bands = data.shape

    #training_set_path = './Indian_pines/IP_TraingSet/indianpines_ts_raw_classes.hdr'
    #import spectral as sp
    #hdr = sp.envi.open(training_set_path)
    #training_set_GRSS = torch.tensor(hdr.load())[:,:,0]
    
    n_classes_raw = len(labels_id_raw)

    #return data, raw_gt, training_set_GRSS, labels_id_raw, n_classes_raw, n_pix, n_bands
    return data, raw_gt, labels_id_raw, n_classes_raw, n_pix, n_bands
