import numpy as np

import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)
torch.backends.cudnn.benchmark = False

import torch.utils.data as data
import Utils as utils
from glob import glob
import os
import os.path
import SimpleITK as sitk
import pandas as pd
from Params import *
import matplotlib.pyplot as plt

def load_ehr_dataframe(root, targets):
    label_path = os.path.join(root, targets)
    data_df = pd.read_pickle(label_path)
    return data_df

def load_ehr(dict, accession):
    for key in dict:
        if key == int(accession):
            out =  np.array(dict[key])
    return out

def load_ct(root, accession):
    img_file = root + str(accession) + ".mhd"
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    return img

class FusionDataset(data.Dataset):
    def __init__(self, ehr_csv, root='.', target=None, transform=None, mode="train", seed=1):
        if target is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))

        if target is not None:
            self.data = pd.read_csv(root + target)
        self.transform = transform
        self.mode = mode
        self.root = root
        self.cts = self.root

        ehr_dict =  load_ehr_dataframe(root, ehr_csv)
        self.ehrs = ehr_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        accession = self.data.loc[idx, ACCESSION_COL]
        label = self.data.loc[idx, LABEL_COL]
        label = torch.tensor(label).type(torch.DoubleTensor)
        label = label.reshape(1)

        # Load the CTPA and the tabular data based on their accession numbers
        ct = load_ct(self.cts, accession)
        ct = ct.astype(np.float32)

        ehrdata_row = self.ehrs.loc[self.ehrs[ACCESSION_COL] == accession]
        ehrdata = ehrdata_row.drop([ACCESSION_COL, LABEL_COL], axis=1).values
        ehrdata = ehrdata.squeeze()
        ehrdata = torch.from_numpy(ehrdata.astype(np.float32))

        # Apply transformations if specified
        if self.transform is not None:
            ct = self.transform(ct)
            ct = ct.reshape((1, ct.shape[0], ct.shape[1], ct.shape[2]))
            ct = ct.permute(0,2, 3, 1)

        if self.mode == "train" or self.mode == "test":
            return {'ct': ct, 'ehr': ehrdata, 'target': label} 
        else: #if self.mode == "infer"
            return {'ct': ct, 'ehr': ehrdata, 'accession': accession}
