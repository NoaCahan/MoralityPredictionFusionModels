from setting import parse_opts 
from datasets.fusion_dataset import FusionDataset
from model import generate_model
import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
import random
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from Params import *
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def inference(loader, model, model_type):

    model.eval()
    for data in loader:
        volume, ehrdata,  series = data['ct'] , data['ehr'], data['accession']
        accession = str(series.item())
        with torch.no_grad():
            volume, ehrdata = volume.cuda(), ehrdata.cuda()
            volume, ehrdata = Variable(volume,  requires_grad=False), Variable(ehrdata, requires_grad=False)

            # 3D image input
            if model_type == 'sanet' or model_type == 'swin':
                probs, emb = model(volume)
            # Tabular data input
            elif model_type == 'tabnet':
                probs, M_loss, emb = model(ehrdata)
            # Multimodal input
            else:
                probs, M_loss, emb = model(volume, ehrdata)

            m = torch.nn.Sigmoid()
            probs = m(probs)

            probs = np.array(probs.flatten().data.cpu())
            probs = probs[0].squeeze()
            print("save {}".format(accession))

            if not os.path.exists("./results/predictions"):
                os.makedirs("./results/predictions")
            if not os.path.exists("./results/embeddings"):
                os.makedirs("./results/embeddings")

            np.save("./results/predictions/" + accession, probs)
            np.save("./results/embeddings/" + accession, emb.flatten().data.cpu())

if __name__ == '__main__':

    # settting
    sets = parse_opts()
    sets.phase = 'infer'
    model_type = sets.model_type
    kwargs = {'num_workers': 1}
    root = ROOT
    test_labels = TEST_LABELS
    tabular = TABULAR_CSV

    normMu = [MEAN]
    normSigma = [STD]

    normTransform = transforms.Normalize(normMu, normSigma)
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    # getting model
    checkpoint = torch.load(sets.pretrain_path)
    sets.batch_size = 1
    net, _ = generate_model(sets)
    if sets.model_type != 'swin':
        net.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        net = net.cuda()

    dataset = FusionDataset(root=ROOT, target=test_labels,  ehr_csv = tabular, transform=testTransform, mode="infer")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, **kwargs)

    inference(loader, net, model_type)
