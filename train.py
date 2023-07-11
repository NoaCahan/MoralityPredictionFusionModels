from setting import parse_opts 
from datasets.fusion_dataset import FusionDataset
from model import generate_model
import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)
import random
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import ndimage
import os
from Params import *
from measurements import *
import loss as clsloss
from sklearn.metrics import auc, roc_curve
import copy
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


def train(no_cuda, epoch, model, trainLoader, optimizer, pos_weight, model_type):

    model.train()
    nProcessed = 0
    ncorrect_class = 0
    train_loss = 0
    gts = []
    probs = []
    nTrain = len(trainLoader.dataset)

    for batch_idx, sample in enumerate(trainLoader):

        volume, ehrdata, target = sample['ct'] , sample['ehr'], sample['target']
        if not no_cuda:
            volume, ehrdata, target = volume.cuda(), ehrdata.cuda(), target.cuda()
        volume, ehrdata, target = Variable(volume), Variable(ehrdata), Variable(target)

        optimizer.zero_grad()
        M_loss = None

        # 3D image input
        if model_type == 'sanet' or model_type == 'swin':
            output, _ = model(volume)
        # Tabular data input
        elif model_type == 'tabnet' or model_type == 'tabtransformer':
            output, M_loss, embs = model(ehrdata)
        # Multimodal input
        else:
            output, M_loss, embs = model(volume, ehrdata)

        loss_func = clsloss.classification_loss
        loss =  loss_func(output, target, pos_weight=pos_weight)

        if M_loss != None:
            # Add the overall sparsity loss
            loss = loss -sets.lambda_sparse * M_loss

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        nProcessed += len(volume)

        # Get Classification prediction
        probs.append(output.data.cpu())
        m = nn.Sigmoid()
        preds =  m(output).data.cpu() > 0.5
        preds = preds.long()
        output = target.long().data.cpu()
        ncorrect_class += np.sum(np.array(preds == output),axis=0)
        gts.append(np.array(output))

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1

        if model_type == 'sanet':
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t '.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),loss.item()))

    gts = np.vstack(gts)
    probs = np.vstack(probs)

    AUROCs = compute_AUCs(gts, probs)
    AUROC_avg = np.array(AUROCs).mean()
    nClassErr = 100 - 100.*ncorrect_class/(nTrain)
    nClassErrAvg = np.mean(nClassErr)       
    train_loss /= len(trainLoader)
    print('Train Epoch: {}\t Mean Classification Error: {:.2f} % \t Mean AUC:  ({:.3f}) \
    '.format( (epoch-1), nClassErrAvg, AUROC_avg))

    # Save: Epoch, Loss, Accuracy, AUC
    return train_loss, 100 - nClassErrAvg, AUROC_avg


def test(no_cuda, epoch, model, testLoader, optimizer, pos_weight, model_type):

    model.eval()
    test_loss = 0
    ncorrect_class = 0
    nTest = len(testLoader.dataset)
    probs = []
    gts = []
    for sample in testLoader:

        M_loss = None

        volume, ehrdata, target = sample['ct'] , sample['ehr'], sample['target']

        if not no_cuda:
            volume, ehrdata, target = volume.cuda(), ehrdata.cuda(), target.cuda()
        with torch.no_grad():
            volume, ehrdata, target = Variable(volume,  requires_grad=False), Variable(ehrdata, requires_grad=False), Variable(target, requires_grad=False)

            # 3D image input
            if model_type == 'sanet' or model_type == 'swin':
                output, _ = model(volume)
            # Tabular data input
            elif model_type == 'tabnet' or model_type == 'tabtransformer':
                output, M_loss, embs = model(ehrdata)
            # Multimodal input
            else:
                output, M_loss, embs = model(volume, ehrdata)

            loss_func = clsloss.classification_loss

            if M_loss != None:
                # Add the overall sparsity loss
                test_loss += loss_func(output, target, pos_weight=pos_weight) - sets.lambda_sparse * M_loss
            else:
                test_loss +=  loss_func(output, target, pos_weight=pos_weight)

            prob = copy.deepcopy(output)
            probs.append(prob.data.cpu())
            
            m = nn.Sigmoid()
            preds =  m(output).data.cpu() > 0.5
            preds = preds.long().data.cpu()
            gt = target.long().data.cpu()

            ncorrect_class += np.sum(np.array(preds == gt), axis=0)
            gts.append(np.array(gt))

    gts = np.vstack(gts)
    probs = np.vstack(probs)

    AUROCs = compute_AUCs(gts, probs)
    AUROC_avg = np.array(AUROCs).mean()
    nClassErr = 100 - (100.*ncorrect_class/ (nTest))
    nClassErrAvg = np.mean(nClassErr)
    test_loss /= len(testLoader)
    print('Test set: Average loss: {:.4f}\t Mean Classification Error: ({:.3f}%) Mean AUC:  ({:.3f}) \
    '.format(test_loss, nClassErrAvg, AUROC_avg))

    return test_loss, 100 - nClassErrAvg, AUROC_avg


if __name__ == '__main__':

    sets = parse_opts()     
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{128 + 1}"
    train_labels = TRAIN_LABELS
    valid_labels = VALID_LABELS
    tabular = TABULAR_CSV

    writer = SummaryWriter(comment=' - base_LR - {}, {} epochs'.format(sets.learning_rate, sets.n_epochs))

    writer.add_scalar('Params/epochs', sets.n_epochs, 1)
    writer.add_scalar('Params/batch_size', sets.batch_size, 1)

    pos_weight = torch.cuda.FloatTensor([POS_WEIGHT]).cuda()
    best_val_auc = 0.

    normMu = [MEAN]
    normSigma = [STD]

    normTransform = transforms.Normalize(normMu, normSigma)

    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 

    model_type = sets.model_type
    if sets.pretrain_path:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
                ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]

    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size = 30)

    if sets.no_cuda:
        sets.pin_memory = False
        kwargs = {}
    else:
        sets.pin_memory = True
        kwargs = {'num_workers': 1, 'pin_memory': True}     

    training_dataset = FusionDataset(root=sets.data_root, target=train_labels, ehr_csv = tabular,  mode="train", transform=trainTransform, seed=sets.manual_seed)
    train_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, drop_last=True, **kwargs)

    validating_dataset = FusionDataset(root=sets.data_root, target=valid_labels, ehr_csv = tabular, mode="test", transform=testTransform, seed=sets.manual_seed)
    valid_loader = DataLoader(validating_dataset, batch_size=sets.batch_size, shuffle=False, **kwargs)

    # training
    if not os.path.exists(sets.save_folder):
        os.makedirs(sets.save_folder)

    for epoch in range(1, sets.n_epochs + 1):

        train_loss, train_acc, train_auc = train(sets.no_cuda, epoch, model, train_loader, optimizer, pos_weight, model_type)
        valid_loss, valid_acc, valid_auc = test(sets.no_cuda, epoch, model, valid_loader, optimizer, pos_weight, model_type)

        writer.add_scalars(f'Loss/epoch', {'train': train_loss, 'validation': valid_loss}, epoch)
        writer.add_scalars(f'AUC/epoch', {'train': train_auc, 'validation': valid_auc}, epoch)

        if valid_auc >= best_val_auc:

            best_val_auc = valid_auc
            model_save_path = '{}/best_val.pth.tar'.format(sets.save_folder)
            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': sets},
                        model_save_path)

        if epoch % sets.save_intervals == 0:

            model_save_path = '{}/epoch_{}.pth.tar'.format(sets.save_folder, epoch)
            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            log.info('Save checkpoints: epoch = {}'.format(epoch)) 
            torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': sets},
                        model_save_path)

    writer.run.summary["best_valid_auc"] = best_val_auc
