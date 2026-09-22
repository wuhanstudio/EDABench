import os
import csv
import functools
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_msssim import SSIM

from congestion.model import CongestionModel
from congestion.dataloader import LibrelaneDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12
        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)
    return loss


def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', sample_wise=False, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss
    return wrapper


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)

def train(rootpath,batch_size,num_epochs,lr,fig_savepath,weight_savepath):

    #data
    train_dataset = LibrelaneDataset(root_dir=rootpath + "/training",transform=True)
    test_dataset = LibrelaneDataset(root_dir=rootpath + "/testing",transform=True)

    len_train_set = len(train_dataset)
    len_test_set  = len(test_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    #model
    model = CongestionModel(device).to(device)

    #criterion
    ssim = SSIM(data_range=1, size_average=True, channel=1)
    criterion = MSELoss()
    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    logger.info('Start training')
    train_losses = []
    valid_losses = []
    train_ssim_losses = []
    valid_mse_losses = []
    best_test_Loss = 99999999999999
    best_train_Loss = 99999999999999

    for e in range(num_epochs):
        logger.info(f'Epoch {e}/{num_epochs - 1}')

        # Training
        t = 0
        t_ssim = 0
        n1 = 0

        for batch_idx, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train'):
            features = features.to(device=device)
            labels = labels.to(device=device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(features)
                    train_loss = criterion(model.sigmoid(pred), labels)
                optimizer.zero_grad()
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(features)
                train_loss = criterion(model.sigmoid(pred), labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            # also track the SSIM-based metric so it's comparable to validation
            with torch.no_grad():
                train_ssim_loss = 1.0 - ssim(model.sigmoid(pred.detach()).float(), labels.float())

            t += train_loss.item()
            t_ssim += train_ssim_loss.item()
            n1 += 1

        train_losses.append(t/n1)
        train_ssim_losses.append(t_ssim/n1)

        # Evaluation
        model.eval()
        v = 0
        v_mse = 0
        n2 = 0
        for batch_idx, (features, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test'):
            features = features.to(device=device)
            labels = labels.to(device=device)

            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = model(features)
                        pred = model.sigmoid(pred)
                        val_mse_loss = criterion(pred, labels)
                        test_loss = 1.0 - ssim(pred.float(), labels.float())
                else:
                    pred = model(features)
                    pred = model.sigmoid(pred)
                    val_mse_loss = criterion(pred, labels)
                    test_loss = 1.0 - ssim(pred.float(), labels.float())

            v += test_loss.item()
            v_mse += val_mse_loss.item()
            n2 += 1

        valid_losses.append(v/n2)
        valid_mse_losses.append(v_mse/n2)

        logger.info("\n")
        logger.info(f'Epoch {e}: Train Loss: {t/n1}  | Test Loss: {v_mse/n2}')

        if t/n1 < best_train_Loss:
            logger.info(f'Best Epoch {e}: Train Loss: {t/n1}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_best_train_weights.pth')
            best_train_Loss = t/n1

        if v_mse/n2 < best_test_Loss:
            logger.info(f'Best Epoch {e}: Test Loss: {v_mse/n2}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_best_test_weights.pth')
            best_test_Loss = v_mse/n2

        # rewritten every epoch so progress survives interruption
        with open(f"{fig_savepath}/losses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_mse_loss", "val_mse_loss", "train_ssim_loss", "val_ssim_loss"])
            for epoch_idx, (tr, va, tr_s, va_s) in enumerate(zip(train_losses, valid_mse_losses, train_ssim_losses, valid_losses)):
                writer.writerow([epoch_idx, tr, va, tr_s, va_s])

        # MSE-based loss: same metric for train and val
        fig = plt.figure()
        epochnum = list(range(0,len(train_losses)))
        plt.plot(epochnum, train_losses, color='black', linewidth=1, label='Train')
        plt.plot(epochnum, valid_mse_losses, color='red', linewidth=1, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, len(train_losses))
        plt.legend(loc='best', fontsize=16)
        plt.title("MSE Loss")
        plt.grid(linestyle=':')
        plt.savefig(f"{fig_savepath}/train_losses.png")
        plt.close(fig)

        # SSIM-based loss: same metric for train and val
        fig = plt.figure()
        epochnum = list(range(0,len(train_losses)))
        plt.plot(epochnum, train_ssim_losses, color='black', linewidth=1, label='Train')
        plt.plot(epochnum, valid_losses, color='red', linewidth=1, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, len(train_losses))
        plt.legend(loc='best', fontsize=16)
        plt.title("SSIM Loss")
        plt.grid(linestyle=':')
        plt.savefig(f"{fig_savepath}/val_losses.png")
        plt.close(fig)


        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        pred = model.sigmoid(pred)
        ax[0].imshow(pred[0,0].detach().cpu())
        ax[1].imshow(labels[0,0].cpu())
        ax[0].title.set_text('Pred')
        ax[1].title.set_text('Label')
        plt.savefig(f"{fig_savepath}/compare.png")
        plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Librelane Congestion Model Training")
    parser.add_argument("--root_path", default="./datasets/", type=str, help='The path of the data file')
    parser.add_argument("--batch_size", default=8, type=int, help='The batch size')
    parser.add_argument("--num_epochs", default=100, type=int, help='The training epochs')
    parser.add_argument("--weight_path", default="./models/model_weight", type=str, help='The path to save the model weight')
    parser.add_argument("--fig_path", default="./figures", type=str, help='The path of the figure file')
    parser.add_argument("--learning_rate", default=0.001, type=float, help='learning rate [0,1]')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import time
    start = time.time()
    args = parse_args()

    if not os.path.exists(args.weight_path):
        os.makedirs(args.weight_path)
    if not os.path.exists(args.fig_path):
        os.makedirs(args.fig_path)

    train(rootpath=args.root_path,batch_size=args.batch_size,num_epochs=args.num_epochs,lr=args.learning_rate,
          fig_savepath=args.fig_path,weight_savepath=args.weight_path)
    end = time.time()
    logger.info("training cost time：%f sec" % (end - start))
