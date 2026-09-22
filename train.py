import os
import csv
import argparse
from math import cos, pi
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger

import torch
from torch.utils.data import DataLoader

from pytorch_msssim import SSIM

from congestion.model import CongestionModel
from congestion.dataloader import LibrelaneDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()


class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                       end: float,
                       factor: float,
                       weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                          f'cumulative_periods {cumulative_periods}')

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]

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
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # lr scheduler: cosine annealing over the full training run
    max_iters = num_epochs * len(train_loader)
    cosine_lr = CosineRestartLr(lr, [max_iters], [1], min_lr=1e-7)
    cosine_lr.set_init_lr(optimizer)
    iter_num = 0

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    logger.info('Start training')
    train_losses = []
    valid_losses = []
    train_ssim_losses = []
    valid_bce_losses = []
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

            regular_lr = cosine_lr.get_regular_lr(iter_num)
            cosine_lr._set_lr(optimizer, regular_lr)
            iter_num += 1

            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(features)
                    train_loss = criterion(pred, labels) * 1000
                optimizer.zero_grad()
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(features)
                train_loss = criterion(pred, labels) * 1000
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
        v_bce = 0
        n2 = 0
        for batch_idx, (features, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test'):
            features = features.to(device=device)
            labels = labels.to(device=device)

            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = model(features)
                        val_bce_loss = criterion(pred, labels) * 1000
                        pred = model.sigmoid(pred)
                        test_loss = 1.0 - ssim(pred.float(), labels.float())
                else:
                    pred = model(features)
                    val_bce_loss = criterion(pred, labels) * 1000
                    pred = model.sigmoid(pred)
                    test_loss = 1.0 - ssim(pred.float(), labels.float())

            v += test_loss.item()
            v_bce += val_bce_loss.item()
            n2 += 1

        valid_losses.append(v/n2)
        valid_bce_losses.append(v_bce/n2)

        logger.info("\n")
        logger.info(f'Epoch {e}: Train Loss: {t/n1/1000}  | Test Loss: {v_bce/n2}  | LR: {optimizer.param_groups[0]["lr"]:.6g}')

        if t/n1 < best_train_Loss:
            logger.info(f'Best Epoch {e}: Train Loss: {t/n1}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_best_train_weights.pth')
            best_train_Loss = t/n1

        if v_bce/n2 < best_test_Loss:
            logger.info(f'Best Epoch {e}: Test Loss: {v_bce/n2}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_best_test_weights.pth')
            best_test_Loss = v_bce/n2

        # rewritten every epoch so progress survives interruption
        with open(f"{fig_savepath}/losses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_bce_loss", "val_bce_loss", "train_ssim_loss", "val_ssim_loss"])
            for epoch_idx, (tr, va, tr_s, va_s) in enumerate(zip(train_losses, valid_bce_losses, train_ssim_losses, valid_losses)):
                writer.writerow([epoch_idx, tr, va, tr_s, va_s])

        # BCE-based loss: same metric for train and val
        fig = plt.figure()
        epochnum = list(range(0,len(train_losses)))
        plt.plot(epochnum, train_losses, color='black', linewidth=1, label='Train')
        plt.plot(epochnum, valid_bce_losses, color='red', linewidth=1, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, len(train_losses))
        plt.legend(loc='best', fontsize=16)
        plt.title("BCE Loss")
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
    parser.add_argument("--num_epochs", default=1000, type=int, help='The training epochs')
    parser.add_argument("--weight_path", default="./models/model_weight", type=str, help='The path to save the model weight')
    parser.add_argument("--fig_path", default="./figures", type=str, help='The path of the figure file')
    parser.add_argument("--learning_rate", default=0.0001, type=float, help='learning rate [0,1]')
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
