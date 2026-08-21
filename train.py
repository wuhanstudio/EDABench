import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from pytorch_msssim import SSIM

from congestion.model import CongestionModel
from congestion.dataloader import LibrelaneDataset

device = 'cpu'  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = False # torch.cuda.is_available()

def train(rootpath,batch_size,num_epochs,lr,fig_savepath,weight_savepath):

    #data
    dataset = LibrelaneDataset(root_dir=rootpath,transform=True)
    len_train_set = int(len(dataset)*0.9)
    len_test_set = len(dataset)-len_train_set
    train_set, test_set = torch.utils.data.random_split(dataset, [len_train_set,len_test_set])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True)

    #model
    model = CongestionModel(device).to(device)

    #criterion
    ssim = SSIM(data_range=1, size_average=True, channel=1)
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print('Start training')
    train_losses = []
    valid_losses = []
    best_test_Loss = 99999999999999
    best_train_Loss = 99999999999999

    for e in tqdm(range(num_epochs), desc='Epoch'):
        t = 0
        n1 = 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device=device)
            labels = labels.to(device=device)

            if use_amp:
                with torch.cuda.amp.autocast():
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

            t += train_loss.item()
            n1 += 1
        train_losses.append(t/n1)



        #eval
        model.eval()
        v = 0
        n2 = 0
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(device=device)
            labels = labels.to(device=device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(features)
                    pred = model.sigmoid(pred)
                    test_loss = 1.0 - ssim(pred.float(), labels.float())
            else:
                pred = model(features)
                pred = model.sigmoid(pred)
                test_loss = 1.0 - ssim(pred.float(), labels.float())

            v += test_loss.item()
            n2 += 1
        valid_losses.append(v/n2)


        print("\n")
        print(f'Epoch {e}: Train Loss: {t/n1}  | Test Loss: {v/n2}')
        if v/n2 < best_test_Loss:
            print(f'Best Epoch {e}: Test Loss: {v/n2}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_weights.pt')
            best_test_Loss = v/n2
        if t/n1 < best_train_Loss:
            print(f'Best Epoch {e}: Train Loss: {t/n1}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_train_weights.pt')
            best_train_Loss = t/n1

        fig = plt.figure()
        epochnum = list(range(0,len(train_losses)))
        plt.plot(epochnum, train_losses, color='black', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, len(train_losses))
        plt.legend("Train", loc='best',fontsize=16)
        plt.title("Train Loss")
        plt.grid(linestyle=':')
        plt.savefig(f"{fig_savepath}/train_losses.png")
        plt.clf()

        fig = plt.figure()
        epochnum = list(range(0,len(train_losses)))
        # plt.plot(epochnum, train_losses, color='black', linewidth=1)
        plt.plot(epochnum, valid_losses, color='red', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, len(train_losses))
        plt.legend(("Val"), loc='best',fontsize=16)
        plt.title("Val Loss")
        plt.grid(linestyle=':')
        plt.savefig(f"{fig_savepath}/val_losses.png")
        plt.clf()


        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        pred = model.sigmoid(pred)
        ax[0].imshow(pred[0,0].detach().cpu())
        ax[1].imshow(labels[0,0].cpu())
        ax[0].title.set_text('Pred')
        ax[1].title.set_text('Label')
        plt.savefig(f"{fig_savepath}/compare.png")
        plt.clf()

def parse_args():
    parser = argparse.ArgumentParser(description="Librelane Congestion Model Training")
    parser.add_argument("--root_path", default="./datasets/training", type=str, help='The path of the data file')
    parser.add_argument("--batch_size", default=8, type=int, help='The batch size')
    parser.add_argument("--num_epochs", default=1000, type=int, help='The training epochs')
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
    print("training cost time：%f sec" % (end - start))
