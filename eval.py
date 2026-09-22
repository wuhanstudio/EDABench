import argparse

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM
from loguru import logger
from tqdm import tqdm

from gpdl import GPDL
from congestion.model import CongestionModel
from congestion.dataloader import LibrelaneDataset

# feature channel order produced by LibrelaneDataset.feature_data: [pin, placement, rudy]
GPDL_CHANNEL_INDICES = [-1]  # rudy

def evaluate_gpdl_model(rootpath, model_path="models/circuitnet_10000.pth", batch_size=8, show=False):
    """Load the pretrained GPDL model and evaluate its predictions against the ground truth."""

    test_dataset = LibrelaneDataset(root_dir=rootpath + "/testing", transform=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = GPDL(in_channels=1, out_channels=1)
    model.init_weights(pretrained=str(model_path))
    model.eval()

    ssim = SSIM(data_range=1, size_average=True, channel=1)
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()

    total_ssim_loss = 0.0
    total_mse_loss = 0.0
    total_bce_loss = 0.0
    n = 0

    with torch.no_grad():
        for features, labels in tqdm(test_loader, total=len(test_loader), desc='Eval'):
            inputs = features[:, GPDL_CHANNEL_INDICES, :, :]
            pred = model(inputs)

            total_ssim_loss += (1.0 - ssim(pred.float(), labels.float())).item()
            total_mse_loss += mse(pred.float(), labels.float()).item()
            total_bce_loss += (bce(pred.float(), labels.float()) * 1000).item()
            n += 1

            if show:
                fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
                ax[0].imshow(pred[0, 0].cpu())
                ax[1].imshow(labels[0, 0].cpu())
                ax[0].title.set_text('Pred')
                ax[1].title.set_text('Label')
                plt.show()
                plt.close(fig)

    avg_ssim_loss = total_ssim_loss / n
    avg_mse_loss = total_mse_loss / n
    avg_bce_loss = total_bce_loss / n

    logger.info(f'GPDL model: SSIM Loss: {avg_ssim_loss}  | MSE Loss: {avg_mse_loss}  | BCE Loss: {avg_bce_loss}')
    return avg_ssim_loss, avg_mse_loss, avg_bce_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the RUDY heatmap and GPDL model as congestion baselines")
    parser.add_argument("--root_path", default="./datasets/", type=str, help='The path of the data file')
    parser.add_argument("--model_path", default="models/circuitnet_10000.pth", type=str, help='The path of the pretrained GPDL model weights')
    parser.add_argument("--batch_size", default=8, type=int, help='The batch size')
    parser.add_argument("--show", action="store_true", help='Display the predicted and ground-truth heatmap for each batch')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate_gpdl_model(rootpath=args.root_path, model_path=args.model_path, batch_size=args.batch_size, show=args.show)
