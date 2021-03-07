import torch
import os
import random
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from piq import SSIMLoss, MultiScaleGMSDLoss
from torch.nn import MSELoss
from sklearn.metrics import auc, roc_curve
from torchvision.utils import make_grid
from PIL import Image
from utils.msgmsd_map import msgmsd
from scipy.ndimage import gaussian_filter

from wrappers.dataset_selector import DatasetSelector
from models.unet import UNet
from models.pixelcnn import PixelCNN
from models.vision_transformer import ViTAE
from inpating_utils import MSGMS_Score
from kornia.filters.median import median_blur

seed = 67
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def gen_mask(k_list, n, im_size, return_global_mask=False):
    while True:
        Ms = []
        for k in k_list:
            N = im_size // k
            rdn = np.random.permutation(N**2)
            additive = N**2 % n
            if additive > 0:
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive))))
            n_index = rdn.reshape(n, -1)
            for index in n_index:
                tmp = [0 if i in index else 1 for i in range(N**2)]
                tmp_global = np.asarray(tmp).reshape(N, N)
                tmp = tmp_global.repeat(k, 0).repeat(k, 1)
                if return_global_mask:
                  Ms.append((tmp, tmp_global))
                else:
                  Ms.append(tmp)
        yield Ms


def train(args, model, epoch, train_loader, optimizer):
    model =  model.train()

    ssim = SSIMLoss()
    mse = MSELoss()
    msgms = MultiScaleGMSDLoss(scale_weights=[1, 1, 1, 1])

    pbar = tqdm(total=len(train_loader))
    epoch_losses = []
    for ix, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)

        k_value = random.sample(args.k_value, 1)
        Ms_generator = gen_mask(k_value, 3, args.image_size)
        Ms = next(Ms_generator)

        inputs = [x * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
        outputs = [model(x) for x in inputs]
        output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))

        l2_loss = mse(x, output)
        gms_loss = msgms(x, output)
        ssim_loss = ssim(x, output)

        loss = args.gamma * l2_loss + args.alpha * gms_loss + args.belta * ssim_loss

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        pbar.set_description(f'Epoch: {epoch} Step loss: {loss.item():.4f}')
        pbar.refresh()
        pbar.update()
        print(f'AUC: {np.mean(epoch_losses):.6f}', file=open('train_epoch_losses.txt', 'a+'))
AUC: 0.966017

def compute_anomaly_score(args, x, model):
    maps = []
    for k in args.k_value:
        img_size = x.size(-1)
        N = img_size // k
        Ms_generator = gen_mask([k], 3, img_size)
        Ms = next(Ms_generator)
        inputs = [x * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
        outputs = [model(x) for x in inputs]
        output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
        msgms_map = msgmsd(output, x)
        maps.append(median_blur(msgms_map, (21, 21)) )
    computed_maps = torch.cat(maps, dim=1).mean(dim=1, keepdim=True)
    return computed_maps, output


def val_from_other(args, model, epoch, test_loader):
    model.eval()
    scores = []

    index = 0
    y_true = []
    for (data, y) in tqdm(test_loader, desc='Evaluating'):
        y_true += y.tolist()
        data = data.to(device)

        with torch.no_grad():
            maps, outputs = compute_anomaly_score(args, data, model)
            score = maps.max()
            scores.append(score.item())

        if index == 1:
            maps = maps[0][0].detach().cpu().numpy()
            heat_map = maps * 255
            heat_map = heat_map.astype(np.uint8)
            
            img = data[0].detach().cpu().numpy()
            img = img.transpose((1, 2, 0))
            img = img * 255
            img = img.astype(np.uint8)
            img_cmap = None
            if img.shape[-1] == 1:
                img = img.reshape(img.shape[:2])
                img_cmap = 'gray'
            plt.imshow(img, cmap=img_cmap)
            plt.imshow(heat_map, cmap='jet', alpha=0.2, interpolation='none')

            plt.savefig(f'heat_map_{epoch}.jpeg')
            plt.clf()
            plt.close('all')

            x = outputs[0].detach().cpu().numpy()
            x = x.transpose((1, 2, 0))
            if x.shape[-1] == 1:
                x = x.reshape(img.shape[:2])
            x = x * 255
            x = x.astype(np.uint8)
            plt.imshow(x, cmap=img_cmap)
            plt.savefig(f'x_{epoch}.jpeg')
            plt.clf()
            plt.close('all')
            
        index += 1

    plt.hist(scores)
    plt.savefig(f'histogram_validation_{epoch}.jpeg')
    plt.clf()
    plt.close('all')

    fpr, tpr, threshold = roc_curve(y_true, scores)
    auc_score = auc(fpr, tpr)
    print(f'AUC: {auc_score.item():.6f}', file=open('aucs.txt', 'a+'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RIAD anomaly detection')
    parser.add_argument('--pdata', type=float, default=.05, help='percentage of data used in traning')
    parser.add_argument('--obj', type=str, default='cable')
    parser.add_argument('--dataset', type=str, default='kaggle_pneumonia')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64) # 256
    parser.add_argument('--grayscale', action='store_true', help='color or grayscale input image')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--belta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of Adam')
    parser.add_argument('--seed', type=int, default=67, help='manual seed')
    parser.add_argument('--k_value', type=int, nargs='+', default=[2, 4, 8, 16]) 
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--in_cls', default=0, type=int)
    args = parser.parse_args()

    channels = 3 if args.dataset == 'cifar-10' or args.dataset == 'mvtech-ad' else 1

    model = UNet(in_channels=channels, out_channels=channels).to(device)
    # model = PixelCNN(input_channels=channels).to(device)
    # model = ViTAE(
    #     image_size = args.image_size,
    #     patch_size = 4,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 8,
    #     mlp_dim = 2048,
    #     dropout = 0.5,
    #     emb_dropout = 0.5,
    #     channels=channels
    # ).to(device)

    print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(model.parameters()))}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader, _ = DatasetSelector.select_dataset(args)

    for epoch in range(args.epochs):
        train(args, model, epoch+1, train_loader, optimizer)
        val_from_other(args, model, epoch+1, val_loader)