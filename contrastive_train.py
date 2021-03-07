import torch
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F
import kornia.augmentation as K
import kornia
import torchvision
import argparse
from wrappers.dataset_selector import DatasetSelector
from piq import SSIMLoss, GMSDLoss, MultiScaleGMSDLoss
from vit_pytorch import ViT
from byol_pytorch import BYOL

from torchvision import models

from sklearn.metrics import auc, roc_curve, recall_score, precision_score
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from utils.train_utils import AverageMeter
from models.unet import UNet
import pdb
from PIL import Image

seed = 88
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def load_image(img_path):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def plot_images(image_path, learner, epoch, label):

    image = load_image(image_path)

    x0 = T.Resize(args.image_size)(image)
    x0 = T.ToTensor()(x0)

    # x0 = T.RandomErasing(p=1.0)(x0)
    
    img = x0.detach().cpu().numpy()
    img = img.transpose((1, 2, 0))

    plt.imshow(img)
    plt.savefig(f'result_images/refimage_{epoch}_{label}.jpeg')
    plt.clf()
    plt.close('all')
    with torch.no_grad():

        x0 = T.Normalize( 
            mean=torch.tensor([0.485, 0.456, 0.406]), 
            std=torch.tensor([0.229, 0.224, 0.225]))(x0)


        learner = learner.eval()
        x0 = x0.cuda()
        emb = learner.online_encoder(x0.unsqueeze(0), return_embedding=True)
        plot_attention_weights(learner, epoch, label)

    learner = learner.train()

def train(args, learner, optimizer, loader, epoch):

    losses = AverageMeter(f"Epoch {epoch +1}")
    learner = learner.train()

    # ssim = SSIMLoss()
    local_progress= tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for idx, (x, _) in enumerate(local_progress):
        x = x.to(device)

        loss = learner(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.size(0))

        data_dict = {"avg loss": losses.avg}
        local_progress.set_postfix(data_dict)
    
        # if idx == 40:
            # ano_image = '/home/esdras/research/research_project/data/mvtech_anomaly/screw/test/thread_side/005.png'
            # normal_image = '/home/esdras/research/research_project/data/mvtech_anomaly/screw/test/good/005.png'
            # plot_images(ano_image, learner, epoch, 'ano_screw')
            # plot_images(normal_image, learner, epoch, 'normal_screw')

    return losses.avg

# def train_decoder(args, generator, learner, optimizer_g, loader, epoch):
#     learner = learner.eval()
#     generator = generator.train()

#     local_progress = tqdm(loader, desc=f'G Epoch {epoch+1}/{args.epochs}')
#     for idx, (x, y) in enumerate(local_progress):
#         x = T.Resize(args.image_size)(x)
#         optimizer_g.zero_grad()
#         x = x.to(device)
#         bz = x.size(0)
#         optimizer.zero_grad()

#         with torch.no_grad():
#             features = learner.online_encoder(x, return_embedding=True)
#             features = features.view(bz, 1, 13, 13)

#         x_hat = generator(features.detach())

#         loss = F.binary_cross_entropy_with_logits(x_hat, x)
#         loss.backward()
#         optimizer_g.step()

#         data_dict = {"recon_loss":loss.item()}
#         local_progress.set_postfix(data_dict)


#         if idx == 3:
#             img = x_hat[0].detach().cpu().numpy()
#             img = img.transpose((1, 2, 0))
#             img = img * 255
#             img = img.astype(np.uint8)
#             img_cmap = None
#             if img.shape[-1] == 1:
#                 img = img.reshape(img.shape[:2])
#                 img_cmap = 'gray'
#             plt.imshow(img, cmap=img_cmap)
#             plt.savefig(f'reconstruction_{epoch}.jpeg')
#             plt.clf()
#             plt.close('all')

#             img = x[0].detach().cpu().numpy()
#             img = img.transpose((1, 2, 0))
#             img = img * 255
#             img = img.astype(np.uint8)
#             img_cmap = None
#             if img.shape[-1] == 1:
#                 img = img.reshape(img.shape[:2])
#                 img_cmap = 'gray'
#             plt.imshow(img, cmap=img_cmap)
#             plt.savefig(f'input_{epoch}.jpeg')
#             plt.clf()
#             plt.close('all')


def get_features(model, dataloader):
    extracted_features, labels = [], []
    with torch.no_grad():
        # extract features
        for x, y in dataloader:
            x = T.Resize(args.image_size)(x)
            x = x.to(device)
            
            features = model.online_encoder(x, return_embedding=True)

            extracted_features += list(features)
            labels += list(y)

        labels = np.array(labels)
    
    out_dim = extracted_features[0].size(-1)
    return torch.stack(extracted_features).reshape(-1, out_dim).to(device), labels


def val(args, model, train_dataloader, val_dataloader, epoch):
    

    group_lasso = LedoitWolf(assume_centered=False)

    epslon= 1e-10
    model = model.eval()

    # get features
    train_features, _ = get_features(model, train_dataloader)
    val_features, labels = get_features(model, val_dataloader)
    # pdb.set_trace()

    train_features = F.normalize(train_features, dim=-1, p=2)
    val_features = F.normalize(val_features, dim=-1, p=2)

    # mu = train_features.mean(dim=0)
    # pdb.set_trace()

    cov = group_lasso.fit(train_features.cpu().numpy())
    # cov_matrix = torch.from_numpy(cov.covariance_).cuda()
    
    # pdb.set_trace()

    # pdb.set_trace()
    # m = torch.distributions.MultivariateNormal(mu, cov_matrix)
    
    # scores = m.log_prob(val_features)
    scores = cov.mahalanobis(val_features.cpu().numpy())
    fpr, tpr, threshold = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    return auc_score

def train_model(args, model, train_dataloader, val_dataloader):
    model = model.to(device)
    base_lr = 0.05
    # lr = base_lr*args.batch_size/256
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    
    # generator = generator.to(device)
    best_auc = 0
    for epoch in range(args.epochs):
        avg_loss = train(args, model, optimizer, train_dataloader, epoch)
        auc_score = val(args, model, train_dataloader, val_dataloader, epoch)
        print(f'auc: {auc_score:.6f}')
        if auc_score > best_auc:
            best_auc = auc_score
            print(f'Saving Model AUC: {best_auc:.6f}')
            model_path = os.path.join(args.model_path)
            torch.save(model.state_dict(), model_path)

def get_scores(tmu, tcov, val_features):
    epslon = 1e-8
    
    # For any matrix A, A+ϵI will be invertible for sufficiently small nonzero |ϵ|.
    tcov += (epslon * torch.eye(tcov.size(0)).to(device)) 
    
    dist = torch.distributions.MultivariateNormal(tmu, tcov)

    dood = dist.log_prob(val_features)
    # import pdb; pdb.set_trace()
    return dood 

# def evaluate(args, model, train_loader, val_loader):
#     epslon= 1e-10
#     if args.model_path is None:
#         raise ValueError('model_path must be informed in eval mode')

#     model_path = os.path.join(args.model_path)

#     model.load_state_dict(torch.load(model_path))
#     model = model.eval()
#     model = model.to(device)


#     # get features
#     # train_features, _ = get_features(model, train_dataloader)
#     val_features, labels = get_features(model, val_loader)
#     #normalize data
#     # train_features /= torch.norm(train_features, dim=-1, keepdim=True) + epslon
#     # val_features /= torch.norm(val_features, dim=-1, keepdim=True) + epslon
    
#     # tmu = torch.mean(train_features, dim=0, keepdim=True)
#     # tcov = get_cov(train_features, bias=True)

#     import pdb; pdb.set_trace()
#     # get scores
#     scores = F.normalize(val_features, dim=1).max(dim=1) #get_scores(tmu, tcov, val_features)

    
#     fpr, tpr, threshold = roc_curve(labels, scores.detach().cpu().numpy())
#     auc_score = auc(fpr, tpr)
#     print(f'get ROC: {auc_score}')


def plot_attention_weights(learner, epoch, label):
    nlayers = len(learner.online_encoder.net.transformer.layers)
    attn_idx = 0
    attention_weight_list = []

    for layer_idx in range(nlayers):
        attn_weights = learner.online_encoder.net.transformer.layers[layer_idx][attn_idx].fn.fn.attention_weights[0]
        attention_weight_list.append(attn_weights)
    
    mask = torch.stack(attention_weight_list).mean(1).mean(0).mean(0).detach().cpu().numpy()
    mask = mask / mask.max()
    grid_size = int(np.sqrt(mask.shape[0]))
    mask = mask[1:].reshape((grid_size, grid_size))
   
    mask = T.Resize(256)(torch.from_numpy(mask).unsqueeze(0))

    plt.imshow(mask[0].numpy())
    plt.savefig(f'result_images/attn_weights_{epoch}_{label}.jpeg')
    plt.clf()
    plt.close('all')

def run(args):
    from utils.kornia_utils import GaussianBlur
    kornia_transforms = nn.Sequential(
        # K.ColorJitter(0.8, 0.8, 0.8, 0.2,  p = 0.3),
        # K.RandomGrayscale(p=0.2)
        K.RandomHorizontalFlip(p=.5),
        GaussianBlur((3, 3), (1.0, 2.0), p=0.2),
        K.RandomCrop((args.image_size, args.image_size), p=1),
        # K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])) # mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
    )

    transform = torch.nn.Sequential(
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((args.image_size, args.image_size)),
            # T.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5])),
    )

    in_channels = 3 if args.dataset == 'cifar-10' or args.dataset == 'mvtech-ad' else 1

    # model = ViT(
    #     image_size = 64,
    #     patch_size = 16,
    #     num_classes = 100,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 16,
    #     mlp_dim = 2048,
    #     pool='mean',
    #     # dropout = 0.1,
    #     # emb_dropout = 0.1,
    #     channels = in_channels
    # )

    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model = UNet(in_channels=1, out_channels=1)

    learner = BYOL(
        model,
        augment_fn=transform,
        image_size = 64,
        hidden_layer='avgpool', #hidden_layer= 'to_latent',
        use_momentum = False       # turn off momentum in the target encoder
    )

    train_dataloader, val_dataloader, test_dataloader = DatasetSelector.select_dataset(args)

    if args.eval:
        if test_dataloader is not None:
            eval_dataloader = test_dataloader
        else:
            eval_dataloader = val_dataloader

        evaluate(args, learner, train_dataloader, eval_dataloader)
    else:
        train_model(args, learner, train_dataloader, val_dataloader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RIAD anomaly detection')
    parser.add_argument('--pdata', type=float, default=.05, help='learning rate of Adam')
    parser.add_argument('--obj', type=str, default='screw')
    parser.add_argument('--model_path', default='saved_models/contrastive/best_model_ViT_BYOL_pneumonia.pt', type=str)
    parser.add_argument('--eval', default=False, type=bool)
    parser.add_argument('--dataset', type=str, default='kaggle_pneumonia') #kaggle_pneumonia
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64) # 256
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--belta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of Adam')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--in_cls', default=0, type=int)
    args = parser.parse_args()

    run(args)


    # x = torch.randn((3, 1, 13, 13))
    # decoder = Generator()
    # import pdb; pdb.set_trace()
    # decoder_out = decoder(x)
    # print('Dncoder out shap/e:', decoder_out.shape) # it would result in torch.Size([3, 1, 24, 24])
    