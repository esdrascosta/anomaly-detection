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
from vit_pytorch import ViT
from vit_pytorch.cross_vit import CrossViT
from contrastive_framework.byol import BYOL

from torchvision import models

from sklearn.metrics import auc, roc_curve, recall_score, precision_score
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, ShrunkCovariance
from utils.train_utils import AverageMeter
from models.unet import UNet
import pdb
from PIL import Image
import optuna
from vit_pytorch.nest import NesT

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

def train(args, learner, optimizer, loader, epoch, lr_scheduler=None):

    losses = AverageMeter(f"Epoch {epoch +1}")
    learner = learner.train()
    
    local_progress= tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for idx, (x, _) in enumerate(local_progress):
        optimizer.zero_grad()
        x = x.to(device)

        loss = learner(x)

        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        losses.update(loss.item(), x.size(0))

        data_dict = {"avg loss": losses.avg}
        local_progress.set_postfix(data_dict)

    return losses.avg

def get_features(model, dataloader):
    extracted_features, labels = [], []
    with torch.no_grad():
        # extract features
        for x, y in dataloader:
            x = T.Resize(args.image_size)(x)
            x = x.to(device)
           
            _, features = model(x, return_embedding=True)

            extracted_features += list(features)
            labels += list(y)

        labels = np.array(labels)
    
    
    out_dim = extracted_features[0].size(-1)
    return torch.stack(extracted_features).reshape(-1, out_dim).to(device), labels


def val(args, model, train_dataloader, val_dataloader, epoch):

    group_lasso = LedoitWolf(assume_centered=False)

    model = model.eval()

    train_features, _ = get_features(model, train_dataloader)
    val_features, labels = get_features(model, val_dataloader)

    train_features = F.normalize(train_features, dim=-1, p=2)
    val_features = F.normalize(val_features, dim=-1, p=2)
    cov = group_lasso.fit(train_features.cpu().numpy())
    # pdb.set_trace()
    scores = cov.mahalanobis(val_features.cpu().numpy())
    fpr, tpr, threshold = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    return auc_score

def train_model(args, model, train_dataloader, val_dataloader, trial=None):
    model = model.to(device)
    
    print(args)

    # if args.optname in ["SGD"]:
    #     optimizer = getattr(torch.optim, args.optname)(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # else:
    optimizer = getattr(torch.optim, args.optname)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, args.epochs * len(train_dataloader), 1e-4
    # )
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[5], gamma=0.1
    # )

    best_auc = 0
    for epoch in range(args.epochs):
        avg_loss = train(args, model, optimizer, train_dataloader, epoch)
        auc_score = val(args, model, train_dataloader, val_dataloader, epoch)

        if trial:
            trial.report(auc_score, epoch+1)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f'auc: {auc_score:.6f}')
        if auc_score > best_auc:
            best_auc = auc_score
            print(f'Saving Model AUC: {best_auc:.6f}')
            model_path = os.path.join(args.model_path)
            torch.save(model.state_dict(), model_path)

    return best_auc       

def run(args, trial=None):
    from utils.kornia_utils import GaussianBlur
    kornia_transforms = nn.Sequential(
        K.ColorJitter(0.8, 0.8, 0.8, 0.2,  p = 0.3),
        K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(p=.5),
        GaussianBlur((3, 3), (1.0, 2.0), p=0.2),
        K.RandomResizedCrop((args.image_size, args.image_size), p=.5),
        K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])) # )
    )

    # transform = torch.nn.Sequential(
    #         T.RandomHorizontalFlip(),
    #         RandomApply(
    #             T.GaussianBlur((3, 3), (1.0, 2.0)),
    #             p = 0.2
    #         ),
    #         RandomApply(
    #             T.RandomResizedCrop((args.image_size // 2, args.image_size // 2)),
    #             p = 0.5
    #         ),
    # )

    in_channels = 3 if args.dataset == 'cifar-10' or args.dataset == 'mvtech-ad' else 1

    # model = ViT(
    #     image_size = args.image_size,
    #     patch_size = 16,
    #     num_classes = 10,
    #     dim = 512, # 512
    #     depth = 6,
    #     heads = 16,
    #     mlp_dim = 1024, # 1024
    #     dropout = 0.5,
    #     emb_dropout = 0.1,
    #     channels = in_channels
    # )

    model =  NesT(
        image_size = args.image_size,
        patch_size = 4,
        dim = 96,
        heads = 3,
        num_hierarchies = 3,        # number of hierarchies
        block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
        num_classes = 512
    )


    # model = models.resnet50(pretrained=False)
    # model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    learner = BYOL(
        model,
        augment_fn=kornia_transforms,
        image_size = args.image_size,
        hidden_layer='mlp_head', #hidden_layer= 'to_latent',
        use_momentum = False       # turn off momentum in the target encoder
    )

    train_dataloader, val_dataloader, _ = DatasetSelector.select_dataset(args)

    best_auc = train_model(args, learner, train_dataloader, val_dataloader, trial)
    return best_auc

def objective(args):

    def final(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0, 0.9)
        momentum = trial.suggest_float("momentum", 0, 0.9)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        args.lr = lr
        # args.optname = optname
        args.amsgrad = amsgrad
        args.weight_decay = weight_decay
        args.momentum = momentum
        return run(args, trial)

    return final
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RIAD anomaly detection')
    parser.add_argument('--pdata', type=float, default=1.0, help='learning rate of Adam')
    parser.add_argument('--obj', type=str, default='screw')
    parser.add_argument('--model_path', default='saved_models/contrastive/best_model_resnet_mvtech', type=str)
    parser.add_argument('--eval', default=False, type=bool)
    parser.add_argument('--dataset', type=str, default='mvtech-ad') #kaggle_pneumonia
    parser.add_argument('--epochs', type=int, default=20, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=12) # 12
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256) # 256
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--belta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0006949058882671142, help='learning rate of Adam') #0.0006949058882671142
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--in_cls', default=0, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--optname', default='Adam', type=str)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--amsgrad', default=False, type=bool)

    args = parser.parse_args()

    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # study = optuna.create_study(direction="maximize", storage="sqlite:///mvtech_experiments.db", study_name="mvtech_cable_vit_adam", load_if_exists=True)
    # study.optimize(objective(args), n_trials=100)

    # # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    auc = run(args)

    with open(f'contrastive_results_{args.seed}.txt', 'a') as fl:
        print(f'obj={args.obj} auc: {auc:.3f}', file=fl)
