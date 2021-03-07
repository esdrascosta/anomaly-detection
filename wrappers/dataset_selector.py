import os
import torch
import random
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
from datasets.pneumonia_dataset import PneumoniaDataset
from datasets.benchmarking_anomaly_dataset import AnomalyDataset
from datasets.mvtech_anomaly import MVTecAD
from torchvision import transforms

class DatasetSelector:

    @classmethod
    def select_dataset(cls, hparams, image_transforms=None):
        print(f'dataset={hparams.dataset}')

        if hparams.dataset == 'kaggle_pneumonia':

            if image_transforms is None:
                image_transformations = [
                    ToPILImage(),
                    Resize(hparams.image_size),
                    ToTensor()
                ]
            else:
                image_transformations = [image_transforms]
            dataset = PneumoniaDataset(transform=Compose(image_transformations))

            train_len = 6000 # default train size
            if hparams.pdata < 1.0:
                train_len = int(train_len * hparams.pdata)

            train_set, val_set, test_set = dataset.get_subsets(number_train_samples=train_len)

            train_loader = DataLoader(train_set, batch_size=hparams.batch_size, 
                                      shuffle=True, 
                                      num_workers=hparams.num_workers)
                                      
            val_loader = DataLoader(val_set, batch_size=hparams.val_batch_size, num_workers=hparams.num_workers)
            test_loader = DataLoader(test_set, batch_size=hparams.test_batch_size, num_workers=hparams.num_workers)
            return train_loader, val_loader, test_loader

        elif hparams.dataset == 'mnist' or hparams.dataset == 'cifar-10':

            dataset_dir = os.path.join('data', hparams.dataset, 'splits')
            in_channel = 3 if hparams.dataset == 'cifar-10' else 1
            
            image_transformations = [
                Resize(hparams.image_size),
                ToTensor()
            ]

            train_set = AnomalyDataset(dataset_dir, split='train', in_channel=in_channel,
                                       transform=Compose(image_transformations),
                                       target_transform=transforms.ToTensor(),
                                       inlier_class=hparams.in_cls)

            val_set = AnomalyDataset(dataset_dir, split=f'test_{hparams.in_cls}', in_channel=in_channel,
                                     transform=Compose(image_transformations),
                                     target_transform=transforms.ToTensor(),
                                     inlier_class=hparams.in_cls)


            train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True,
                                                       num_workers=hparams.num_workers)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=hparams.val_batch_size,
                                                     num_workers=hparams.num_workers)

            return train_loader, val_loader, None
        
        elif hparams.dataset == 'mvtech-ad':
            # if image_transforms is None:
            image_transformations = [
                # ToPILImage(),
                Resize(hparams.image_size),
                ToTensor()
            ]
            # else:
                # image_transformations = [image_transforms]

            obj = hparams.obj if hparams.obj else None
                
           
            train_set = MVTecAD(transform=Compose(image_transformations), mode='train', obj=obj)
            val_set = MVTecAD(transform=Compose(image_transformations), mode='test', obj=obj)

            # import pdb; pdb.set_trace()

            subset = hparams.pdata < 1.0
            sampler = None

            if subset:
                data_len = len(train_set)
                num_samples = int(data_len * hparams.pdata)
                idxs = random.sample(list(range(data_len)), k=num_samples)
                sampler = torch.utils.data.SubsetRandomSampler(idxs)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.batch_size, shuffle=(not subset),
                                                       num_workers=hparams.num_workers, sampler=sampler, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=hparams.val_batch_size,
                                                     num_workers=hparams.num_workers)

            return train_loader, val_loader, None
