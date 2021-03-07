# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
import numpy as np
import subprocess
import os
import pandas as pd
import zipfile
import pydicom
import random


class PneumoniaDataset(Dataset):
    
    ZIP_FILE = 'data/rsna-pneumonia-detection-challenge.zip'
    DATASET_DIR = 'data/rsna-pneumonia-detection-challenge'
    IMAGE_PATH = "data/rsna-pneumonia-detection-challenge/stage_2_train_images/{}.dcm"
    TRAIN_LABELS_FILE = 'data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
    DETAIL_CLASS_INFO_FILE = 'data/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv'
    
    def __init__(self, transform=None, target_transform=None, download=False, only_lung_opacity=True, shuffle=True):
        
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.only_lung_opacity = only_lung_opacity
        self.shuffle = shuffle
        self.data = []

        if download and not self._check_download():
            self._download()

        self._process_classes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        patient_id, label = self.data[idx]

        img = pydicom.read_file(self.IMAGE_PATH.format(patient_id)).pixel_array
        img = np.expand_dims(img, axis=2)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)


        return img, label

    def _process_classes(self):
        train_labels = pd.read_csv(self.TRAIN_LABELS_FILE)
        class_info = pd.read_csv(self.DETAIL_CLASS_INFO_FILE)
        all_dataset = pd.merge(train_labels, class_info, on='patientId')
        normal_cases = all_dataset[all_dataset['class'] == 'Normal']['patientId'].to_list()
        if self.only_lung_opacity:
            abnormal_cases = all_dataset[all_dataset['class'] == 'Lung Opacity']['patientId'].to_list()
        else:
            abnormal_cases = all_dataset[all_dataset['class'] != 'Normal']['patientId'].to_list()

        normal_cases_tuples = [(x, 0) for x in normal_cases]
        abnormal_cases_tuples = [(x, 1) for x in abnormal_cases]
        self.data = normal_cases_tuples + abnormal_cases_tuples

    def _download(self):
        downloaded_dataset = subprocess.call(['kaggle', 'competitions', 'download', '-c',
                                              'rsna-pneumonia-detection-challenge', '-p', 'data'])
        if downloaded_dataset:
            print('download failed! try download manually')
            return
        
        print(f"Extracting: {self.ZIP_FILE}")

        with zipfile.ZipFile(self.ZIP_FILE) as f:
            f.extractall(self.DATASET_DIR)
        os.unlink(self.ZIP_FILE)   
    
    def _check_download(self):
        return (os.path.exists(self.ZIP_FILE) and os.path.isfile(self.ZIP_FILE)) or \
               (os.path.exists(self.DATASET_DIR) and os.path.isdir(self.DATASET_DIR))

    def get_subsets(self, number_train_samples=6000):
        normal_cases = set()
        abnormal_cases = set()

        for i, (_, abnormal) in enumerate(self.data):
            if abnormal:
                abnormal_cases.add(i)
            else:
                normal_cases.add(i)

        # draw 6000 samples from normal cases
        # all samplings operations are without replacement
        # i.e. we need to remove the remaining ones from population
        train_sample = {*random.sample(normal_cases, k=number_train_samples)}
        normal_cases -= train_sample

        # draw 1025 samples from normal cases
        validation_normal_cases = {*random.sample(normal_cases, k=1025)}
        normal_cases -= validation_normal_cases

        # draw 1025 samples from abnormal cases
        validation_abnormal_cases = {*random.sample(abnormal_cases, k=1025)}
        abnormal_cases -= validation_abnormal_cases

        # draw 1000 samples from normal cases
        test_normal_cases = {*random.sample(normal_cases, k=1000)}
        normal_cases -= test_normal_cases

        # draw 1000 samples from abnormal cases
        test_abnormal_cases = {*random.sample(abnormal_cases, k=1000)}
        abnormal_cases -= test_abnormal_cases

        train_indexes = list(train_sample)
        validation_indexes = list(validation_abnormal_cases | validation_normal_cases)
        test_indexes = list(test_abnormal_cases | test_normal_cases)

        return Subset(self, train_indexes), Subset(self, validation_indexes), Subset(self, test_indexes)
