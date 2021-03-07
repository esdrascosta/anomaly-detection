import csv
import os
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.core.saving import save_hparams_to_yaml
from typing import Optional, Union


class CustomLogger(LightningLoggerBase):
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(self, save_dir: str,
                 name: Optional[str] = "default",
                 version: Optional[Union[int, str]] = None,
                 **kwargs):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ''
        self._version = version
        self.hparams = {}

        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def root_dir(self):
        if self.name is None or len(self.name) == 0:
            return self.save_dir
        else:
            return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self):
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self.save_dir, self.name)

        if not os.path.isdir(root_dir):
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            with open(os.path.join(self.log_dir, f'{k}.csv'), 'a+') as f:
                print(v, file=f)

    @property
    def experiment(self):
      pass

    @property
    def name(self):
        return self._name

    @rank_zero_only
    def save(self):
        super().save()
        dir_path = self.log_dir
        if not os.path.isdir(dir_path):
            dir_path = self.save_dir

        # prepare the file path
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file
        save_hparams_to_yaml(hparams_file, self.hparams)

    @rank_zero_only
    def finalize(self, status: str):
        self.save()

    @rank_zero_only
    def log_hyperparams(self, params):
        params = self._convert_params(params)

        # store params to output
        self.hparams.update(params)
