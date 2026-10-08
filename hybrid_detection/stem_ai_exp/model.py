
import os
import logging

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import BinaryStatScores


class AiStemDetector(LightningModule):

    def __init__(self, input_size, n_hidden_layers=5, n_units=32):
        super().__init__()

        self.save_hyperparameters(logger=False)

        model = nn.Sequential()
        model.append(nn.Linear(input_size, n_units))
        model.append(nn.ReLU())
        for i in range(n_hidden_layers):
            model.append(nn.Linear(n_units, n_units))
            model.append(nn.ReLU())
        model.append(nn.Linear(n_units, 1))
        model.append(nn.Sigmoid())

        self.model = model

        self.loss_fn = nn.BCELoss()
        self.stat_scores = BinaryStatScores()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.stat_scores.update(y_hat, y)  # accumulate tp, fp, tn, fn
        return loss

    def on_train_epoch_end(self):
        tp, fp, tn, fn, _ = self.stat_scores.compute()

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        self.log("train_tpr", tpr, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_fpr", fpr, prog_bar=True, logger=True, sync_dist=True)

        self.stat_scores.reset()

    def configure_optimizers(self):
         return torch.optim.Adam(self.parameters(), lr=1e-3)

    @classmethod
    def load_from_checkpoint_dir(cls, dir_, **kwargs):
        file_list = sorted([f for f in os.listdir(dir_) if f.endswith('.ckpt')], reverse=True)
        if len(file_list) > 1:
            logging.warning(f'Found {len(file_list)} checkpoints in {dir_}\n'
                            f'Will load {file_list[0]} by default.')

        return cls.load_from_checkpoint(checkpoint_path=os.path.join(dir_, file_list[0]), **kwargs)
