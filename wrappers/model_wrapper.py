import torch
import os
import pytorch_lightning as ptl
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from wrappers.loss_selector import LossSelector
from wrappers.model_selector import ModelSelector
from utils.anomaly_scores import compute_gms_anomaly_score

class ModelWrapper(ptl.LightningModule):
    def __init__(self, hparams):
        super(ModelWrapper, self).__init__()
        self.hparams = hparams
        self.net = ModelSelector.select_model(hparams)
        self.loss_function = LossSelector.select_loss(hparams)

    def forward(self, x):
        
        return self.net(x)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, _ = batch
        # import pdb; pdb.set_trace()
        results = self(x)

        M_N = self.hparams.batch_size / ( len(self.train_dataloader()) * self.hparams.batch_size )

        if self.hparams.model == 'vae':
            loss = self.net.loss_function(
                *results, 
                M_N = M_N,
                optimizer_idx = optimizer_idx,
                batch_idx = batch_idx
            )
        else:
            loss = self.loss_function(results, x)
            # loss = compute_gms_anomaly_score(recon_batch, x)

        return {'loss': loss}

    def get_anomaly_score(self):
        # import pdb; pdb.set_trace()
        attn_idx = 0
        # patch_size = self.net.patch_size
        layer_idx = 5
        # for layer_idx in range(nlayers):
        attn_weights = self.net.transformer.layers[layer_idx][attn_idx].fn.fn.attention_weights[0]
        # nheads = attn_weights.shape[0]
        # hdim = attn_weights.shape[1]
        # attn_lhm = attn_weights.sum(dim=0) / nheads
        # attn_lhm = attn_lhm.sum(dim=0) / hdim
        # return attn_lhm.mean(dim=-1)
        return attn_weights.max()

    def plot_attention_weights(self, batch_idx):

        # import pdb; pdb.set_trace()

        nlayers = len(self.net.transformer.layers)
        attn_idx = 0
        patch_size = self.net.patch_size
        attns = []
        for layer_idx in range(nlayers):
            attn_weights = self.net.transformer.layers[layer_idx][attn_idx].fn.fn.attention_weights[0]
            attns.append(attn_weights)

            # nheads = attn_weights.shape[0]
            # hdim = attn_weights.shape[1]
            # attn_lhm = attn_weights.sum(dim=0) / nheads
            # attn_lhm = attn_lhm.sum(dim=0) / hdim
            # attn_lhm = attn_lhm.detach().cpu().numpy()
            # attn_lhm = attn_lhm[1:]
            # attn_lhm = attn_lhm.reshape((patch_size, patch_size))
            # for head_idx in range(attn_weights.shape[0]):
            #     attn_lh = attn_weights[head_idx]
            #     attn_lh = attn_lh.detach().cpu().numpy()


        attn_lhm = torch.stack(attns).mean(1).mean(0).mean(0).detach().cpu().numpy()
        # attn_lhm = attn_lhm[1:]
        attn_lhm = attn_lhm.reshape((patch_size, patch_size))

        plt.imshow(attn_lhm)
        plt.savefig(f'attn_weights_{batch_idx}.jpeg')
        plt.clf()
        plt.close('all')
        

    def validation_step(self, batch, batch_idx, optimizer_idx =0):
        x, y = batch
        # import pdb; pdb.set_trace()
        recon_batch = self(x)
        
        M_N = self.hparams.batch_size / ( len(self.val_dataloader()) * self.hparams.batch_size )

        if self.hparams.model == 'vae':
            loss = self.net.loss_function(
                *recon_batch, 
                M_N = M_N,
                optimizer_idx = optimizer_idx,
                batch_idx = batch_idx
            )
        else:
            loss = self.loss_function(recon_batch, x)
            # loss = self.get_anomaly_score()
        
        # if (self.current_epoch == 20) and ( batch_idx == 0 or batch_idx == 43 ):

        #     import numpy as np
        #     img = recon_batch[0]
        #     # if img.shape[1] == 3:
        #     img = img.detach().cpu().numpy()
        #     img = img.transpose((1, 2, 0))
        #     # else:
        #         # img = img[0].detach().cpu().numpy()
        #     img = img * 255
        #     img = img.astype(np.uint8)
        #     plt.imshow(img)

        #     plt.savefig(f'recon_{batch_idx}.jpeg')
        #     plt.clf()
        #     plt.close('all')
        
        # if ( self.current_epoch == 20 ) and (batch_idx == 0 or batch_idx == 43 ):

        #     img = x[0]
        #     img = img.detach().cpu().numpy()
        #     img = img.transpose((1, 2, 0))
        #     img = img * 255
        #     img = img.astype(np.uint8)
        #     plt.imshow(img)
        #     plt.savefig(f'input_fig_{batch_idx}.jpeg')
        #     plt.clf()
        #     plt.close('all')
        #     self.plot_attention_weights(batch_idx)

        return {'val_loss': loss, 'x_hat': recon_batch, 'y_batch': y}

    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        x, y = batch

        results = self(x)
        
        M_N = self.hparams.batch_size / ( len(self.test_dataloader()) * self.hparams.batch_size )

        if self.hparams.model == 'vae':
            loss = self.net.loss_function(
                *results, 
                M_N = M_N,
                optimizer_idx=optimizer_idx,
                batch_idx = batch_idx
            )
        else:
            loss = self.loss_function(results, x)
            # loss = compute_gms_anomaly_score(recon_batch, x)

        return {'val_loss': loss, 'x_hat': results, 'y_batch': y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr, weight_decay=0.0001)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            'train_loss': avg_loss
        }
        return {'loss': avg_loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        return self.__calculate_metrics(outputs)

    def test_epoch_end(self, outputs):
        return self.__calculate_metrics(outputs, phase='test')

    def save_roc_fig(self, fpr, tpr, auc_score, phase='val'):

        log_dir = self.logger.log_dir
        roc_pictures = os.path.join(log_dir, 'roc_pictures')
        os.makedirs(roc_pictures, exist_ok=True)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(roc_pictures, f"roc_auc_{phase}_epoch_{self.current_epoch+1}.png"))
        plt.clf()
        plt.close('all')

    def __calculate_metrics(self, outputs, phase='val'):
        
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        iy = []
        iloss = []
        for output in outputs:
            iy.append(output['y_batch'].item())
            iloss.append(output['val_loss'].item())

        fpr, tpr, threshold = roc_curve(iy, iloss)
        auc_score = auc(fpr, tpr)
        self.save_roc_fig(fpr, tpr, auc_score, phase)

        t_auc_score = torch.tensor(auc_score)
        phase_auc = phase + '_auc'
        logs = {"val_loss": avg_loss, phase_auc: t_auc_score}
        return {phase_auc: t_auc_score, "val_loss": avg_loss, "log": logs}
