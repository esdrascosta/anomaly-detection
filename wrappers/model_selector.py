from models.pixel_snail import PixelSNAIL
from models.pixelcnn_residual import ResidualPixelCNN
from models.pixelcnn import PixelCNN
from models.vae import VAE
from models.unet2 import UNet
from models.vision_transformer import ViTAE

class ModelSelector:

    @classmethod
    def select_model(cls,  hparams):

        in_channel = 3 if hparams.dataset == 'cifar-10' or hparams.dataset == 'mvtech-ad' else 1
        if hparams.model == 'pixel_snail':
            return PixelSNAIL(image_dims=(in_channel, hparams.image_size, hparams.image_size), attn_n_layers=1)
        elif hparams.model == 'residual_pixel_cnn':
            return ResidualPixelCNN(input_channels=in_channel)
        elif hparams.model == 'pixel_cnn':
            return PixelCNN(input_channels=in_channel)
        elif hparams.model == 'vae':
            return VAE(image_channels=in_channel, latent_dim=128, hidden_dims=[8, 16, 32, 64])
        elif hparams.model == 'unet':
            return UNet(n_channels=in_channel)
        elif hparams.model == 'vitae':
            return ViTAE(
                        image_size = hparams.image_size,
                        patch_size = 16,
                        dim = 1024,
                        depth = 6,
                        heads = 8,
                        mlp_dim = 1024,
                        # dropout = 0.5,
                        # emb_dropout = 0.5,
                        channels=in_channel
                    )
