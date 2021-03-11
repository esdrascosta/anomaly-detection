from wrappers.model_wrapper import ModelWrapper
from utils.custom_logger import CustomLogger
from argparse import ArgumentParser
from wrappers.dataset_selector import DatasetSelector
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = Trainer.add_argparse_args(ArgumentParser())
    parser.add_argument('--pdata', type=float, default=1)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--val_batch_size', default=1, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--loss', default='BCEWithLogitsLoss', type=str)
    parser.add_argument('--loss_backbone', default='vgg16', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model', default='unet', type=str)
    parser.add_argument('--dataset', default='mvtech-ad', type=str)
    parser.add_argument('--in_cls', default=0, type=int)
    parser.add_argument('--obj', type=str, default='cable')
    args = parser.parse_args()
    seed_everything(42)

    logger = CustomLogger('training_results/', name=args.model)
    model = ModelWrapper(hparams=args)
    checkpoint_callback = ModelCheckpoint(filepath=f'training_results/{args.model}/model_checkpoints/',
                                          monitor='val_auc',
                                          mode='max')

    # select dataset
    train, val, test = DatasetSelector.select_dataset(args)

    trainer = Trainer.from_argparse_args(args,
                                         gpus=1,
                                         deterministic=True,
                                         max_epochs=args.epochs,
                                         logger=logger,
                                         checkpoint_callback=checkpoint_callback)

    trainer.fit(model, train_dataloader=train, val_dataloaders=val)

    if test is not None:
        trainer.test(test_dataloaders=test, ckpt_path=checkpoint_callback.best_model_path)
