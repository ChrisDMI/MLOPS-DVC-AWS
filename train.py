import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

import hydra
from omegaconf import DictConfig, OmegaConf
 
# Initialize W&B logger
wandb_logger = WandbLogger(project="cola-classification")

def get_device():
    """Determine the best device available (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def setup_data(batch_size, max_length):
    """Set up the data module for training and validation.""" 
    data_module = DataModule(batch_size=batch_size, max_length=max_length)
    # Prepare and setup the data (ensures train and validation sets are loaded)
    data_module.prepare_dataset()
    data_module.setup(stage="fit")
    
    # Reduce dataset size to speed up training
    data_module.train_data = data_module.train_data.select(range(500))  # Only take 500 samples
    data_module.val_data = data_module.val_data.select(range(100))  # Only take 100 samples for validation
    return data_module

def setup_model(model_name):
    """Set up the model to be trained."""
    return ColaModel(model_name=model_name)

def setup_callbacks(model_dir="./models"):
    """Set up checkpointing and early stopping callbacks."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, 
        filename="best-checkpoint",
        monitor="valid/loss_epoch", 
        mode="min", 
        save_top_k=1  # Save only the best model
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss_epoch", patience=2, verbose=True, mode="min"
    )
    return [checkpoint_callback, early_stopping_callback]

def setup_trainer(callbacks, max_epochs, log_every_n_steps, deterministic, limit_train_batches, limit_val_batches):
    """Set up the PyTorch Lightning trainer with the best available device."""
    device = get_device()

    # Initialize the Trainer
    return pl.Trainer(
        default_root_dir="logs",
        accelerator=device,
        devices=1,
        max_epochs=max_epochs,
        logger=wandb_logger,  # Use WandbLogger
        callbacks=callbacks,
        precision=16 if device == "mps" else 32,  # Use mixed precision on Mac M1
        fast_dev_run=False,  # Set to True for a single batch run for quick debugging
        log_every_n_steps=log_every_n_steps,  # Log to W&B every 10 steps
        deterministic=deterministic,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        #check_val_every_n_epoch=1  # Avoid running validation before training
    )


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main function to set up data, model, and trainer and start training."""
    # Optional: Print the config for debugging purposes
    # Print the config with interpolation resolved
    print(OmegaConf.to_yaml(cfg))  # This will print 0.25 for limit_val_batches


    # Set up data, model, and trainer
    data_module = setup_data(batch_size=cfg.processing.batch_size, max_length=cfg.processing.max_length)
    model = setup_model(model_name=cfg.model.name)
    callbacks = setup_callbacks()
    trainer = setup_trainer(
        callbacks=callbacks, 
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )

    # Start training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()