import sys
import os

import unittest
from unittest.mock import patch, MagicMock
import torch
import pytorch_lightning as pl

# Add the parent directory (Project_Setup) to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import get_device, setup_data, setup_model, setup_callbacks, setup_logger, setup_trainer


class TestTrainingSetup(unittest.TestCase):

    def test_get_device_cuda(self):
        """Test that CUDA is selected if available."""
        with unittest.mock.patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(get_device(), "cuda")

    def test_get_device_mps(self):
        """Test that MPS is selected if CUDA is unavailable and MPS is available."""
        with unittest.mock.patch('torch.cuda.is_available', return_value=False):
            with unittest.mock.patch('torch.backends.mps.is_available', return_value=True):
                self.assertEqual(get_device(), "mps")

    def test_get_device_cpu(self):
        """Test that CPU is selected if neither CUDA nor MPS is available."""
        with unittest.mock.patch('torch.cuda.is_available', return_value=False):
            with unittest.mock.patch('torch.backends.mps.is_available', return_value=False):
                self.assertEqual(get_device(), "cpu")

    def test_setup_data(self):
        """Test that the data module is set up correctly."""
        data_module = setup_data()
        self.assertIsInstance(data_module, pl.LightningDataModule)

    def test_setup_model(self):
        """Test that the model is set up correctly."""
        model = setup_model()
        self.assertIsInstance(model, pl.LightningModule)

    def test_setup_callbacks(self):
        """Test that the callbacks are set up correctly."""
        callbacks = setup_callbacks()
        self.assertEqual(len(callbacks), 2)
        self.assertIsInstance(callbacks[0], pl.callbacks.ModelCheckpoint)
        self.assertIsInstance(callbacks[1], pl.callbacks.EarlyStopping)

    def test_setup_logger(self):
        """Test that the logger is set up correctly."""
        logger = setup_logger(log_dir="test_logs", name="test", version=1)
        self.assertIsInstance(logger, pl.loggers.TensorBoardLogger)
        self.assertEqual(logger.log_dir, "test_logs/test/version_1")

    @patch("pytorch_lightning.Trainer.fit")
    def test_setup_trainer(self, mock_trainer_fit):
        """Test that the trainer is set up correctly and fit is called."""
        data_module = MagicMock()
        model = MagicMock()

        callbacks = setup_callbacks()
        logger = setup_logger()
        trainer = setup_trainer(callbacks, logger, max_epochs=3)

        # Check if trainer is correctly set up
        self.assertIsInstance(trainer, pl.Trainer)
        self.assertEqual(trainer.max_epochs, 3)

        # Mock the fit function and check if it is called correctly
        trainer.fit(model, data_module)
        mock_trainer_fit.assert_called_once_with(model, data_module)


if __name__ == "__main__":
    unittest.main()