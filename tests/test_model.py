import sys
import os

import unittest
from unittest.mock import patch
import torch
from transformers import AutoModel

# Add the parent directory (Project_Setup) to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import ColaModel


class TestColaModel(unittest.TestCase):

    def setUp(self):
        """Set up the model and create a dummy input for testing."""
        self.model = ColaModel(model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2)
        
        # Dummy inputs for testing
        self.batch_size = 2
        self.seq_length = 10
        self.num_classes = 2

        self.dummy_input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        self.dummy_attention_mask = torch.ones((self.batch_size, self.seq_length))
        self.dummy_labels = torch.randint(0, self.num_classes, (self.batch_size,))

    @patch("model.AutoModel")
    def test_model_initialization(self, mock_bert):
        """Test that the model initializes correctly with given hyperparameters."""
        self.assertIsInstance(self.model, ColaModel)
        self.assertEqual(self.model.hparams["lr"], 1e-2)
        self.assertEqual(self.model.hparams["num_classes"], 2)
        self.assertIsInstance(self.model.classifier, torch.nn.Linear)

    @patch("model.AutoModel")
    def test_forward_pass(self, mock_bert):
        """Test that the forward pass returns logits with the correct shape."""
        # Mocking the output of BERT to have the correct shape
        mock_bert.return_value = torch.randn(self.batch_size, self.seq_length, self.model.bert.config.hidden_size)
        logits = self.model(self.dummy_input_ids, self.dummy_attention_mask)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))

    @patch("model.AutoModel")
    def test_compute_loss(self, mock_bert):
        """Test the compute_loss function for cross-entropy loss calculation."""
        mock_bert.return_value = torch.randn(self.batch_size, self.seq_length, self.model.bert.config.hidden_size)
        logits = self.model(self.dummy_input_ids, self.dummy_attention_mask)
        loss = self.model.compute_loss(logits, self.dummy_labels)
        self.assertIsInstance(loss, torch.Tensor)

    @patch("model.AutoModel")
    def test_compute_accuracy(self, mock_bert):
        """Test the compute_accuracy function for accuracy calculation."""
        mock_bert.return_value = torch.randn(self.batch_size, self.seq_length, self.model.bert.config.hidden_size)
        logits = self.model(self.dummy_input_ids, self.dummy_attention_mask)
        accuracy = self.model.compute_accuracy(logits, self.dummy_labels)
        self.assertIsInstance(accuracy, torch.Tensor)

    @patch("model.AutoModel")
    def test_training_step(self, mock_bert):
        """Test the training step to ensure loss is computed and logged."""
        batch = {
            "input_ids": self.dummy_input_ids,
            "attention_mask": self.dummy_attention_mask,
            "label": self.dummy_labels
        }
        mock_bert.return_value = torch.randn(self.batch_size, self.seq_length, self.model.bert.config.hidden_size)
        loss = self.model.training_step(batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)

    @patch("model.AutoModel")
    def test_validation_step(self, mock_bert):
        """Test the validation step to ensure loss and accuracy are computed and logged."""
        batch = {
            "input_ids": self.dummy_input_ids,
            "attention_mask": self.dummy_attention_mask,
            "label": self.dummy_labels
        }
        mock_bert.return_value = torch.randn(self.batch_size, self.seq_length, self.model.bert.config.hidden_size)
        self.model.validation_step(batch, batch_idx=0)

    @patch("model.AutoModel")
    def test_configure_optimizers(self, mock_bert):
        """Test that the optimizer is correctly configured."""
        optimizer = self.model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults['lr'], 1e-2)


if __name__ == "__main__":
    unittest.main()