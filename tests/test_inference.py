import sys
import os

import unittest
from unittest.mock import patch, MagicMock
import torch

# Add the parent directory (Project_Setup) to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference import ColaPredictor, get_device


class TestColaPredictor(unittest.TestCase):
    
    def setUp(self):
        """Set up the test case with mock model checkpoint path and input sentence."""
        self.model_path = "./models/epoch=0-step=16.ckpt"
        self.sentence = "The boy is sitting on a bench."
        self.device = get_device()  # Ensure the device is passed correctly
        self.predictor = ColaPredictor(self.model_path, self.device)

    @patch('inference.ColaModel.load_from_checkpoint')
    def test_model_loading_success(self, mock_load_from_checkpoint):
        """Test that the model loads successfully from the checkpoint."""
        mock_model = MagicMock()
        mock_load_from_checkpoint.return_value = mock_model
        model = self.predictor.load_model(self.model_path)
        mock_load_from_checkpoint.assert_called_once_with(self.model_path)
        self.assertEqual(model, mock_model)

    @patch('inference.ColaModel.load_from_checkpoint')
    def test_model_loading_failure(self, mock_load_from_checkpoint):
        """Test that the model raises an error if loading fails."""
        mock_load_from_checkpoint.side_effect = Exception("Failed to load model")
        with self.assertRaises(RuntimeError):
            self.predictor.load_model(self.model_path)

    @patch('inference.ColaModel.load_from_checkpoint')
    @patch.object(ColaPredictor, 'preprocess')
    def test_preprocess(self, mock_preprocess, mock_load_from_checkpoint):
        """Test that the input text is tokenized and converted to tensors."""
        mock_preprocess.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], device=self.device),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], device=self.device)
        }
        processed_input = self.predictor.preprocess(self.sentence)
        mock_preprocess.assert_called_once_with(self.sentence)
        self.assertIn("input_ids", processed_input)
        self.assertIn("attention_mask", processed_input)
        self.assertIsInstance(processed_input["input_ids"], torch.Tensor)
        self.assertIsInstance(processed_input["attention_mask"], torch.Tensor)

    @patch('inference.ColaModel.load_from_checkpoint')
    @patch('torch.nn.Softmax.forward')  # Mock the Softmax forward function
    def test_predict(self, mock_softmax_forward, mock_load_from_checkpoint):
        """Test the predict function returns correct labels and scores."""
        # Mocking the model and softmax forward
        mock_model = MagicMock()
        mock_load_from_checkpoint.return_value = mock_model
        mock_model.return_value = torch.randn(1, 2, device=self.device)  # Mock the logits
        mock_softmax_forward.return_value = torch.tensor([[0.7, 0.3]], device=self.device)  # Mock softmax output
        
        predictor = ColaPredictor(self.model_path, self.device)
        result = predictor.predict(self.sentence)

        self.assertEqual(len(result), 2)  # Check for two predictions (acceptable, unacceptable)
        self.assertEqual(result[0]["label"], "unacceptable")
        self.assertEqual(result[1]["label"], "acceptable")
        self.assertAlmostEqual(result[0]["score"], 0.7)
        self.assertAlmostEqual(result[1]["score"], 0.3)

    @patch('inference.ColaModel.load_from_checkpoint')
    @patch('torch.nn.Softmax.forward')  # Mock the Softmax forward function
    def test_end_to_end_prediction(self, mock_softmax_forward, mock_load_from_checkpoint):
        """Test the full prediction process."""
        mock_model = MagicMock()
        mock_load_from_checkpoint.return_value = mock_model
        mock_model.return_value = torch.randn(1, 2, device=self.device)  # Mock logits
        mock_softmax_forward.return_value = torch.tensor([[0.8, 0.2]], device=self.device)  # Mock softmax output

        predictor = ColaPredictor(self.model_path, self.device)
        predictions = predictor.predict(self.sentence)
        
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]["label"], "unacceptable")
        self.assertEqual(predictions[1]["label"], "acceptable")
        self.assertAlmostEqual(predictions[0]["score"], 0.8)
        self.assertAlmostEqual(predictions[1]["score"], 0.2)


if __name__ == '__main__':
    unittest.main()