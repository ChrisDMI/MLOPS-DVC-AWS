import sys
import os
import unittest
from transformers import AutoTokenizer, BertTokenizerFast
from datasets import Dataset
import torch



# Add the parent directory (Project_Setup) to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import DataModule  # Assuming your DataModule is defined in data.py


class TestDataModule(unittest.TestCase):
    def setUp(self):
        """Set up the test environment by creating an instance of DataModule."""
        self.data_module = DataModule(
            model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32
        )
    
    def test_initialization(self):
        """Test if the DataModule is initialized properly."""
        # Check if tokenizer is loaded correctly
        #self.assertIsInstance(self.data_module.tokenizer, AutoTokenizer)
        self.assertIsInstance(self.data_module.tokenizer, BertTokenizerFast)
        # Check if batch size is set correctly
        self.assertEqual(self.data_module.batch_size, 32)
        # Check if train_data and val_data are initialized to None
        self.assertIsNone(self.data_module.train_data)
        self.assertIsNone(self.data_module.val_data)

    def test_prepare_dataset(self):
        """Test if the dataset is loaded correctly."""
        self.data_module.prepare_dataset()
        # Check if train_data and val_data are loaded as DatasetDict objects
        self.assertIsNotNone(self.data_module.train_data)
        self.assertIsNotNone(self.data_module.val_data)
        self.assertIn("sentence", self.data_module.train_data.column_names)
        self.assertIn("sentence", self.data_module.val_data.column_names)

    def test_tokenize_data(self):
        """Test the tokenization process."""
        sample_sentence = {"sentence": "This is a test sentence."}
        tokenized = self.data_module.tokenize_data(sample_sentence)
        
        # Check if the tokenization returns the expected fields
        self.assertIn("input_ids", tokenized)
        self.assertIn("attention_mask", tokenized)
        
        # Check the token length matches max_length (512)
        self.assertEqual(len(tokenized["input_ids"]), 512)
        self.assertEqual(len(tokenized["attention_mask"]), 512)
    
    def test_process_and_format(self):
        """Test if datasets are tokenized and formatted properly."""
        self.data_module.prepare_dataset()
        processed_data = self.data_module._process_and_format(self.data_module.train_data)
        
        # Check that the dataset is formatted to torch tensors
        self.assertEqual(processed_data.format["type"], "torch")
        self.assertIn("input_ids", processed_data.column_names)
        self.assertIn("attention_mask", processed_data.column_names)
        self.assertIn("label", processed_data.column_names)
        
        # Ensure it's still a dataset after processing
        #self.assertIsInstance(processed_data, DatasetDict)
        self.assertIsInstance(processed_data, Dataset)

    def test_dataloader_creation(self):
        """Test if DataLoader is created correctly."""
        self.data_module.prepare_dataset()
        self.data_module.setup()

        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()

        # Check that DataLoader returns correct batch size and tensor types
        train_batch = next(iter(train_dataloader))
        self.assertEqual(train_batch["input_ids"].shape[0], 32)  # batch size
        self.assertIsInstance(train_batch["input_ids"], torch.Tensor)
        self.assertIsInstance(train_batch["attention_mask"], torch.Tensor)
        self.assertIsInstance(train_batch["label"], torch.Tensor)

if __name__ == "__main__":
    unittest.main()