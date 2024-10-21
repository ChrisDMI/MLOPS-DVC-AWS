import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", max_length=512, batch_size=32):
        """Initializes the DataModule with the tokenizer and batch size."""
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_data = None
        self.val_data = None 

    def prepare_dataset(self):
        """Loads the CoLA dataset from Hugging Face's GLUE benchmark."""
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]
        print(f"Dataset loaded. Train samples: {len(self.train_data)}, Val samples: {len(self.val_data)}")


    def tokenize_data(self, examples):
        """Tokenizes the input sentences in the dataset."""
        return self.tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    def setup(self, stage=None):
        """Processes the datasets according to the stage (train or validation)."""
        if stage == "fit" or stage is None:
            print(f"Setup called. Train data: {self.train_data}, Validation data: {self.val_data}")

            self.train_data = self._process_and_format(self.train_data)
            self.val_data = self._process_and_format(self.val_data)

    def _process_and_format(self, dataset):
        """Helper method that applies tokenization and formatting to the dataset."""
        dataset = dataset.map(self.tokenize_data, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return dataset
    
    def train_dataloader(self):
        """Create a DataLoader for the training set"""
        return self._create_dataloader(self.train_data, shuffle=True)
    
    def val_dataloader(self):
        """Create a DataLoader for the validation set."""
        return self._create_dataloader(self.val_data, shuffle=False)
    
    def _create_dataloader(self, dataset, shuffle):
        """Helper method to create a DataLoader for a given dataset."""
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )


if __name__ == "__main__":
    data_module = DataModule()
    data_module.prepare_dataset()
    data_module.setup()

    # Fetch a batch and print the input shape of tokenized sentences
    sample_batch = next(iter(data_module.train_dataloader()))
    print(sample_batch["input_ids"].shape, sample_batch['attention_mask'].shape, sample_batch['label'].shape)

