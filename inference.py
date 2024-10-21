import torch
from model import ColaModel
from data import DataModule
from typing import List, Dict
from utils import timing


class ColaPredictor:
    def __init__(self, model_path: str, device: torch.device, labels: List[str] = None):
        """
        Initialize the ColaPredictor class.

        Args:
            model_path (str): Path to the saved model checkpoint.
            device (torch.device): The device on which to run the model (CPU, MPS, or CUDA).
            labels (List[str]): Optional list of labels for classification. Defaults to ["unacceptable", "acceptable"].
        """
        self.model_path = model_path
        self.device = device  # Store the device
        self.model = self.load_model(model_path)
        self.processor = DataModule()  # For tokenization and data processing
        self.softmax = torch.nn.Softmax(dim=1)
        self.labels = labels if labels else ["unacceptable", "acceptable"]

    def load_model(self, model_path: str) -> ColaModel:
        """
        Load the model from a given checkpoint.

        Args:
            model_path (str): Path to the saved model checkpoint.

        Returns:
            ColaModel: Loaded ColaModel in evaluation mode with parameters frozen.
        """
        try:
            model = ColaModel.load_from_checkpoint(model_path)
            model.eval()
            model.to(self.device)  # Move model to the appropriate device
            model.freeze()
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess the input text by tokenizing and converting to tensors.

        Args:
            text (str): The input text to classify.

        Returns:
            Dict[str, torch.Tensor]: Tokenized input IDs and attention mask tensors.
        """
        inference_sample = {"sentence": text}
        processed_input = self.processor.tokenize_data(inference_sample)
        input_ids = torch.tensor([processed_input["input_ids"]], device=self.device)
        attention_mask = torch.tensor([processed_input["attention_mask"]], device=self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


    @timing
    def predict(self, text: str) -> List[Dict[str, float]]:
        """
        Tokenize the input text and run it through the model to get predictions.

        Args:
            text (str): The input text to classify.

        Returns:
            List[Dict[str, float]]: A list of label-score dictionaries for each classification label.
        """
        inputs = self.preprocess(text)

        # Run the inputs through the model
        with torch.no_grad():  # Disable gradient calculation during inference
            logits_output = self.model(**inputs)

        # Extract logits from the model output
        logits = logits_output.logits

        # Compute softmax probabilities
        scores = self.softmax(logits).tolist()[0]
        scores = [float(score) for score in scores]  # Convert to native Python floats


        # Return the predictions
        predictions = [{"label": label, "score": score} for label, score in zip(self.labels, scores)]
        return predictions


def get_device() -> torch.device:
    """
    Determine the best device available (MPS, CUDA, or CPU).

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    model_path = "./models/best-checkpoint.ckpt"

    # Get the appropriate device
    device = get_device()

    # Initialize the predictor
    predictor = ColaPredictor(model_path, device)

    # Make a prediction
    predictions = predictor.predict(sentence)

    print(predictions)