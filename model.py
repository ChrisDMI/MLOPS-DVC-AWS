import torch
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import torchmetrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.classification.F1Score(num_classes=self.num_classes, task="binary")
        self.precision_macro_metric = torchmetrics.classification.Precision(
            average="macro", num_classes=self.num_classes, task="binary"
        )
        self.recall_macro_metric = torchmetrics.classification.Recall(
            average="macro", num_classes=self.num_classes, task="binary"
        )
        self.precision_micro_metric = torchmetrics.classification.Precision(average="micro", task="binary")
        self.recall_micro_metric = torchmetrics.classification.Recall(average="micro", task="binary")

        # Storage for validation outputs
        self.validation_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/loss_epoch", outputs.loss, prog_bar=True, on_epoch=True)  # Add this line
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)

        # Store outputs for the epoch end
        self.validation_outputs.append({"labels": labels, "logits": outputs.logits})

        return {"labels": labels, "logits": outputs.logits}

    def on_validation_epoch_end(self):
        # Gather labels and logits for confusion matrix
        labels = torch.cat([x["labels"] for x in self.validation_outputs]).cpu()  # Move to CPU
        logits = torch.cat([x["logits"] for x in self.validation_outputs]).cpu()  # Move to CPU
        preds = torch.argmax(logits, 1)

        # Clear validation outputs for the next epoch
        self.validation_outputs = []

        # Log confusion matrix using W&B's built-in method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    preds=preds.numpy(), y_true=labels.numpy()
                )
            }
        )

        # Optionally, you can use seaborn or sklearn for more customization in logging confusion matrices.
        # Uncomment the code below to use custom visualization.
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])