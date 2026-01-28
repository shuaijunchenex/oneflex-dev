from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import torch
import torch.nn as nn

from ..ml_utils import console

class ModelEvaluator:
    """
    A stateful evaluator for PyTorch models.
    Initialized with model, validation dataloader, and device.
    """

    def __init__(self, model, val_loader, criterion = None, device = "cpu"):
        """
        :param model: PyTorch model to evaluate
        :param val_loader: DataLoader with validation or test data
        :param criterion: Loss function (e.g., CrossEntropyLoss). If None, uses CrossEntropyLoss
        :param device: Computation device ('cpu' or 'cuda')
        """

        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        
        # Default to CrossEntropyLoss if no criterion provided
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.latest_metrics = {}

    def change_model(self, model, weight=None):
        if weight is not None:
            model.load_state_dict(weight, strict=True)
        self.model = model.to(self.device)

    def update_model(self, weight):
        self.model.load_state_dict(weight, strict=True)

    def evaluate(self, average="macro"):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, total_samples = 0.0, 0

        with torch.inference_mode():
            for inputs, labels in getattr(self.val_loader, "test_data_loader", self.val_loader):
                # Move inputs/labels to device; handle HF BatchEncoding/dict
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.device, non_blocking=True)
                elif isinstance(inputs, dict):
                    inputs = {k: (v.to(self.device, non_blocking=True) if hasattr(v, "to") else v)
                              for k, v in inputs.items()}
                labels = labels.to(self.device).long()

                outputs = self.model(inputs)

                # Guard: only clamp out-of-range labels; avoid per-batch remapping that skews metrics
                num_classes = outputs.shape[1] if outputs.dim() > 1 else 1
                if labels.numel() > 0:
                    min_label = labels.min().item()
                    max_label = labels.max().item()
                    if min_label < 0 or max_label >= num_classes:
                        console.warn(
                            f"Label values out of range [{min_label}, {max_label}] for num_classes={num_classes}; clamping to [0, {num_classes - 1}]."
                        )
                        labels = torch.clamp(labels, min=0, max=max(num_classes - 1, 0))

                loss = self.criterion(outputs, labels)

                # Determine batch size robustly (tensors, dict/BatchEncoding, lists)
                if hasattr(inputs, "size"):
                    batch_size = int(inputs.size(0))
                elif isinstance(inputs, dict):
                    first = next((v for v in inputs.values() if torch.is_tensor(v)), None)
                    batch_size = int(first.size(0)) if first is not None else len(inputs)
                else:
                    try:
                        batch_size = len(inputs)  # type: ignore[arg-type]
                    except Exception:
                        batch_size = int(labels.size(0))

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                predicted = outputs.argmax(dim=1)
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(total_samples, 1)

        self.latest_metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "average_loss": avg_loss,
            "precision": precision_score(all_labels, all_preds, average=average, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=average, zero_division=0),
            "f1_score": f1_score(all_labels, all_preds, average=average, zero_division=0),
            "mcc": matthews_corrcoef(all_labels, all_preds),
            "total_test_samples": total_samples,
        }
        return self.latest_metrics
        
    def print_results(self):
        """
        Pretty-print the latest evaluation metrics.
        Should be called after evaluate().
        """

        if not self.latest_metrics:
            console.error("No evaluation metrics available. run .evaluate() first.")
            return

        console.info("Evaluation Summary:")
        console.info(f"  - Loss     : {self.latest_metrics['average_loss']:.4f}")
        console.info(f"  - Accuracy : {self.latest_metrics['accuracy'] * 100:.2f}%")
        console.info(f"  - Precision: {self.latest_metrics['precision']:.4f}")
        console.info(f"  - Recall   : {self.latest_metrics['recall']:.4f}")
        console.info(f"  - F1-Score : {self.latest_metrics['f1_score']:.4f}")
        console.info(f"  - MCC      : {self.latest_metrics.get('mcc', 0.0):.4f}")
        console.info(f"  - Samples  : {self.latest_metrics['total_test_samples']}")
        return

    def get_accuracy(self):
        """
        Quick access to accuracy metric.
        :return: accuracy value or None if not evaluated yet
        """
        return self.latest_metrics.get('accuracy', None)


    def get_loss(self):
        return self.latest_metrics.get('average_loss', None)


