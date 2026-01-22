from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device).long()

                outputs = self.model(inputs)

                print(f"=== 标签调试信息 ===")
                print(f"标签形状: {labels.shape}")
                print(f"标签原始值: {labels.tolist()}")
                print(f"标签范围: [{labels.min().item()}, {labels.max().item()}]")
                print(f"唯一标签值: {torch.unique(labels).tolist()}")
                # 定位异常标签的位置
                abnormal_indices = torch.where((labels < 0) | (labels > 1))[0]
                if len(abnormal_indices) > 0:
                    print(f"异常标签位置: {abnormal_indices.tolist()}")
                    print(f"异常标签值: {labels[abnormal_indices].tolist()}")
                print("====================\n")

                # Guard: remap labels to valid range [0, num_classes-1]
                num_classes = outputs.shape[1] if outputs.dim() > 1 else 1
                uniq = torch.unique(labels)
                if uniq.numel() > 0:
                    # Build contiguous mapping
                    label_map = {int(l): i for i, l in enumerate(sorted(uniq.tolist()))}
                    remapped = torch.tensor([label_map[int(l)] for l in labels.tolist()], device=self.device)
                    if remapped.max().item() >= num_classes or len(label_map) > num_classes:
                        from ..ml_utils import console
                        console.warn(
                            f"Label values {uniq.tolist()} exceed model classes ({num_classes}); remapping and clipping."
                        )
                        remapped = torch.clamp(remapped, max=num_classes - 1)
                    labels = remapped

                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                predicted = outputs.argmax(dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)

        self.latest_metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "average_loss": avg_loss,
            "precision": precision_score(all_labels, all_preds, average=average, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=average, zero_division=0),
            "f1_score": f1_score(all_labels, all_preds, average=average, zero_division=0),
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


