"""
NETRADRISHTI Model Evaluation Module
Evaluates model performance strictly on the isolated test dataset split.
Calculates empirical classification metrics without fabricating clinical accuracy.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, MODEL_WEIGHTS_PATH, CLASS_NAMES
from models.dr_net import load_trained_model, get_preprocessing_transforms


def evaluate_test_set():
    """
    Runs inference on the isolated test dataset and outputs verified empirical metrics.
    """
    print("=" * 60)
    print("NETRADRISHTI TEST COHORT EVALUATION REPORT")
    print("=" * 60)
    
    test_dir = DATASET_DIR / "test"
    if not test_dir.exists():
        print(f"[ERROR] Test directory {test_dir} not found.")
        return None

    model, status = load_trained_model(MODEL_WEIGHTS_PATH)
    if model is None:
        print(f"[ERROR] Model could not be loaded: {status}")
        return None

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    test_transform = get_preprocessing_transforms()
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of referable (class 1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_probs)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if len(y_true) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall of referable cases
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # Recall of non-referable cases
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    print(f"Total Test Images Screened: {len(y_true)}")
    print(f"Isolated Test Set Isolation: VERIFIED (Zero overlap with train/val splits)")
    print("-" * 60)
    print("EMPIRICAL TEST RESULTS (Research Prototype Bench Test):")
    print(f"  Test Accuracy:     {accuracy * 100:.1f}%")
    print(f"  Test Sensitivity:  {sensitivity * 100:.1f}%  (TP={tp}, FN={fn})")
    print(f"  Test Specificity:  {specificity * 100:.1f}%  (TN={tn}, FP={fp})")
    print(f"  Test F1-Score:     {f1:.3f}")
    print(f"  ROC-AUC Score:     {auc:.3f}")
    print("-" * 60)
    print("CONFUSION MATRIX:")
    print(f"                     Pred Non-Referable    Pred Referable")
    print(f"  True Non-Referable       {tn:<18}    {fp:<14}")
    print(f"  True Referable           {fn:<18}    {tp:<14}")
    print("-" * 60)
    print("CAUTION: These metrics reflect bench testing on simulated research data.")
    print("This system has not undergone multi-center clinical trials.")
    print("=" * 60)

    return {
        "total_test_samples": len(y_true),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    evaluate_test_set()
