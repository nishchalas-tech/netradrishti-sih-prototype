"""
NETRADRISHTI Deep Learning Model Architecture (NetraNet)
Convolutional Neural Network for Diabetic Retinopathy Risk Stratification
Extensible for Binary (Referable / Non-Referable) and 5-class ICDR grading.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from config import MODEL_WEIGHTS_PATH, CLASS_NAMES, NUM_CLASSES


class NetraNet(nn.Module):
    """
    Deep Convolutional Architecture with accessible intermediate feature maps
    optimized for Grad-CAM Explainable AI (XAI).
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(NetraNet, self).__init__()
        
        # Feature Extraction Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Target layer for Grad-CAM explainability
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_target_layer_for_cam(self) -> nn.Module:
        """Returns the final convolutional block for Grad-CAM feature attribution."""
        return self.conv4[0]


def get_preprocessing_transforms():
    """Standard clinical RGB normalization for fundus images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_trained_model(weights_path: Optional[Path] = None, device: str = 'cpu') -> Tuple[Optional[NetraNet], str]:
    """
    Loads real trained weights from disk.
    If weights file is missing, returns (None, 'MODEL NOT LOADED').
    """
    path = weights_path or MODEL_WEIGHTS_PATH
    if not Path(path).is_file():
        return None, "MODEL NOT LOADED"

    try:
        model = NetraNet(num_classes=NUM_CLASSES)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # Handle state_dict or full checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint

        model.to(device)
        model.eval()
        return model, "ONLINE"
    except Exception as e:
        return None, f"ERROR LOADING MODEL: {str(e)}"


def get_model_status() -> Tuple[bool, str]:
    """Checks whether the real model weights exist and are loadable."""
    if not MODEL_WEIGHTS_PATH.is_file():
        return False, "MODEL NOT LOADED"
    model, msg = load_trained_model()
    return (model is not None), msg


def run_inference(model: NetraNet, image: Image.Image, device: str = 'cpu'):
    """
    Executes actual forward pass on model without hard-coding predictions.
    Returns:
      prediction: "REFERABLE" or "NON-REFERABLE"
      confidence: float in [0.0, 1.0] (real softmax probability)
      probabilities: dict of class name to probability
      input_tensor: normalized tensor suitable for Grad-CAM
    """
    tf = get_preprocessing_transforms()
    tensor = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "class_index": pred_idx,
        "input_tensor": tensor
    }
