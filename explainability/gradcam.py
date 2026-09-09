"""
NETRADRISHTI Explainability Module
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
Extracts spatial feature attribution maps from the final convolutional layer.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

from models.dr_net import NetraNet, get_preprocessing_transforms


class GradCAM:
    """
    Computes Grad-CAM for a given PyTorch CNN and target convolutional layer.
    """
    def __init__(self, model: NetraNet, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.get_target_layer_for_cam()
        
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] contains the gradient of the loss with respect to module output
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates normalized Grad-CAM heatmap [0.0, 1.0] for the specified class.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Ensure tensor tracks gradients for backprop
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[0, class_idx]
        score.backward()

        # Channel-wise global average pooling of gradients
        # gradients shape: [1, C, H, W]
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by pooled gradients
        activations = self.activations[0] # [C, H, W]
        for i in range(pooled_gradients.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        # ReLU on weighted combination
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normalize between 0 and 1
        heatmap_np = heatmap.cpu().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 1e-8:
            heatmap_np = heatmap_np / max_val
        else:
            heatmap_np = np.zeros_like(heatmap_np)

        return heatmap_np


def generate_cam_overlay(
    model: NetraNet,
    image: Image.Image,
    class_idx: Optional[int] = None,
    alpha: float = 0.45,
    colormap_name: str = 'jet'
) -> Tuple[Image.Image, Image.Image, Image.Image, str]:
    """
    End-to-end explainability pipeline:
    Returns:
      1. Original Fundus Image (resized)
      2. Standalone Heatmap Image
      3. Blended Overlay Image (Fundus + Grad-CAM)
      4. Evidence Explanation string
    """
    target_size = (224, 224)
    orig_resized = image.convert('RGB').resize(target_size)
    orig_np = np.array(orig_resized, dtype=np.float32) / 255.0

    tf = get_preprocessing_transforms()
    input_tensor = tf(image).unsqueeze(0)

    grad_cam = GradCAM(model)
    try:
        raw_heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=class_idx)
    finally:
        grad_cam.remove_hooks()

    # Resize heatmap to 224x224 using PIL
    heatmap_pil = Image.fromarray((raw_heatmap * 255).astype(np.uint8)).resize(target_size, resample=Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap_name)
    colored_cam = cmap(heatmap_resized)[:, :, :3] # discard alpha channel of colormap

    # Create standalone heatmap image
    heatmap_image = Image.fromarray((colored_cam * 255).astype(np.uint8))

    # Blend overlay with original fundus
    overlay_np = (alpha * colored_cam) + ((1.0 - alpha) * orig_np)
    overlay_np = np.clip(overlay_np, 0.0, 1.0)
    overlay_image = Image.fromarray((overlay_np * 255).astype(np.uint8))

    evidence_text = (
        "Highlighted regions indicate spatial retinal structures that contributed most "
        "strongly to the neural network's prototype risk classification. "
        "Grad-CAM demonstrates model feature localization and must be cross-examined by an ophthalmologist."
    )

    return orig_resized, heatmap_image, overlay_image, evidence_text
