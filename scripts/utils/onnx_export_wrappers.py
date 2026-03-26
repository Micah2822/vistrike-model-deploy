"""
ONNX-friendly wrappers for VISTRIKE trained models.

Thin nn.Module wrappers that sit on top of already-loaded real models so
torch.onnx.export sees a single tensor in and a flat tuple of tensors out.

Import-only; not a runnable script.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ActionONNXWrapper(nn.Module):
    """Wraps a loaded MultiHeadActionModel for ONNX export.

    Converts the dict-based forward output to a flat tuple of logit tensors
    in ``model.targets`` order, suitable for ``torch.onnx.export``.

    Input:  x ``[B, 3, T, H, W]``  — video clip, float32, ImageNet-normalized
    Output: tuple of logit tensors ``[B, num_classes_i]``, one per target
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.targets: List[str] = list(model.targets)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.model(x)
        return tuple(out[t] for t in self.targets)


class UnifiedONNXWrapper(nn.Module):
    """Wraps a loaded UnifiedBoxingModel (Faster R-CNN) for ONNX export.

    Converts variable-length detection + attribute outputs to fixed-shape
    tensors padded/truncated to *max_detections* (default 100, matching
    ``box_detections_per_img`` in training).

    Input:  x ``[1, 3, H, W]``  — single image, float32 0-1, RGB NCHW
            (the model's internal ``GeneralizedRCNNTransform`` handles
            ImageNet normalization and resizing)

    Output: tuple of fixed-shape tensors (see :attr:`output_names`):

        ============================  =================  ===========
        Name                          Shape              Dtype
        ============================  =================  ===========
        ``num_detections``            ``[1]``            int64
        ``boxes``                     ``[1, K, 4]``      float32
        ``scores``                    ``[1, K]``          float32
        ``labels``                    ``[1, K]``          int64
        ``box_mask``                  ``[1, K]``          bool
        ``attr_<name>_prob``          ``[1, K, C]``       float32
        ============================  =================  ===========

    .. note::

       Padding uses operations whose shapes are captured at trace time.
       The ONNX graph is faithful for the detection count observed during
       export.  Deploy with the **same** input H x W used at export time.
       For fully dynamic detection counts, use ``torch.onnx.dynamo_export``
       (PyTorch 2.1+) or handle variable outputs in post-processing.
    """

    def __init__(
        self,
        model: nn.Module,
        max_detections: int = 100,
        attribute_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.max_det = max_detections

        if attribute_names is None:
            if hasattr(model, 'attribute_config') and model.attribute_config:
                attribute_names = sorted(model.attribute_config.keys())
            else:
                attribute_names = []
        self.attribute_names: List[str] = attribute_names

        self._attr_num_classes = {}
        for name in self.attribute_names:
            self._attr_num_classes[name] = model.num_attribute_classes[name]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Single-image export (B=1); model expects a list of tensors.
        outputs = self.model([x[0]])
        det = outputs[0]

        boxes = det['boxes']       # [N, 4]
        scores = det['scores']     # [N]
        labels = det['labels']     # [N]

        K = self.max_det
        device = x.device
        n = boxes.shape[0]
        k = min(n, K)

        num_det = torch.tensor([k], dtype=torch.int64, device=device)

        padded_boxes = torch.zeros(1, K, 4, dtype=x.dtype, device=device)
        padded_scores = torch.zeros(1, K, dtype=x.dtype, device=device)
        padded_labels = torch.zeros(1, K, dtype=torch.int64, device=device)
        mask = torch.zeros(1, K, dtype=torch.bool, device=device)

        padded_boxes[0, :k] = boxes[:k]
        padded_scores[0, :k] = scores[:k]
        padded_labels[0, :k] = labels[:k]
        mask[0, :k] = True

        result: List[torch.Tensor] = [
            num_det, padded_boxes, padded_scores, padded_labels, mask,
        ]

        attrs = det.get('attributes', {})
        for attr_name in self.attribute_names:
            C = self._attr_num_classes[attr_name]
            padded = torch.zeros(1, K, C, dtype=x.dtype, device=device)
            if attr_name in attrs and k > 0:
                probs = attrs[attr_name]['probabilities']
                # Guard against 1-D empty tensors from 0-detection fallback
                if probs.dim() == 2 and probs.shape[0] > 0:
                    padded[0, :k] = probs[:k]
            result.append(padded)

        return tuple(result)

    @property
    def output_names(self) -> List[str]:
        """ONNX output tensor names in ``forward()`` return order."""
        names = ['num_detections', 'boxes', 'scores', 'labels', 'box_mask']
        for attr_name in self.attribute_names:
            names.append(f'attr_{attr_name}_prob')
        return names
