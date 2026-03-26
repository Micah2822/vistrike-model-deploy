"""Load ONNX model metadata for production inference.

Used by scripts/utils/ort_video_backend.py / batch ONNX path in
scripts/utils/batch_video_analyzer.py.

Production: reads `.meta.json` sidecars emitted by scripts/12_export_onnx.py.
Dev/legacy fallback: metadata from `.pt` checkpoints via torch.load.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_meta_json(meta_path: Path) -> Dict[str, Any]:
    """Load a .meta.json sidecar file."""
    with open(meta_path, 'r') as f:
        return json.load(f)


def load_unified_metadata(
    models_dir: Path,
    allow_torch_fallback: bool = False,
) -> Dict[str, Any]:
    """Load unified model metadata.

    Args:
        models_dir: Root models directory (e.g. models/).
        allow_torch_fallback: If True and .meta.json is missing, attempt to
            extract metadata from the .pt checkpoint. Should only be used in
            dev/testing; production must ship .meta.json files.

    Returns:
        Dict with keys: input_height, input_width, output_names,
        attribute_names, attribute_classes, num_classes, fp16, etc.

    Raises:
        FileNotFoundError: When metadata is unavailable.
    """
    meta_path = models_dir / 'unified' / 'unified_model.meta.json'
    if meta_path.exists():
        return load_meta_json(meta_path)

    if allow_torch_fallback:
        return _unified_meta_from_checkpoint(models_dir)

    raise FileNotFoundError(
        f"Unified model metadata not found: {meta_path}. "
        "Run 12_export_onnx.py to generate .meta.json sidecars."
    )


def load_action_metadata(
    models_dir: Path,
    action_type: str,
    allow_torch_fallback: bool = False,
) -> Dict[str, Any]:
    """Load action model metadata.

    Args:
        models_dir: Root models directory.
        action_type: E.g. 'punch', 'defense'.
        allow_torch_fallback: See load_unified_metadata.

    Returns:
        Dict with keys: targets, label_maps, window_size, img_size,
        output_names, etc.
    """
    meta_path = (
        models_dir / 'actions' / action_type / f'{action_type}_model.meta.json'
    )
    if meta_path.exists():
        return load_meta_json(meta_path)

    if allow_torch_fallback:
        return _action_meta_from_checkpoint(models_dir, action_type)

    raise FileNotFoundError(
        f"Action model metadata not found: {meta_path}. "
        "Run 12_export_onnx.py to generate .meta.json sidecars."
    )


# ---------------------------------------------------------------------------
# Dev/legacy fallback: extract metadata from PyTorch checkpoints
# ---------------------------------------------------------------------------

def _unified_meta_from_checkpoint(models_dir: Path) -> Dict[str, Any]:
    """Extract unified model metadata from best.pt (dev fallback)."""
    import torch  # lazy

    ckpt_path = models_dir / 'unified' / 'best.pt'
    if not ckpt_path.exists():
        ckpt_path = models_dir / 'unified' / 'last.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No unified checkpoint found in {models_dir / 'unified'}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    attr_config = config.get('attribute_config', {})

    attribute_names = []
    attribute_classes = {}
    if isinstance(attr_config, dict):
        attribute_names = sorted(attr_config.keys())
        for name, classes in attr_config.items():
            if isinstance(classes, dict):
                attribute_classes[name] = classes.get('classes', [])
            else:
                attribute_classes[name] = list(classes)

    return {
        'model_type': 'unified',
        'backbone_name': config.get('backbone_name', 'efficientnet_b0'),
        'input_height': 384,
        'input_width': 640,
        'input_format': 'float32 0-1 RGB NCHW (no ImageNet normalization)',
        'output_names': None,  # not available from checkpoint
        'attribute_names': attribute_names,
        'attribute_classes': attribute_classes,
        'num_classes': config.get('num_classes', 2),
        'fp16': False,
        '_source': 'torch_fallback',
    }


def _action_meta_from_checkpoint(
    models_dir: Path,
    action_type: str,
) -> Dict[str, Any]:
    """Extract action model metadata from best.pt (dev fallback)."""
    import torch  # lazy

    ckpt_path = models_dir / 'actions' / action_type / 'best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    targets = checkpoint.get('targets', [])
    label_maps_raw = checkpoint.get('label_maps', {})
    reversed_label_maps = {}
    for target, mapping in label_maps_raw.items():
        if isinstance(mapping, dict):
            reversed_label_maps[target] = {str(v): k for k, v in mapping.items()}
        elif isinstance(mapping, list):
            reversed_label_maps[target] = {str(i): name for i, name in enumerate(mapping)}
        else:
            reversed_label_maps[target] = mapping

    return {
        'model_type': 'action',
        'action_type': action_type,
        'architecture': checkpoint.get('architecture', 'r2plus1d'),
        'targets': targets,
        'label_maps': reversed_label_maps,
        'num_classes_per_target': checkpoint.get('num_classes_per_target', {}),
        'window_size': checkpoint.get('window_size', 16),
        'img_size': int(checkpoint.get('img_size', 112)),
        'input_format': 'float32 ImageNet-normalized [B,3,T,H,W]',
        'output_names': targets,
        '_source': 'torch_fallback',
    }
