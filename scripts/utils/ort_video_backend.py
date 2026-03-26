"""ONNX Runtime backend for VISTRIKE batch video inference.

Used by scripts/utils/batch_video_analyzer.py when backend='onnx'. Handles ORT
session creation (CUDA-first providers, ORT_ENABLE_ALL), preprocessing,
inference, and output parsing for unified and action models.

Lazy-imports onnxruntime (fails only when create_session is called).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .onnx_model_metadata import load_action_metadata, load_unified_metadata

# ImageNet normalization (must match training pipeline)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

def get_ort_providers(device: str = 'auto') -> List[str]:
    """Return ORT execution providers list, CUDA first when applicable."""
    d = device.lower().strip()
    if d in ('auto', 'cuda'):
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def create_session(
    onnx_path: Path,
    device: str = 'auto',
) -> 'ort.InferenceSession':
    """Create an ORT InferenceSession with optimised settings."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = get_ort_providers(device)
    session = ort.InferenceSession(str(onnx_path), opts, providers=providers)

    active = session.get_providers()
    print(f"  ORT session: {onnx_path.name}")
    print(f"    Requested providers: {providers}")
    print(f"    Active providers:    {active}")
    return session


# ---------------------------------------------------------------------------
# Unified model: preprocessing
# ---------------------------------------------------------------------------

def preprocess_unified(
    frame_bgr: np.ndarray,
    input_height: int,
    input_width: int,
) -> Tuple[np.ndarray, float, float]:
    """Prepare a BGR frame for the unified ONNX model.

    Returns:
        input_tensor: float32 [1, 3, H, W] in 0-1 range, RGB, NO ImageNet norm
        scale_x: original_width / input_width
        scale_y: original_height / input_height
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    scale_x = orig_w / input_width
    scale_y = orig_h / input_height

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

    tensor = resized.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
    tensor = np.expand_dims(tensor, 0)  # add batch dim
    return tensor, scale_x, scale_y


# ---------------------------------------------------------------------------
# Unified model: run + parse
# ---------------------------------------------------------------------------

def run_unified(
    session: 'ort.InferenceSession',
    input_tensor: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run unified ONNX model and return raw output dict keyed by name."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: input_tensor})
    return dict(zip(output_names, outputs))


def parse_unified_outputs(
    raw: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    scale_x: float,
    scale_y: float,
    confidence: float,
    attr_confidence: float = 0.0,
) -> List[Dict]:
    """Parse unified ONNX outputs into person dicts matching PyTorch format.

    Boxes are rescaled to original frame coordinates.
    """
    num_det = int(raw['num_detections'][0])
    boxes = raw['boxes'][0][:num_det]       # [N, 4]
    scores = raw['scores'][0][:num_det]     # [N]
    labels = raw['labels'][0][:num_det]     # [N]

    attribute_names = meta.get('attribute_names', [])
    attribute_classes = meta.get('attribute_classes', {})

    persons = []
    for i in range(num_det):
        score = float(scores[i])
        if score < confidence:
            continue

        x1, y1, x2, y2 = boxes[i]
        person: Dict[str, Any] = {
            'id': i,
            'box': [
                float(x1 * scale_x),
                float(y1 * scale_y),
                float(x2 * scale_x),
                float(y2 * scale_y),
            ],
            'confidence': score,
            'label': int(labels[i]),
            'embedding': None,
        }

        for attr_name in attribute_names:
            prob_key = f'attr_{attr_name}_prob'
            if prob_key not in raw:
                person[attr_name] = 'uncertain'
                person[f'{attr_name}_confidence'] = 0.0
                continue

            probs = raw[prob_key][0][i]  # [C]
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])
            person[f'{attr_name}_confidence'] = pred_conf

            classes = attribute_classes.get(attr_name, [])
            if pred_conf < attr_confidence:
                person[attr_name] = 'uncertain'
            elif pred_idx < len(classes):
                person[attr_name] = classes[pred_idx]
            else:
                person[attr_name] = f'class_{pred_idx}'

        for attr_name in ['corner', 'guard', 'stance', 'lead_hand', 'visibility']:
            if attr_name not in person:
                person[attr_name] = 'unknown'

        persons.append(person)

    return persons


# ---------------------------------------------------------------------------
# Action model: preprocessing
# ---------------------------------------------------------------------------

def preprocess_action_clip(
    frames_rgb: List[np.ndarray],
    window_size: int,
    img_size: int,
) -> np.ndarray:
    """Build an action clip tensor from RGB frames.

    Matches the PyTorch _detect_actions pipeline: evenly sample window_size
    frames, resize to img_size x img_size, ImageNet normalize, stack to
    [1, 3, T, H, W].
    """
    if len(frames_rgb) < window_size:
        return None

    indices = np.linspace(0, len(frames_rgb) - 1, window_size, dtype=int)
    sampled = [frames_rgb[i] for i in indices]

    tensors = []
    for frame in sampled:
        img = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        tensors.append(img)

    clip = np.stack(tensors, axis=1)  # [3, T, H, W]
    return np.expand_dims(clip, 0).astype(np.float32)  # [1, 3, T, H, W]


# ---------------------------------------------------------------------------
# Action model: run + parse
# ---------------------------------------------------------------------------

def run_action(
    session: 'ort.InferenceSession',
    clip_tensor: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run action ONNX model and return raw output dict keyed by target name."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: clip_tensor})
    return dict(zip(output_names, outputs))


def parse_action_outputs(
    raw: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    action_confidence: float,
) -> Dict[str, Any]:
    """Parse action ONNX outputs into action result dict matching PyTorch format."""
    targets = meta.get('targets', [])
    label_maps = meta.get('label_maps', {})

    result: Dict[str, Any] = {'detected': False}

    for target in targets:
        if target not in raw:
            continue
        logits = raw[target][0]  # [num_classes]
        probs = _softmax(logits)
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        target_map = label_maps.get(target, {})
        if isinstance(target_map, dict):
            pred_label = target_map.get(str(pred_idx), str(pred_idx))
        elif isinstance(target_map, list) and pred_idx < len(target_map):
            pred_label = target_map[pred_idx]
        else:
            pred_label = str(pred_idx)

        result[target] = pred_label
        result[f'{target}_confidence'] = conf

        if conf > action_confidence:
            result['detected'] = True

    return result


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
