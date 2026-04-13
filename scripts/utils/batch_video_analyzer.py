"""Shared batch video analysis pipeline.

Used by:
- scripts/10_inference.py — PyTorch checkpoints (backend='pytorch')
- scripts/inference_onnx.py — ONNX Runtime + .meta.json (backend='onnx')

Tracking, event counting, side mapping, gap grouping, and summary logic are
identical for both backends; only model forward passes differ (PyTorch vs ORT).

Entry-point scripts call run_main(args, backend=...). See also
scripts/utils/ort_video_backend.py and scripts/utils/onnx_model_metadata.py.

When analyze_video() is called with progress_callback, callback kwargs include
score_range, detection_threshold, and boxes_above_threshold for live status UIs.
"""


# SSL certificate workaround for macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Core dependencies
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch torchvision")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import torchvision.transforms as T
except ImportError:
    print("ERROR: torchvision is required. Install with: pip install torchvision")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("WARNING: tqdm not found. Progress bars disabled.")
    tqdm = lambda x, **kwargs: x

import numpy as np

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Resolved directory constants — this file lives in scripts/utils/, so:
#   SCRIPTS_DIR  = scripts/
#   REPO_ROOT    = project root (parent of scripts/)
_THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = _THIS_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

# Action types config path (repo-root configs/)
ACTION_TYPES_CONFIG_PATH = REPO_ROOT / "configs" / "action_types.yaml"


def load_action_types_config() -> tuple:
    """
    Load action types config from YAML.
    
    Returns:
        Tuple of (ACTION_TYPES list, COLORS dict, fighter_keys dict, fighter_fields dict,
                  confidence_keys dict, key_attributes_map dict)
        
        fighter_keys: maps action_type -> field the model predicts (e.g. 'attacker_side')
        fighter_fields: maps action_type -> original annotation field (e.g. 'attacker')
        confidence_keys: maps action_type -> list of confidence key names used for event
            detection. Built from confidence_attributes (when set in YAML) or key_attributes
            (fallback), each becomes attr_confidence; plus fighter_field_confidence for
            backward compat with old checkpoints.
        key_attributes_map: maps action_type -> list of key attribute names from YAML.
            Used to build event display text and summary breakdowns dynamically.
    """
    # Fallback only when the config file is missing. When it exists, everything below
    # is built from config.get('action_types', {}).items() — so adding a new tag in
    # the YAML requires no code changes; the new type is picked up automatically.
    if not ACTION_TYPES_CONFIG_PATH.exists():
        print(f"WARNING: Action types config not found: {ACTION_TYPES_CONFIG_PATH}, using defaults")
        return (
            ['punch', 'defense', 'footwork', 'clinch', 'knockdown', 'wobble', 'stoppage'],
            {
                'red': (0, 0, 255), 'blue': (255, 0, 0), 'unknown': (128, 128, 128),
                'punch': (0, 255, 255), 'defense': (0, 255, 0), 'footwork': (255, 0, 255),
                'clinch': (0, 165, 255), 'knockdown': (0, 0, 255), 'wobble': (0, 128, 255),
                'stoppage': (128, 0, 128),
            },
            {'punch': 'attacker_side', 'defense': 'defender_side', 'footwork': 'fighter_side', 
             'clinch': 'initiator_side', 'knockdown': 'fighter_side', 'wobble': 'fighter_side', 'stoppage': None},
            {'punch': 'attacker', 'defense': 'defender', 'footwork': 'fighter',
             'clinch': 'initiator', 'knockdown': 'fighter', 'wobble': 'fighter', 'stoppage': None},
            # Confidence keys: match the previously hardcoded maps. Include both _side (new)
            # and non-_side (old checkpoint) fighter confidence for backward compat.
            {
                'punch': ['type_confidence', 'attacker_side_confidence', 'attacker_confidence', 'result_confidence'],
                'defense': ['type_confidence', 'defender_side_confidence', 'defender_confidence', 'success_confidence'],
                'footwork': ['type_confidence', 'fighter_side_confidence', 'fighter_confidence'],
                'clinch': ['state_confidence', 'initiator_side_confidence', 'initiator_confidence'],
                'knockdown': ['type_confidence', 'fighter_side_confidence', 'fighter_confidence'],
                'wobble': ['type_confidence', 'severity_confidence', 'fighter_side_confidence', 'fighter_confidence'],
                'stoppage': ['type_confidence'],
            },
            # Key attributes: for event text and summary breakdowns
            {
                'punch': ['attacker_side', 'hand', 'type', 'result'],
                'defense': ['defender_side', 'type', 'success'],
                'footwork': ['fighter_side', 'type'],
                'clinch': ['initiator_side', 'state'],
                'knockdown': ['fighter_side', 'type'],
                'wobble': ['fighter_side', 'severity'],
                'stoppage': ['type', 'reason'],
            },
        )
    
    with open(ACTION_TYPES_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    action_types = list(config.get('action_types', {}).keys())
    
    # Build COLORS dict
    colors = {}
    # Add visualization colors (red, blue, unknown)
    for name, bgr in config.get('visualization_colors', {}).items():
        colors[name] = tuple(bgr)
    # Add action type colors
    for name, cfg in config.get('action_types', {}).items():
        if 'color_bgr' in cfg:
            colors[name] = tuple(cfg['color_bgr'])
    
    # Build fighter_keys dict (model prediction field, e.g. 'attacker_side')
    fighter_keys = {}
    for name, cfg in config.get('action_types', {}).items():
        fighter_keys[name] = cfg.get('fighter_key')  # None for stoppage etc.
    
    # Build fighter_fields dict (original annotation field, e.g. 'attacker')
    fighter_fields = {}
    for name, cfg in config.get('action_types', {}).items():
        fighter_fields[name] = cfg.get('fighter_field')  # None for stoppage etc.
    
    # Build confidence_keys dict for event detection.
    # Uses confidence_attributes when present (controls which attribute confidences
    # drive event detection); falls back to key_attributes when absent.
    # If fighter_field is set, fighter_field_confidence is appended for backward compat
    # with old checkpoints that output e.g. 'attacker_confidence' instead of 'attacker_side_confidence'.
    confidence_keys = {}
    for name, cfg in config.get('action_types', {}).items():
        # Prefer confidence_attributes (explicit control over event confidence);
        # fall back to key_attributes (display/summary attributes).
        conf_attrs = cfg.get('confidence_attributes')
        if conf_attrs is None:
            conf_attrs = cfg.get('key_attributes', [])
        # Normalize to list (protect against scalar value in YAML)
        if not isinstance(conf_attrs, list):
            conf_attrs = [conf_attrs] if conf_attrs else []
        keys = [f"{attr}_confidence" for attr in conf_attrs]
        # Append legacy fighter_field confidence for backward compat with old checkpoints
        ff = cfg.get('fighter_field')  # None for stoppage etc.
        if ff:
            legacy_key = f"{ff}_confidence"
            if legacy_key not in keys:
                keys.append(legacy_key)
        # Fallback: at minimum use type_confidence
        if not keys:
            keys = ['type_confidence']
        confidence_keys[name] = keys
    
    # Build key_attributes_map for event text and summary breakdowns
    key_attributes_map = {}
    for name, cfg in config.get('action_types', {}).items():
        ka = cfg.get('key_attributes', [])
        if not isinstance(ka, list):
            ka = [ka] if ka else []
        key_attributes_map[name] = ka
    
    return action_types, colors, fighter_keys, fighter_fields, confidence_keys, key_attributes_map


# Load action types configuration from YAML
ACTION_TYPES, COLORS, FIGHTER_KEYS_MAP, FIGHTER_FIELDS_MAP, CONFIDENCE_KEYS_MAP, KEY_ATTRIBUTES_MAP = load_action_types_config()


def _format_event_text(action_type: str, action: dict) -> str:
    """Build display text for an action event using config-driven attributes.
    
    Uses FIGHTER_FIELDS_MAP and KEY_ATTRIBUTES_MAP to build a human-readable string.
    The fighter_key (_side prediction like left/right) is skipped since the mapped
    fighter_field (red/blue) is more meaningful to the user.
    """
    fighter_field = FIGHTER_FIELDS_MAP.get(action_type)
    key_attrs = KEY_ATTRIBUTES_MAP.get(action_type, [])
    if not isinstance(key_attrs, list):
        key_attrs = []
    fighter_key = FIGHTER_KEYS_MAP.get(action_type)
    
    parts = []
    # Add fighter identity (red/blue) if this type has a fighter_field
    if fighter_field:
        parts.append(str(action.get(fighter_field, '?')))
    
    # Add key attribute values, skipping the internal _side prediction (left/right)
    for attr in key_attrs:
        if attr == fighter_key:
            continue  # Skip left/right side; fighter_field already gives red/blue
        val = action.get(attr)
        if val is not None:
            parts.append(str(val))
    
    if parts:
        return f"{action_type.upper()}: {' '.join(parts)}"
    return f"{action_type.upper()}: detected"


# =============================================================================
# MODEL ARCHITECTURES FOR INFERENCE
# =============================================================================
# UnifiedBoxingModel (Faster R-CNN + attributes) is imported from the training
# script (07_train_unified.py) to guarantee architecture parity and FPN support.
# Other model classes (MPS, action) are defined locally.

from torchvision import models
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torch.nn.functional as F
from typing import Tuple

# ---------------------------------------------------------------------------
# Lazy import of UnifiedBoxingModel from 07_train_unified.py
# ---------------------------------------------------------------------------
_train_unified_module = None


def _get_train_unified_module():
    """Lazy-load the training module to access UnifiedBoxingModel."""
    global _train_unified_module
    if _train_unified_module is None:
        filepath = SCRIPTS_DIR / "07_train_unified.py"
        if not filepath.exists():
            raise ImportError(f"Cannot find training script: {filepath}")
        spec = importlib.util.spec_from_file_location("train_unified", filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _train_unified_module = mod
    return _train_unified_module


def _get_unified_boxing_model_class():
    """Return the UnifiedBoxingModel class from 07_train_unified.py."""
    return _get_train_unified_module().UnifiedBoxingModel


class AttributeHead(nn.Module):
    """Classification head for a single attribute (used by UnifiedMPSModel)."""
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = [
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        if num_layers >= 2:
            mid_dim = hidden_dim // 2
            layers.extend([
                nn.Linear(hidden_dim, mid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            layers.append(nn.Linear(mid_dim, num_classes))
        else:
            layers.append(nn.Linear(hidden_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SSDDetectionHead(nn.Module):
    """SSD-style detection head for MPS model."""
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
    
    def forward(self, features):
        cls_logits = self.cls_head(features)
        box_regression = self.reg_head(features)
        return cls_logits, box_regression


class UnifiedMPSModel(nn.Module):
    """MPS-optimized unified detection + attribute model (SSD-style)."""
    
    def __init__(
        self,
        backbone_name: str = 'mobilenet_v3_large',
        num_classes: int = 2,
        attribute_config: Dict = None,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.attribute_config = attribute_config or {}
        
        # Create backbone
        if backbone_name == 'mobilenet_v3_large':
            net = models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained_backbone else None)
            self.backbone = net.features
            backbone_channels = 960
        elif backbone_name == 'mobilenet_v3_small':
            net = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained_backbone else None)
            self.backbone = net.features
            backbone_channels = 576
        elif backbone_name == 'efficientnet_b0':
            net = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained_backbone else None)
            self.backbone = net.features
            backbone_channels = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Detection head
        self.num_anchors = 6
        self.detection_head = SSDDetectionHead(backbone_channels, self.num_anchors, num_classes)
        
        # Attribute heads
        self.roi_pool_size = 7
        attr_input_dim = backbone_channels * self.roi_pool_size * self.roi_pool_size
        
        # Handle both dict formats: {'attr': ['class1', 'class2']} or {'attr': {'classes': [...]}}
        self.attribute_heads = nn.ModuleDict()
        for name, cfg in self.attribute_config.items():
            if isinstance(cfg, dict):
                num_cls = len(cfg.get('classes', []))
            else:
                num_cls = len(cfg)
            self.attribute_heads[name] = AttributeHead(attr_input_dim, num_cls)
        
        # Store anchor info
        self.register_buffer('anchor_scales', torch.tensor([32, 64, 128, 192, 256, 320], dtype=torch.float32))
        self.register_buffer('anchor_ratios', torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32))
    
    def forward(self, images, targets=None):
        """Forward pass."""
        features = self.backbone(images)
        cls_logits, box_regression = self.detection_head(features)
        
        if self.training:
            # Training mode would compute losses
            return {'cls_logits': cls_logits, 'box_regression': box_regression}
        
        # Inference mode - return raw outputs (postprocessing done separately)
        return {
            'cls_logits': cls_logits,
            'box_regression': box_regression,
            'features': features
        }


def _head_key(name: str) -> str:
    """ModuleDict key for a target; avoid 'type' which conflicts with nn.Module.type.
    
    This must match the training script (09_train_actions.py) to load checkpoints correctly.
    """
    return "head_type" if name == "type" else name


class MultiHeadActionModel(nn.Module):
    """Multi-head action recognition model."""
    
    def __init__(
        self,
        architecture: str,
        targets: List[str],
        num_classes_per_target: Dict[str, int],
    ):
        super().__init__()
        
        self.architecture = architecture
        self.targets = targets
        self.num_classes_per_target = num_classes_per_target
        
        # Create backbone based on architecture
        if architecture == 'r2plus1d':
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            backbone = r2plus1d_18(weights=weights)
            self.feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Create classification heads using _head_key to match training script
        # This ensures state_dict keys match (e.g., "heads.head_type.0.weight" not "heads.type.0.weight")
        self.heads = nn.ModuleDict()
        for target in targets:
            num_classes = num_classes_per_target[target]
            head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feat_dim, num_classes)
            )
            self.heads[_head_key(target)] = head
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass - returns logits for each target."""
        features = self.backbone(x)
        # Use _head_key to access heads, but return with original target names
        return {target: self.heads[_head_key(target)](features) for target in self.targets}


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_device(device_str: str) -> torch.device:
    """Resolve torch.device from CLI string; fall back if the backend is unavailable.

    ``auto`` prefers CUDA, then CPU — never MPS (explicit ``--device mps`` only).
    """
    s = (device_str or 'auto').strip().lower()
    mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if s == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    if s == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("WARNING: --device cuda but CUDA is not available; using CPU.")
        return torch.device('cpu')

    if s == 'mps':
        if mps_ok:
            return torch.device('mps')
        print("WARNING: --device mps but MPS is not available; using CUDA or CPU.")
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    return torch.device(device_str)


class BoxingAnalyzer:
    """
    Main analyzer class that loads models and processes videos.
    
    Supports two modes:
    1. Unified model (default): Detection + all attributes in one pass
    2. Separate models: Individual detector + attribute classifiers
    
    Side-based fighter assignment options (see 05_VIDEO_INFERENCE.md):
    - assign_single_fighter: when only 1 fighter tracked, map action to that fighter
    - side_confidence_min: min _side confidence to keep fighter (else unknown)
    - stable_side_frames: use median center_x over last K frames for left/right mapping
    """
    
    def __init__(
        self,
        models_dir: str = 'models/',
        device: str = 'auto',
        confidence: float = 0.5,
        attr_confidence: float = 0.0,
        action_confidence: float = 0.6,
        min_event_separation: int = 3,
        assign_single_fighter: bool = False,
        side_confidence_min: float = 0.0,
        stable_side_frames: int = 0,
        use_gap_grouping: bool = False,
        backend: str = 'pytorch',
    ):
        self.models_dir = Path(models_dir)
        self.backend = backend
        self.confidence = confidence
        self.attr_confidence = attr_confidence
        self.action_confidence = action_confidence
        self.min_event_separation = min_event_separation
        self.assign_single_fighter = assign_single_fighter
        self.side_confidence_min = side_confidence_min
        self.stable_side_frames = stable_side_frames
        self.use_gap_grouping = use_gap_grouping

        # ONNX-specific state (populated by _load_onnx_models)
        self._ort_unified_session = None
        self._ort_unified_meta = None
        self._ort_action_sessions = {}
        self._ort_action_metas = {}

        # PyTorch-specific state
        self.unified_model = None
        self.action_models = {}
        self.label_maps = {}
        self._detection_unavailable_warned = False

        if backend == 'onnx':
            self.device = device  # stored as string for ORT provider selection
            self._load_onnx_models()
        else:
            self.device = get_device(device)
            print(f"Using device: {self.device}")
            self._load_models()
            self._setup_transforms()

        self._init_tracker()

        # Gap threshold derived from action model window sizes (~0.75 * max window)
        window_sizes = []
        if self.action_models:
            window_sizes = [m.get('window_size', 16) for m in self.action_models.values()]
        if self._ort_action_metas:
            window_sizes = [m.get('window_size', 16) for m in self._ort_action_metas.values()]
        if window_sizes:
            self._gap_threshold = max(12, int(0.75 * max(window_sizes)))
        else:
            self._gap_threshold = None
    
    def get_gap_threshold(self):
        """Return gap_threshold if gap-based grouping is enabled, else None.
        
        When None, callers use peak detection (default behaviour).
        When an int, callers use _group_action_events() and peak detection
        is disabled for non-defense action types.
        """
        if self.use_gap_grouping and self._gap_threshold is not None:
            return self._gap_threshold
        return None
    
    def _init_tracker(self):
        """Initialize tracker state for persistent corner assignment and temporal smoothing.
        
        Track dict (created in _assign_tracks_and_corners) also includes:
        - box_history: last 10 box positions for velocity prediction (Story 6/7)
        - embedding: L2-normalized ROI features for appearance-based matching
        """
        self.tracks = {}  # {track_id: {
                          #   'box': [...],
                          #   'corner': 'blue'|'red'|None,
                          #   'frames_seen': 0,
                          #   'last_seen': frame_idx,
                          #   'role_history': [],      # Last N role predictions
                          #   'stance_history': [],    # Last N stance predictions
                          #   'lead_hand_history': [], # Last N lead_hand predictions
                          #   'headgear_history': [],  # Last N headgear predictions
                          #   'box_history': [],       # Last 10 box positions (velocity)
                          #   'embedding': np.ndarray  # Appearance for matching
                          # }}
        self.next_track_id = 0
        self.iou_threshold = 0.2  # Minimum IoU to match detection to existing track (lowered for better tracking)
        self.max_frames_missing = 9999  # Effectively never remove tracks during a video (corners are permanent)
        self.corner_confidence_threshold = 0.7  # High confidence needed to assign corner initially
        
        # Story 8: Role confidence thresholds for referee detection
        self.role_confidence_threshold = 0.7  # Require >0.7 to trust 'fighter' prediction
        self.role_confidence_referee_default = 0.5  # Below this, default to referee
        self.referee_appearance_threshold = 0.4  # Max appearance similarity to fighters below this = likely referee
        
        # Temporal smoothing settings
        self.role_history_size = 60  # 2 seconds at 30fps - longer for critical attribute
        self.stance_history_size = 30  # 1 second at 30fps
        self.lead_hand_history_size = 30
        self.headgear_history_size = 30
        
        # Event log for persistent display (prevents flickering)
        self.event_log = []  # List of recent events: {'frame': int, 'type': str, 'text': str, 'color': tuple}
        self.max_event_log_size = 5  # Show last 5 events on screen
        self.event_display_duration = 90  # Keep events visible for ~3 seconds at 30fps
        
        # Peak detection state for real-time event detection
        # Track per-action, per-fighter confidence for peak detection
        self.action_confidence_history = {at: {'red': [], 'blue': [], 'unknown': []} for at in ACTION_TYPES}
        self.last_event_frame = {at: {'red': -999, 'blue': -999, 'unknown': -999} for at in ACTION_TYPES}
        # Note: self.min_event_separation is set in __init__
    
    def _reset_tracker(self):
        """Reset tracker state (call at start of new video)."""
        self.tracks = {}
        self.next_track_id = 0
        
        # Reset event log
        self.event_log = []
        
        # Reset peak detection state
        self.action_confidence_history = {at: {'red': [], 'blue': [], 'unknown': []} for at in ACTION_TYPES}
        self.last_event_frame = {at: {'red': -999, 'blue': -999, 'unknown': -999} for at in ACTION_TYPES}
    
    def _get_majority_vote(self, history: List[str], default: str = 'unknown') -> str:
        """Get the most common value from a history list (majority vote)."""
        if not history:
            return default
        # Count occurrences, excluding 'unknown' and 'uncertain'
        valid_votes = [v for v in history if v not in ['unknown', 'uncertain', None]]
        if not valid_votes:
            return default
        from collections import Counter
        counts = Counter(valid_votes)
        return counts.most_common(1)[0][0]
    
    def _apply_temporal_smoothing(self, person: Dict, track: Dict) -> Dict:
        """
        Apply temporal smoothing to attributes that should be stable over time.
        Uses majority vote from recent history.
        
        - role: Critical - use 60 frame history (self-correcting)
        - stance/lead_hand: Use 30 frame history
        - headgear: Use 30 frame history
        """
        # Update histories with current predictions
        role_pred = person.get('role', 'unknown')
        stance_pred = person.get('stance', 'unknown')
        lead_hand_pred = person.get('lead_hand', 'unknown')
        headgear_pred = person.get('headgear', 'unknown')
        
        # Add to history (keeping only last N)
        track['role_history'].append(role_pred)
        track['role_history'] = track['role_history'][-self.role_history_size:]
        
        track['stance_history'].append(stance_pred)
        track['stance_history'] = track['stance_history'][-self.stance_history_size:]
        
        track['lead_hand_history'].append(lead_hand_pred)
        track['lead_hand_history'] = track['lead_hand_history'][-self.lead_hand_history_size:]
        
        track['headgear_history'].append(headgear_pred)
        track['headgear_history'] = track['headgear_history'][-self.headgear_history_size:]
        
        # Apply majority vote
        person['role'] = self._get_majority_vote(track['role_history'], role_pred)
        person['stance'] = self._get_majority_vote(track['stance_history'], stance_pred)
        person['lead_hand'] = self._get_majority_vote(track['lead_hand_history'], lead_hand_pred)
        person['headgear'] = self._get_majority_vote(track['headgear_history'], headgear_pred)
        
        return person
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / max(union, 1e-6)
    
    def _interpolate_box(self, box1: List[float], box2: List[float], t: float) -> List[float]:
        """
        Linearly interpolate between two boxes (Story 7).
        
        Args:
            box1: Start box [x1, y1, x2, y2]
            box2: End box [x1, y1, x2, y2]
            t: Interpolation factor in [0, 1] (0 = box1, 1 = box2)
            
        Returns:
            Interpolated box [x1, y1, x2, y2]
        """
        return [box1[i] + t * (box2[i] - box1[i]) for i in range(4)]
    
    def _predict_next_box(self, track: Dict) -> List[float]:
        """
        Predict next box position using velocity from box history (Story 7).
        
        Uses the last two positions in box_history to compute velocity,
        then extrapolates to predict the next position.
        
        Args:
            track: Track dictionary with 'box' and 'box_history' fields
            
        Returns:
            Predicted box [x1, y1, x2, y2]
        """
        box_history = track.get('box_history', [])
        current_box = track.get('box', [0, 0, 0, 0])
        
        # Need at least 2 boxes to compute velocity
        if len(box_history) < 2:
            return current_box
        
        # Compute velocity as difference between last two boxes
        prev_box = box_history[-2]
        last_box = box_history[-1]
        velocity = [last_box[i] - prev_box[i] for i in range(4)]
        
        # Predict next position: current + velocity
        predicted = [current_box[i] + velocity[i] for i in range(4)]
        
        return predicted
    
    def _smooth_box(self, old_box: List[float], new_box: List[float], alpha: float = 0.7) -> List[float]:
        """
        Apply exponential moving average smoothing to box position.
        
        Reduces jitter from frame-to-frame detection noise while still
        allowing the box to track movement.
        
        Args:
            old_box: Previous box position [x1, y1, x2, y2]
            new_box: New detected box position [x1, y1, x2, y2]
            alpha: Smoothing factor (0-1). Higher = more weight on new detection.
                   0.7 means 70% new, 30% old - responsive but smooth.
            
        Returns:
            Smoothed box [x1, y1, x2, y2]
        """
        return [alpha * new_box[i] + (1 - alpha) * old_box[i] for i in range(4)]
    
    def _compute_appearance_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two appearance embeddings.
        
        Since embeddings are L2-normalized, cosine similarity = dot product.
        
        Args:
            embedding1: First embedding (numpy array)
            embedding2: Second embedding (numpy array)
            
        Returns:
            Similarity score in [-1, 1], higher = more similar
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2))
    
    def _assign_tracks_and_corners(self, persons: List[Dict], frame_idx: int) -> List[Dict]:
        """
        Assign persistent track IDs and enforce sticky corner assignment.
        
        Boxing-specific strategy (max 2 fighters):
        1. Match detections to tracks by combined score: 0.7*IoU + 0.3*appearance_similarity
           (IoU threshold 0.2; appearance from L2-normalized ROI embeddings breaks ties in clinch)
        2. If <2 tracks exist: create new track for unmatched detections (if role=fighter)
        3. If 2 tracks exist: assign unmatched to best-matching track only if IoU >= 0.05
           (prevents false positives from hijacking tracks)
        4. Corners are PERMANENT once assigned - never switch
        5. Box positions are smoothed (70% new, 30% old) to reduce jitter
        6. Unmatched fighter tracks: ghost box added using velocity prediction (up to 60 frames)
        
        Story 8 - Referee detection:
        - Third detection rule: if 2 fighter tracks exist and detection is unmatched,
          classify as referee (id=-1, corner='unknown', role='referee')
        - Role confidence threshold: require >0.7 to trust 'fighter' prediction;
          <0.5 defaults to referee and doesn't create fighter track
        - Appearance-based support: low similarity to both fighters reinforces referee
        
        Embeddings are used internally only; they are stripped before saving to JSON.
        """
        # Count existing fighter tracks (those with corners assigned)
        fighter_tracks = {tid: t for tid, t in self.tracks.items() if t['corner'] in ['blue', 'red']}
        num_fighter_tracks = len(fighter_tracks)
        
        track_ids = list(self.tracks.keys())
        matched_detections = set()
        matched_tracks = set()
        
        # Step 1: Match detections to tracks by combined IoU + appearance similarity
        # Uses appearance to break ties when IoU is ambiguous (e.g. clinch scenarios)
        matches = []
        appearance_weight = 0.3  # Weight for appearance vs IoU (0.3 = 30% appearance, 70% IoU)
        
        for det_idx, person in enumerate(persons):
            box = person['box']
            det_embedding = person.get('embedding')
            
            for track_id in track_ids:
                track = self.tracks[track_id]
                iou = self._compute_iou(box, track['box'])
                
                # Compute appearance similarity (cosine, range [-1, 1] -> normalize to [0, 1])
                track_embedding = track.get('embedding')
                if det_embedding is not None and track_embedding is not None:
                    appearance_sim = self._compute_appearance_similarity(det_embedding, track_embedding)
                    # Normalize from [-1, 1] to [0, 1]
                    appearance_sim = (appearance_sim + 1) / 2
                else:
                    appearance_sim = 0.5  # Neutral if no embedding available
                
                # Combined score: weighted sum of IoU and appearance
                combined_score = (1 - appearance_weight) * iou + appearance_weight * appearance_sim
                matches.append((combined_score, iou, det_idx, track_id))
        
        # Sort by combined score descending
        matches.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy assignment with IoU threshold (still require minimum IoU)
        for combined_score, iou, det_idx, track_id in matches:
            if det_idx in matched_detections or track_id in matched_tracks:
                continue
            if iou < self.iou_threshold:
                continue  # Below threshold, will handle later
            
            # Match found
            person = persons[det_idx]
            track = self.tracks[track_id]
            
            person['id'] = track_id
            
            # Use sticky corner (never change)
            if track['corner'] is not None:
                person['corner'] = track['corner']
            
            # Apply temporal smoothing for role, stance, lead_hand, headgear
            person = self._apply_temporal_smoothing(person, track)
            
            # Update track state with box smoothing to reduce jitter
            smoothed_box = self._smooth_box(track['box'], person['box'], alpha=0.7)
            track['box'] = smoothed_box
            person['box'] = smoothed_box  # Update displayed box too
            track['last_seen'] = frame_idx
            track['frames_seen'] += 1
            # Update box history for velocity prediction (Story 6)
            track['box_history'].append(smoothed_box)
            track['box_history'] = track['box_history'][-10:]  # Keep last 10 positions
            
            # Update track embedding (exponential moving average for stability)
            if person.get('embedding') is not None:
                if track.get('embedding') is not None:
                    # Blend: 70% new, 30% old for responsiveness
                    track['embedding'] = 0.7 * person['embedding'] + 0.3 * track['embedding']
                else:
                    track['embedding'] = person['embedding']
            
            matched_detections.add(det_idx)
            matched_tracks.add(track_id)
        
        # Step 2: Handle unmatched detections
        for det_idx, person in enumerate(persons):
            if det_idx in matched_detections:
                continue
            
            box = person['box']
            det_embedding = person.get('embedding')
            
            # If we already have 2 fighter tracks, assign to closest track (no new tracks)
            if num_fighter_tracks >= 2:
                # Find best unmatched track by combined IoU + appearance
                # Require minimum IoU to prevent false positives from hijacking tracks
                min_iou_for_assignment = 0.05  # Very low threshold - just needs SOME overlap
                best_score = -1
                best_iou = -1
                best_track_id = None
                
                for track_id in fighter_tracks:
                    if track_id in matched_tracks:
                        continue
                    track = self.tracks[track_id]
                    iou = self._compute_iou(box, track['box'])
                    
                    # Compute appearance similarity
                    track_embedding = track.get('embedding')
                    if det_embedding is not None and track_embedding is not None:
                        appearance_sim = self._compute_appearance_similarity(det_embedding, track_embedding)
                        appearance_sim = (appearance_sim + 1) / 2  # Normalize to [0, 1]
                    else:
                        appearance_sim = 0.5
                    
                    # Combined score (same weights as Step 1)
                    combined_score = 0.7 * iou + 0.3 * appearance_sim
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_iou = iou
                        best_track_id = track_id
                
                # Only assign if IoU meets minimum threshold (prevents false positive hijacking)
                if best_track_id is not None and best_iou >= min_iou_for_assignment:
                    # Assign to best matching track
                    track = self.tracks[best_track_id]
                    person['id'] = best_track_id
                    person['corner'] = track['corner']  # Sticky corner
                    
                    # Apply temporal smoothing for role, stance, lead_hand, headgear
                    person = self._apply_temporal_smoothing(person, track)
                    
                    # Update track state with box smoothing to reduce jitter
                    smoothed_box = self._smooth_box(track['box'], box, alpha=0.7)
                    track['box'] = smoothed_box
                    person['box'] = smoothed_box  # Update displayed box too
                    track['last_seen'] = frame_idx
                    track['frames_seen'] += 1
                    # Update box history for velocity prediction (Story 6)
                    track['box_history'].append(smoothed_box)
                    track['box_history'] = track['box_history'][-10:]  # Keep last 10 positions
                    
                    # Update track embedding
                    if det_embedding is not None:
                        if track.get('embedding') is not None:
                            track['embedding'] = 0.7 * det_embedding + 0.3 * track['embedding']
                        else:
                            track['embedding'] = det_embedding
                    
                    matched_tracks.add(best_track_id)
                else:
                    # Story 8: Third-detection rule - if 2 fighter tracks exist and this
                    # detection doesn't match either, treat as referee (not a false positive)
                    # 
                    # Additional signal: compute max appearance similarity to both fighters.
                    # If similarity is low, this strongly supports referee classification
                    # (referee looks different from both fighters).
                    max_appearance_sim = 0.0
                    for tid in fighter_tracks:
                        track = self.tracks[tid]
                        track_embedding = track.get('embedding')
                        if det_embedding is not None and track_embedding is not None:
                            sim = self._compute_appearance_similarity(det_embedding, track_embedding)
                            sim = (sim + 1) / 2  # Normalize to [0, 1]
                            max_appearance_sim = max(max_appearance_sim, sim)
                    
                    # Classify as referee: unmatched third person with 2 fighters already tracked
                    person['id'] = -1  # Non-fighter ID
                    person['corner'] = 'unknown'
                    person['role'] = 'referee'  # Explicit referee classification (Story 8)
                    
                    # If appearance similarity is also low, we have high confidence this is referee
                    if max_appearance_sim < self.referee_appearance_threshold:
                        person['referee_confidence'] = 'high'  # Low similarity = definitely not a fighter
                    else:
                        person['referee_confidence'] = 'medium'  # Rule-based (third person)
            else:
                # Less than 2 fighter tracks - potentially create new track
                # 
                # Story 8: Check role confidence before treating as fighter
                # If model predicts 'referee' or confidence is too low, don't create fighter track
                role_pred = person.get('role', 'fighter')
                role_conf = person.get('role_confidence', 1.0)  # Default high if not provided
                
                # Case 1: Model explicitly predicts referee
                # Case 2: Model predicts fighter but confidence is very low (< 0.5)
                # In either case, treat as referee and don't create fighter track
                if role_pred == 'referee' or role_conf < self.role_confidence_referee_default:
                    person['id'] = -1
                    person['corner'] = 'unknown'
                    person['role'] = 'referee'
                    if role_pred == 'referee':
                        person['referee_confidence'] = 'high'  # Model says referee
                    else:
                        person['referee_confidence'] = 'low'  # Uncertain fighter → default referee
                    # Don't create a track for this person
                    continue
                
                # Case 3: Model predicts fighter but confidence is medium (0.5-0.7)
                # Create track but mark as uncertain; temporal smoothing will help
                is_uncertain_fighter = role_conf < self.role_confidence_threshold
                
                track_id = self.next_track_id
                self.next_track_id += 1
                person['id'] = track_id
                
                # Determine corner assignment
                taken_corners = {t['corner'] for t in self.tracks.values() if t['corner'] in ['blue', 'red']}
                available_corners = [c for c in ['blue', 'red'] if c not in taken_corners]
                
                assigned_corner = None
                corner_conf = person.get('corner_confidence', 0.0)
                corner_pred = person.get('corner')
                
                # Use model prediction if confident and available
                if corner_conf >= self.corner_confidence_threshold and corner_pred in available_corners:
                    assigned_corner = corner_pred
                    available_corners.remove(corner_pred)
                
                # Otherwise assign arbitrarily from available
                if assigned_corner is None and available_corners:
                    assigned_corner = available_corners.pop(0)
                
                if assigned_corner is not None:
                    person['corner'] = assigned_corner
                    num_fighter_tracks += 1
                
                # Mark uncertain fighters for debugging/analysis
                if is_uncertain_fighter:
                    person['role_uncertain'] = True
                
                # Create new track with history arrays for temporal smoothing
                self.tracks[track_id] = {
                    'box': box,
                    'corner': assigned_corner,
                    'frames_seen': 1,
                    'last_seen': frame_idx,
                    'role_history': [],
                    'stance_history': [],
                    'lead_hand_history': [],
                    'headgear_history': [],
                    'box_history': [],  # For velocity prediction (Story 6)
                    'embedding': person.get('embedding'),  # For appearance tracking
                }
                
                # Initialize box history with first position (Story 6)
                self.tracks[track_id]['box_history'].append(box)
                
                # Apply temporal smoothing (initializes history with first prediction)
                person = self._apply_temporal_smoothing(person, self.tracks[track_id])
                
                # Update fighter_tracks for subsequent iterations
                if assigned_corner in ['blue', 'red']:
                    fighter_tracks[track_id] = self.tracks[track_id]
        
        # Step 3: Carry forward unmatched fighter tracks (reduces flickering)
        # If a fighter track wasn't matched, add a "ghost" detection using velocity prediction (Story 7)
        for track_id, track in fighter_tracks.items():
            if track_id not in matched_tracks:
                # This fighter wasn't detected this frame - predict position using velocity
                # Only carry forward if recently seen (within 60 frames = ~2 seconds at 30fps)
                # Extended from 30 frames for smoother visual continuity (Story 6)
                frames_missing = frame_idx - track.get('last_seen', 0)
                if frames_missing <= 60:
                    # Use velocity-based prediction for smoother tracking (Story 7)
                    # For first missing frame, use predicted box; for subsequent frames,
                    # continue predicting from the last predicted position
                    if frames_missing == 1:
                        predicted_box = self._predict_next_box(track)
                    else:
                        # For longer gaps, extrapolate further using velocity
                        # Each frame adds one more velocity step
                        box_history = track.get('box_history', [])
                        if len(box_history) >= 2:
                            prev_box = box_history[-2]
                            last_box = box_history[-1]
                            velocity = [last_box[i] - prev_box[i] for i in range(4)]
                            # Extrapolate: last_known + velocity * frames_missing
                            predicted_box = [track['box'][i] + velocity[i] * frames_missing for i in range(4)]
                        else:
                            predicted_box = track['box']  # Fallback to last known
                    
                    ghost_person = {
                        'id': track_id,
                        'box': predicted_box,  # Velocity-predicted position (Story 7)
                        'confidence': 0.0,  # Mark as interpolated
                        'label': 1,
                        'corner': track['corner'],
                        'interpolated': True,  # Flag to indicate this is carried forward
                        # Use majority vote from history for attributes
                        'role': self._get_majority_vote(track.get('role_history', []), 'fighter'),
                        'stance': self._get_majority_vote(track.get('stance_history', []), 'unknown'),
                        'lead_hand': self._get_majority_vote(track.get('lead_hand_history', []), 'unknown'),
                        'headgear': self._get_majority_vote(track.get('headgear_history', []), 'unknown'),
                        'guard': 'unknown',
                        'visibility': 'partial',
                    }
                    persons.append(ghost_person)
        
        return persons
    
    def _load_onnx_models(self):
        """Load ONNX models and metadata (onnx backend)."""
        from .ort_video_backend import create_session
        from .onnx_model_metadata import load_unified_metadata, load_action_metadata

        device_str = self.device if isinstance(self.device, str) else 'auto'

        # Unified model
        unified_onnx = self.models_dir / 'unified' / 'unified_model.onnx'
        if unified_onnx.exists():
            print(f"Loading unified ONNX model: {unified_onnx}")
            self._ort_unified_meta = load_unified_metadata(self.models_dir)
            self._ort_unified_session = create_session(unified_onnx, device_str)
        else:
            print(f"WARNING: Unified ONNX model not found: {unified_onnx}")

        # Action models
        actions_dir = self.models_dir / 'actions'
        if actions_dir.exists():
            for action_type in ACTION_TYPES:
                onnx_path = actions_dir / action_type / f'{action_type}_model.onnx'
                if onnx_path.exists():
                    print(f"Loading {action_type} ONNX model: {onnx_path}")
                    meta = load_action_metadata(self.models_dir, action_type)
                    session = create_session(onnx_path, device_str)
                    self._ort_action_sessions[action_type] = session
                    self._ort_action_metas[action_type] = meta

        if not self._ort_action_sessions:
            print("WARNING: No action ONNX models found.")

    def _load_models(self):
        """Load all available models."""
        
        # Try to load unified model - check both standard and MPS versions (best.pt or last.pt)
        unified_dir = self.models_dir / 'unified'
        unified_mps_dir = self.models_dir / 'unified_mps'
        unified_path = (unified_dir / 'best.pt') if (unified_dir / 'best.pt').exists() else (unified_dir / 'last.pt')
        unified_mps_path = (unified_mps_dir / 'best.pt') if (unified_mps_dir / 'best.pt').exists() else (unified_mps_dir / 'last.pt')
        
        if unified_path.exists():
            print(f"Loading unified model (Faster R-CNN): {unified_path}")
            self.unified_model = self._load_unified_model(unified_path, model_type='fasterrcnn')
            print("  Unified model loaded successfully")
        elif unified_mps_path.exists():
            print(f"Loading unified model (SSD/MPS): {unified_mps_path}")
            self.unified_model = self._load_unified_model(unified_mps_path, model_type='ssd_mps')
            print("  Unified MPS model loaded successfully")
        else:
            print(f"WARNING: No unified model found at:")
            print(f"  - {unified_dir / 'best.pt'} or {unified_dir / 'last.pt'}")
            print(f"  - {unified_mps_dir / 'best.pt'} or {unified_mps_dir / 'last.pt'}")
            print("  Detection and attribute classification will not be available.")
        
        # Load action models (for all 7 types if present)
        actions_dir = self.models_dir / 'actions'
        if actions_dir.exists():
            for action_type in ACTION_TYPES:
                action_path = actions_dir / action_type / 'best.pt'
                if action_path.exists():
                    print(f"Loading {action_type} model: {action_path}")
                    self.action_models[action_type] = self._load_action_model(action_path)
                    print(f"  {action_type} model loaded")
        
        if not self.action_models:
            print("WARNING: No action models found. Action detection will not be available.")
    
    def _load_unified_model(self, path: Path, model_type: str = 'fasterrcnn') -> Dict:
        """Load unified detection + attributes model.
        
        Reconstructs architecture from checkpoint config (hidden_dim, num_layers,
        attribute_dropout) so both new (512/2-layer) and old (256/1-layer) checkpoints
        load correctly without shape mismatch.
        
        Args:
            path: Path to the checkpoint file
            model_type: Either 'fasterrcnn' (standard) or 'ssd_mps' (MPS-optimized)
        """
        # Load on CPU first to avoid torch.load map_location bugs (e.g. cuda on macOS → mps path).
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Fix attribute_config if it's just a list of names (old checkpoint format)
        attr_config = config.get('attribute_config', {})
        if isinstance(attr_config, list):
            print("  Reconstructing attribute_config from label maps...")
            attr_config = self._load_attribute_config_from_label_maps()
            config['attribute_config'] = attr_config
        
        # Reconstruct the model based on type
        try:
            if model_type == 'fasterrcnn':
                # Reconstruct Faster R-CNN based model (architecture must match checkpoint)
                # Old checkpoints lack hidden_dim/num_layers/attribute_dropout → use 256, 1, None
                UnifiedBoxingModel = _get_unified_boxing_model_class()
                hidden_dim = config.get('hidden_dim', 256)
                num_layers = config.get('num_layers', 1)
                attribute_dropout = config.get('attribute_dropout')
                model = UnifiedBoxingModel(
                    backbone_name=config.get('backbone_name', 'efficientnet_b0'),
                    pretrained_backbone=False,  # We're loading weights
                    num_classes=config.get('num_classes', 2),
                    attribute_config=attr_config,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    attribute_dropout=attribute_dropout,
                    use_focal_loss=config.get('use_focal_loss', False),
                )
            else:  # ssd_mps
                # Reconstruct SSD-based MPS model
                model = UnifiedMPSModel(
                    backbone_name=config.get('backbone_name', 'mobilenet_v3_large'),
                    num_classes=config.get('num_classes', 2),
                    attribute_config=attr_config,
                    pretrained_backbone=False,
                )
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            
            print(f"  Model reconstructed: {config.get('backbone_name', 'unknown')} backbone")
            
            return {
                'model': model,
                'config': config,
                'model_type': model_type,
            }
            
        except Exception as e:
            print(f"  ERROR: Could not fully reconstruct model: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Config was: {config}")
            print(f"  Falling back to checkpoint-only mode (limited inference)")
            return {
                'checkpoint': checkpoint,
                'config': config,
                'state_dict': state_dict,
                'model_type': model_type,
                'model': None,
            }
    
    def _load_attribute_config_from_label_maps(self) -> Dict[str, List[str]]:
        """Load attribute config from label map files in data/attributes/."""
        import json
        
        attr_config = {}
        attributes_dir = Path('data/attributes')
        
        if not attributes_dir.exists():
            print(f"  WARNING: attributes dir not found: {attributes_dir}")
            return {}
        
        for attr_dir in attributes_dir.iterdir():
            if attr_dir.is_dir():
                label_map_file = attr_dir / 'label_map.json'
                if label_map_file.exists():
                    with open(label_map_file, 'r') as f:
                        label_map = json.load(f)
                    # label_map is {class_name: index}, we need list of class names sorted by index
                    classes = sorted(label_map.keys(), key=lambda x: label_map[x])
                    attr_config[attr_dir.name] = classes
        
        print(f"  Loaded {len(attr_config)} attribute configs from label maps")
        return attr_config
    
    def _load_action_model(self, path: Path) -> Dict:
        """Load action recognition model."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        config = checkpoint.get('config', {})
        targets = checkpoint.get('targets', [])
        label_maps = checkpoint.get('label_maps', {})
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Architecture is saved at top-level in checkpoint, not in config
        architecture = checkpoint.get('architecture', config.get('architecture', 'r2plus1d'))
        
        # Get num_classes_per_target from checkpoint (preferred) or calculate from label_maps
        num_classes_per_target = checkpoint.get('num_classes_per_target', {})
        if not num_classes_per_target:
            for target in targets:
                if target in label_maps:
                    num_classes_per_target[target] = len(label_maps[target])
                else:
                    num_classes_per_target[target] = config.get('num_classes', 10)
        
        # Create reversed label maps for inference (index -> class_name)
        # Training saves {class_name: index}, but inference needs {index: class_name}
        reversed_label_maps = {}
        for target, mapping in label_maps.items():
            if isinstance(mapping, dict):
                # Reverse the mapping: {class_name: index} -> {index: class_name}
                reversed_label_maps[target] = {v: k for k, v in mapping.items()}
            elif isinstance(mapping, list):
                # Already a list of class names indexed by position
                reversed_label_maps[target] = {i: name for i, name in enumerate(mapping)}
            else:
                reversed_label_maps[target] = mapping
        
        # Try to reconstruct the model
        try:
            model = MultiHeadActionModel(
                architecture=architecture,
                targets=targets,
                num_classes_per_target=num_classes_per_target,
            )
            
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            
            print(f"    Targets: {targets}")
            print(f"    Architecture: {architecture}")
            
            return {
                'model': model,
                'config': config,
                'targets': targets,
                'label_maps': reversed_label_maps,  # Use reversed maps for inference
                'window_size': checkpoint.get('window_size', 16),
                'img_size': int(checkpoint.get('img_size', 112)),
            }
            
        except Exception as e:
            print(f"    WARNING: Could not reconstruct action model: {e}")
            import traceback
            traceback.print_exc()
            return {
                'checkpoint': checkpoint,
                'config': config,
                'targets': targets,
                'label_maps': reversed_label_maps,
                'window_size': checkpoint.get('window_size', 16),
                'img_size': int(checkpoint.get('img_size', 112)),
                'model': None,
            }
    
    @staticmethod
    def _make_action_transform(sz: int) -> T.Compose:
        """Build an action-clip transform for a given spatial size."""
        return T.Compose([
            T.Resize((sz, sz)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _setup_transforms(self):
        """Setup image transforms for inference."""
        # Transform for unified model (detection)
        # IMPORTANT: Pass PIL Images to Faster R-CNN, not pre-transformed tensors
        # Faster R-CNN's transform will handle resizing and coordinate transformation
        # This ensures boxes are returned in original image coordinates
        self.detection_transform = None  # No pre-transform - Faster R-CNN handles it
        
        # Store original image size for coordinate transformation if needed
        self._original_image_size = None
        
        # Per-spatial-size action transforms (derived from checkpoint img_size)
        sizes = {m.get('img_size', 112) for m in self.action_models.values()}
        if not sizes:
            sizes = {112}
        self._action_transforms = {sz: self._make_action_transform(sz) for sz in sizes}
        # Keep a legacy attribute for any external callers
        self.action_transform = self._action_transforms.get(112, next(iter(self._action_transforms.values())))
    
    def analyze_video(
        self,
        video_path: str,
        progress: bool = True,
        progress_callback=None,
    ) -> Dict:
        """
        Analyze entire video.
        
        Args:
            video_path: Path to video file
            progress: Show progress bar
            progress_callback: Optional callable(current_frame, total_frames, **kw)
                invoked every few frames with live stats for remote progress reporting.
            
        Returns:
            Dict with video metadata and frame-by-frame analysis
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nAnalyzing video: {video_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Reset tracker for new video
        self._reset_tracker()
        
        # Prepare result structure
        results = {
            'video': str(video_path),
            'video_name': video_path.stem,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'analysis': [],
        }
        
        # Frame buffer for action models (temporal context)
        # Buffer size = largest window_size across all loaded action models
        frame_buffer = []
        ws_list = [m.get('window_size', 16) for m in self.action_models.values()]
        ws_list += [m.get('window_size', 16) for m in self._ort_action_metas.values()]
        window_size = max(ws_list) if ws_list else 16
        
        # Process frames
        frame_idx = 0
        _pc_t0 = time.time()
        iterator = range(total_frames)
        if progress:
            iterator = tqdm(iterator, desc="Analyzing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            frame_result = self.analyze_frame(frame, frame_idx, frame_buffer)
            results['analysis'].append(frame_result)
            
            # Update frame buffer for action models
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            if len(frame_buffer) > window_size:
                frame_buffer.pop(0)
            
            frame_idx += 1
            
            if progress:
                iterator.update(1) if hasattr(iterator, 'update') else None
            
            if progress_callback and frame_idx % 5 == 0:
                boxes = len(frame_result.get('persons', []))
                confs = [
                    p['confidence'] for p in frame_result.get('persons', [])
                    if 'confidence' in p
                ]
                score_range = (
                    f"{min(confs):.4f}-{max(confs):.4f}"
                    if confs else "unknown"
                )
                progress_callback(
                    current_frame=frame_idx,
                    total_frames=total_frames,
                    fps=frame_idx / max(time.time() - _pc_t0, 1e-6),
                    boxes_detected=boxes,
                    avg_confidence=sum(confs) / len(confs) if confs else 0.0,
                    video_resolution=f"{width}x{height}",
                    video_fps=fps,
                    score_range=score_range,
                    detection_threshold=self.confidence,
                    boxes_above_threshold=boxes,
                )
        
        cap.release()
        
        return results
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_buffer: List[np.ndarray] = None,
    ) -> Dict:
        """
        Analyze single frame.
        
        Args:
            frame: BGR image (OpenCV format)
            frame_idx: Frame index
            frame_buffer: Recent frames for temporal action detection
            
        Returns:
            Dict with persons and actions detected
        """
        frame_data = {
            'frame': frame_idx,
            'timestamp': None,  # Filled in by caller if needed
            'persons': [],
            'actions': {at: {'detected': False} for at in ACTION_TYPES},
        }
        
        # Convert to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Detect persons and classify attributes
        has_unified = (
            self.unified_model is not None
            or self._ort_unified_session is not None
        )
        if has_unified:
            persons = self._detect_persons_unified(frame_rgb)
            # Apply tracking and sticky corner assignment
            persons = self._assign_tracks_and_corners(persons, frame_idx)
            # Strip embeddings before storing (not JSON serializable, only used for tracking)
            frame_data['persons'] = [
                {k: v for k, v in p.items() if k != 'embedding'}
                for p in persons
            ]
        
        # 2. Detect actions (if we have enough temporal context)
        if frame_buffer and len(frame_buffer) >= 8:
            frame_data['actions'] = self._detect_actions(frame_buffer)
            
            # 2b. Map side predictions (left/right) to corners (red/blue) using
            # detection tracking. This is the bridge between the action model
            # (which predicts "which side of the frame") and detection (which
            # knows "which side is red/blue").
            if frame_data['persons']:
                # Optional: use median position over recent frames for stable
                # left/right mapping (--stable_side_frames > 0).
                mapping_persons = frame_data['persons']
                if self.stable_side_frames > 0:
                    mapping_persons = self._get_stable_persons(frame_data['persons'])
                frame_data['actions'] = self._map_action_sides_to_corners(
                    frame_data['actions'], mapping_persons
                )
            
            # 3. Detect and log discrete events (for persistent display)
            self._detect_and_log_events(frame_idx, frame_data['actions'])
        
        return frame_data
    
    def _detect_persons_unified(self, frame_rgb: np.ndarray) -> List[Dict]:
        """
        Detect persons and classify attributes using unified model.
        
        Supports two model architectures:
        - 'fasterrcnn': From 07_train_unified.py (standard, uses torchvision Faster R-CNN)
        - 'ssd_mps': From 07_train_unified_mps.py (MPS-optimized, custom SSD architecture)
        - ONNX: Via ORT session when backend='onnx'
        """
        if self.backend == 'onnx':
            return self._detect_persons_unified_onnx(frame_rgb)

        if self.unified_model is None or self.unified_model.get('model') is None:
            if not self._detection_unavailable_warned:
                reason = "not loaded" if self.unified_model is None else "reconstruction failed"
                print(f"WARNING: Unified detection unavailable ({reason}). "
                      "Person boxes will be empty for all frames.")
                self._detection_unavailable_warned = True
            return []
        
        model = self.unified_model['model']
        
        persons = []
        model_type = self.unified_model.get('model_type', 'fasterrcnn')
        config = self.unified_model.get('config', {})
        attr_config = config.get('attribute_config', {})
        
        pil_image = Image.fromarray(frame_rgb)
        
        with torch.no_grad():
            if model_type == 'fasterrcnn':
                # Convert to tensor on model device so backbone/transform don't
                # hit a CPU-vs-accelerator mismatch.
                image_tensor = T.functional.to_tensor(pil_image).to(self.device)
                outputs = model([image_tensor])
                
                if not outputs or len(outputs) == 0:
                    print("DEBUG: Model returned empty outputs list")
                    return []
                
                output = outputs[0]
                boxes = output.get('boxes', torch.tensor([]))
                scores = output.get('scores', torch.tensor([]))
                labels = output.get('labels', torch.tensor([]))
                attributes = output.get('attributes', {})
                embeddings = output.get('embeddings', torch.tensor([]))  # For appearance tracking
                
                # Debug: print detection info (only first frame to avoid spam)
                if not hasattr(self, '_debug_printed'):
                    if len(boxes) == 0:
                        print(f"DEBUG: Model returned 0 boxes (confidence threshold: {self.confidence})", flush=True)
                    elif len(boxes) > 0:
                        max_score = float(scores.max()) if len(scores) > 0 else 0.0
                        min_score = float(scores.min()) if len(scores) > 0 else 0.0
                        print(f"DEBUG: Model returned {len(boxes)} boxes, scores range: [{min_score:.4f}, {max_score:.4f}], threshold: {self.confidence}", flush=True)
                        print(f"DEBUG: Boxes above threshold: {sum(1 for s in scores if s >= self.confidence)}", flush=True)
                    self._debug_printed = True
                
                # Process ALL detected boxes (not just first frame!)
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    if score < self.confidence:
                        continue
                    
                    person = {
                        'id': i,
                        'box': box.cpu().tolist(),
                        'confidence': float(score.cpu()),
                        'label': int(label.cpu()),
                    }
                    
                    # Extract appearance embedding for tracking
                    if len(embeddings) > i:
                        person['embedding'] = embeddings[i].cpu().numpy()
                    else:
                        person['embedding'] = None
                    
                    # Extract attribute predictions with confidence
                    for attr_name, attr_data in attributes.items():
                        preds = attr_data.get('predictions', torch.tensor([]))
                        probs = attr_data.get('probabilities', torch.tensor([]))
                        
                        if i < len(preds):
                            pred_idx = int(preds[i].cpu())
                            # Get class name from config
                            attr_classes = attr_config.get(attr_name, [])
                            
                            # Get confidence for this prediction
                            attr_conf = 0.0
                            if i < len(probs) and pred_idx < probs.shape[1]:
                                attr_conf = float(probs[i, pred_idx].cpu())
                            person[f'{attr_name}_confidence'] = attr_conf
                            
                            # Apply confidence threshold
                            if attr_conf < self.attr_confidence:
                                person[attr_name] = 'uncertain'
                            elif pred_idx < len(attr_classes):
                                person[attr_name] = attr_classes[pred_idx]
                            else:
                                person[attr_name] = f'class_{pred_idx}'
                        else:
                            person[attr_name] = 'uncertain'
                            person[f'{attr_name}_confidence'] = 0.0
                    
                    # Set defaults for missing attributes
                    for attr_name in ['corner', 'guard', 'stance', 'lead_hand', 'visibility']:
                        if attr_name not in person:
                            person[attr_name] = 'unknown'
                    
                    persons.append(person)
            
            else:  # ssd_mps
                # SSD model returns dict with raw outputs
                # For MPS model, need to transform image to tensor
                input_tensor = T.Compose([
                    T.Resize(640),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])(pil_image).to(self.device)
                outputs = model(input_tensor.unsqueeze(0))
                
                # Parse SSD outputs (simplified - full implementation would do NMS etc.)
                cls_logits = outputs.get('cls_logits')
                if cls_logits is not None:
                    # Get predictions from logits
                    probs = torch.softmax(cls_logits, dim=-1)
                    # This is simplified - real implementation needs anchor decoding
                    # For now, return empty if MPS model (needs more complex postprocessing)
                    pass
        
        return persons
    
    def _detect_actions(self, frame_buffer: List[np.ndarray]) -> Dict:
        """
        Detect actions using temporal action models.
        
        Args:
            frame_buffer: List of recent frames (RGB numpy arrays)
            
        Returns:
            Dict with detection results for each action type
        """
        if self.backend == 'onnx':
            return self._detect_actions_onnx(frame_buffer)

        actions = {at: {'detected': False} for at in ACTION_TYPES}
        if not self.action_models:
            return actions
        
        # For each action type with a loaded model
        for action_type, model_info in self.action_models.items():
            model = model_info.get('model')
            if model is None:
                continue
            
            window_size = model_info.get('window_size', 16)
            label_maps = model_info.get('label_maps', {})
            targets = model_info.get('targets', [])
            
            # Sample frames from buffer
            if len(frame_buffer) < window_size:
                continue
            
            # Get evenly spaced frames
            indices = np.linspace(0, len(frame_buffer) - 1, window_size, dtype=int)
            frames = [frame_buffer[i] for i in indices]
            
            # Select transform matching this model's training spatial size
            sz = model_info.get('img_size', 112)
            tf = self._action_transforms.get(sz)
            if tf is None:
                tf = self._make_action_transform(sz)
                self._action_transforms[sz] = tf
            
            # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
            frame_tensors = []
            for frame in frames:
                img = Image.fromarray(frame)
                tensor = tf(img)
                frame_tensors.append(tensor)
            
            # Stack: (T, C, H, W) -> (C, T, H, W)
            video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0)  # (1, C, T, H, W)
            video_tensor = video_tensor.to(self.device)
            
            try:
                with torch.no_grad():
                    outputs = model(video_tensor)
                
                # Parse multi-head outputs
                action_result = {'detected': False}
                
                for target in targets:
                    if target not in outputs:
                        continue
                    
                    logits = outputs[target]
                    probs = torch.softmax(logits, dim=-1)
                    pred_class = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0, pred_class].item()
                    
                    # Decode label using reversed label_maps (index -> class_name)
                    if target in label_maps:
                        target_map = label_maps[target]
                        if isinstance(target_map, dict) and pred_class in target_map:
                            pred_label = target_map[pred_class]
                        elif isinstance(target_map, list) and pred_class < len(target_map):
                            pred_label = target_map[pred_class]
                        else:
                            pred_label = str(pred_class)
                    else:
                        pred_label = str(pred_class)
                    
                    action_result[target] = pred_label
                    action_result[f'{target}_confidence'] = confidence
                    
                    # Mark as detected if any target has confidence above threshold
                    # Higher threshold reduces over-counting by creating gaps for grouping
                    if confidence > self.action_confidence:
                        action_result['detected'] = True
                
                actions[action_type] = action_result
                
            except Exception as e:
                # Model inference failed - log error for debugging
                if not hasattr(self, '_action_error_logged'):
                    print(f"    DEBUG: Action model inference error ({action_type}): {e}")
                    self._action_error_logged = True
        
        return actions
    
    # -----------------------------------------------------------------
    # ONNX backend implementations
    # -----------------------------------------------------------------

    def _detect_persons_unified_onnx(self, frame_rgb: np.ndarray) -> List[Dict]:
        """Run unified detection via ORT session."""
        if self._ort_unified_session is None:
            return []

        from .ort_video_backend import (
            preprocess_unified,
            run_unified,
            parse_unified_outputs,
        )

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        meta = self._ort_unified_meta
        inp, sx, sy = preprocess_unified(
            frame_bgr,
            meta['input_height'],
            meta['input_width'],
        )
        raw = run_unified(self._ort_unified_session, inp)
        return parse_unified_outputs(
            raw, meta, sx, sy,
            confidence=self.confidence,
            attr_confidence=self.attr_confidence,
        )

    def _detect_actions_onnx(self, frame_buffer: List[np.ndarray]) -> Dict:
        """Run action detection via ORT sessions."""
        from .ort_video_backend import (
            preprocess_action_clip,
            run_action,
            parse_action_outputs,
        )

        actions = {at: {'detected': False} for at in ACTION_TYPES}
        if not self._ort_action_sessions:
            return actions

        for action_type, session in self._ort_action_sessions.items():
            meta = self._ort_action_metas[action_type]
            window_size = meta.get('window_size', 16)
            img_size = meta.get('img_size', 112)

            clip = preprocess_action_clip(frame_buffer, window_size, img_size)
            if clip is None:
                continue

            try:
                raw = run_action(session, clip)
                actions[action_type] = parse_action_outputs(
                    raw, meta, self.action_confidence,
                )
            except Exception as e:
                if not hasattr(self, '_onnx_action_error_logged'):
                    print(f"    DEBUG: ORT action inference error ({action_type}): {e}")
                    self._onnx_action_error_logged = True

        return actions

    def _get_stable_persons(self, persons: List[Dict]) -> List[Dict]:
        """
        Return a copy of persons with box positions replaced by median positions
        over the last K frames (from track box_history).
        
        This smooths out momentary swaps so left/right mapping is more stable.
        K is self.stable_side_frames.
        """
        stable = []
        k = self.stable_side_frames
        for p in persons:
            p_copy = dict(p)
            track_id = p.get('id')
            if track_id is not None and track_id >= 0 and track_id in self.tracks:
                history = self.tracks[track_id].get('box_history', [])
                recent = history[-k:] if len(history) >= k else history
                if recent:
                    # Compute median center_x from recent boxes and build a
                    # synthetic box centered at that position (same width/height
                    # as current box).
                    median_cx = statistics.median((b[0] + b[2]) / 2 for b in recent)
                    box = list(p_copy['box'])
                    w = box[2] - box[0]
                    box[0] = median_cx - w / 2
                    box[2] = median_cx + w / 2
                    p_copy['box'] = box
            stable.append(p_copy)
        return stable
    
    def _map_action_sides_to_corners(
        self,
        actions: Dict,
        persons: List[Dict],
    ) -> Dict:
        """
        Map action model side predictions (left/right) to corners (red/blue).
        
        The action model predicts which SIDE of the frame performed the action
        (e.g. attacker_side=left). Detection+tracking knows which person is on
        which side and what their corner is. This method bridges the two:
        
        1. Sort tracked fighters by center_x (leftmost = "left", rightmost = "right")
           Deterministic tie-break: (center_x, center_y, corner) with 'blue' < 'red'
        2. Build a side-to-corner mapping (e.g. left -> red, right -> blue)
        3. For each action, map the _side prediction to the original field name
           (e.g. attacker_side=left -> attacker=red)
        
        Guards:
        - Requires exactly 2 fighters with one red and one blue corner.
        - Duplicate corners or 3+ tracks -> all fighter fields set to 'unknown'.
        - Optional: single-fighter assignment (self.assign_single_fighter).
        - Optional: side confidence threshold (self.side_confidence_min).
        
        Args:
            actions: Action detection results per action type
            persons: Tracked person dicts with 'box' and 'corner' keys
            
        Returns:
            Updated actions dict with original fighter fields (attacker, defender, etc.)
            populated from the side-to-corner mapping.
        """
        # Get tracked fighters sorted by center_x; deterministic tie-break
        # by center_y then corner ('blue' < 'red') for stable ordering.
        fighters = [
            p for p in persons
            if p.get('corner') in ('red', 'blue')
        ]
        _cx = lambda p: (p['box'][0] + p['box'][2]) / 2
        _cy = lambda p: (p['box'][1] + p['box'][3]) / 2
        fighters.sort(key=lambda p: (_cx(p), _cy(p), p.get('corner', '')))
        
        # Guard: require exactly 2 fighters with one red and one blue.
        # Duplicate corners or 3+ tracks produce ambiguous mapping.
        corners = {p['corner'] for p in fighters}
        valid_pair = len(fighters) == 2 and corners == {'red', 'blue'}
        
        # Single-fighter assignment (optional): when exactly 1 fighter is
        # tracked, map both side predictions to that fighter's corner.
        single_fighter = (
            len(fighters) == 1
            and getattr(self, 'assign_single_fighter', False)
        )
        
        if not valid_pair and not single_fighter:
            # Can't determine left/right mapping.
            # Only clear fighter for new-model actions (side_val is left/right).
            # Old models set attacker/defender to red/blue directly — leave those alone.
            for action_type, action in actions.items():
                fighter_key = FIGHTER_KEYS_MAP.get(action_type)
                fighter_field = FIGHTER_FIELDS_MAP.get(action_type)
                if fighter_key and fighter_field and action.get('detected'):
                    side_val = action.get(fighter_key)
                    if side_val in ('left', 'right'):
                        action[fighter_field] = 'unknown'
            return actions
        
        # Build mapping: side -> corner
        if single_fighter:
            # Both sides map to the only visible fighter's corner
            side_to_corner = {
                'left': fighters[0]['corner'],
                'right': fighters[0]['corner'],
            }
        else:
            side_to_corner = {
                'left': fighters[0]['corner'],   # leftmost fighter's corner
                'right': fighters[-1]['corner'],  # rightmost fighter's corner
            }
        
        # Map each action's _side prediction to the original fighter field
        for action_type, action in actions.items():
            if not action.get('detected'):
                continue
            
            fighter_key = FIGHTER_KEYS_MAP.get(action_type)   # e.g. 'attacker_side'
            fighter_field = FIGHTER_FIELDS_MAP.get(action_type)  # e.g. 'attacker'
            
            if not fighter_key or not fighter_field:
                continue  # stoppage etc. - no fighter
            
            side_val = action.get(fighter_key)  # e.g. 'left' (new model) or None (old model)
            
            # Backward compatibility: only map side→corner when the model
            # actually predicts _side (left/right). Old checkpoints output
            # attacker/defender as red/blue directly — leave those untouched.
            if side_val in ('left', 'right'):
                corner = side_to_corner.get(side_val, 'unknown')
                action[fighter_field] = corner
                
                # Side confidence guard (overlap/clinch)
                side_conf_min = getattr(self, 'side_confidence_min', 0.0)
                if side_conf_min > 0:
                    side_conf = action.get(f'{fighter_key}_confidence', 1.0)
                    if side_conf < side_conf_min:
                        action[fighter_field] = 'unknown'
        
        return actions
    
    def _detect_and_log_events(self, frame_idx: int, actions: Dict):
        """
        Detect new action events using real-time peak detection and add to event log.
        
        Uses per-fighter confidence tracking to find peaks (local maxima) that represent
        discrete events. This prevents flickering by only logging when a new event occurs.
        
        Args:
            frame_idx: Current frame index
            actions: Action detection results for current frame
        """
        # Process each action type
        for action_type in ACTION_TYPES:
            action = actions.get(action_type, {})
            if not action.get('detected'):
                continue
            
            # Get fighter identifier: use the ORIGINAL field (attacker, defender, etc.)
            # which was populated by _map_action_sides_to_corners from the _side prediction
            fighter_field = FIGHTER_FIELDS_MAP.get(action_type)
            fighter = action.get(fighter_field, 'unknown') if fighter_field else 'unknown'
            if fighter not in ['red', 'blue']:
                fighter = 'unknown'
            
            # Get confidence for this detection (keys from YAML via CONFIDENCE_KEYS_MAP)
            confidences = [action.get(key, 0) for key in CONFIDENCE_KEYS_MAP.get(action_type, ['type_confidence'])]
            confidence = max(confidences) if confidences else 0
            
            # Check if this is above minimum threshold
            if confidence < self.action_confidence:
                continue
            
            # Update confidence history (keep last 10 frames)
            history = self.action_confidence_history[action_type][fighter]
            history.append((frame_idx, confidence))
            # Keep only recent history
            self.action_confidence_history[action_type][fighter] = [
                (f, c) for f, c in history if frame_idx - f <= 15
            ]
            
            # Check if this frame is a local peak for this fighter
            # A peak occurs when current confidence is >= all recent confidences
            # and we haven't logged an event for this fighter recently
            last_event = self.last_event_frame[action_type][fighter]
            if frame_idx - last_event < self.min_event_separation:
                continue  # Too soon since last event for this fighter
            
            # Check if this is a peak (current confidence is highest in recent window)
            is_peak = True
            for f, c in self.action_confidence_history[action_type][fighter]:
                if f != frame_idx and c > confidence:
                    is_peak = False
                    break
            
            if is_peak:
                # This is a new event! Log it.
                self.last_event_frame[action_type][fighter] = frame_idx
                
                # Format event text from config (KEY_ATTRIBUTES_MAP + FIGHTER_FIELDS_MAP)
                text = _format_event_text(action_type, action)
                
                # Add to event log
                event = {
                    'frame': frame_idx,
                    'type': action_type,
                    'text': text,
                    'color': COLORS.get(action_type, (255, 255, 255)),
                    'confidence': confidence,
                }
                self.event_log.append(event)
                
                # Keep only last N events
                if len(self.event_log) > self.max_event_log_size:
                    self.event_log.pop(0)
    
    def _get_visible_events(self, current_frame: int) -> List[Dict]:
        """
        Get events that should be visible on the current frame.
        
        Returns events from the log that are within the display duration.
        
        Args:
            current_frame: Current frame index
            
        Returns:
            List of events to display (most recent first)
        """
        visible = []
        for event in reversed(self.event_log):
            # Show all events in the log (they're already limited to max_event_log_size)
            # But add age info for potential fading effect
            event_copy = event.copy()
            event_copy['age_frames'] = current_frame - event['frame']
            visible.append(event_copy)
            
            if len(visible) >= self.max_event_log_size:
                break
        
        return visible
    
    def create_annotated_video(
        self,
        video_path: str,
        results: Dict,
        output_path: str,
        progress: bool = True,
    ):
        """
        Create video with overlaid annotations.
        
        Args:
            video_path: Original video path
            results: Analysis results from analyze_video
            output_path: Output video path
            progress: Show progress bar
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try different codecs
        codecs = ['mp4v', 'avc1', 'XVID']
        out = None
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if out.isOpened():
                break
        
        if not out or not out.isOpened():
            print(f"WARNING: Could not create video writer. Trying fallback...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Pre-compute events using peak detection on stored results
        # This builds an event timeline for consistent annotation
        event_timeline = self._build_event_timeline(results)
        
        frame_idx = 0
        total_frames = len(results['analysis'])
        iterator = range(total_frames)
        if progress:
            iterator = tqdm(iterator, desc="Creating annotated video")
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break
            
            frame_data = results['analysis'][frame_idx]
            # Get visible events for this frame
            visible_events = self._get_events_for_frame(event_timeline, frame_idx)
            annotated = self.draw_annotations(frame, frame_data, width, height, visible_events)
            out.write(annotated)
            
            frame_idx += 1
            if progress:
                iterator.update(1) if hasattr(iterator, 'update') else None
        
        cap.release()
        out.release()
        
        print(f"Annotated video saved: {output_path}")
    
    def _build_event_timeline(self, results: Dict) -> List[Dict]:
        """
        Build a timeline of discrete events from analysis results.
        
        Uses the shared compute_all_events() function (single source of truth)
        and formats for video display.
        
        Args:
            results: Analysis results with frame-by-frame data
            
        Returns:
            List of events sorted by frame, each with 'frame', 'type', 'text', 'color'
        """
        # Use the single source of truth for event computation
        raw_events = compute_all_events(
            results,
            action_confidence=self.action_confidence,
            min_separation=self.min_event_separation,
            window_size=10,
            gap_threshold=self.get_gap_threshold(),
        )
        
        # Format events for video display
        formatted_events = []
        for event in raw_events:
            action = event['action']
            action_type = event['type']
            
            # Format event text from config (KEY_ATTRIBUTES_MAP + FIGHTER_FIELDS_MAP)
            text = _format_event_text(action_type, action)
            
            formatted_events.append({
                'frame': event['frame'],
                'type': action_type,
                'text': text,
                'color': COLORS.get(action_type, (255, 255, 255)),
                'confidence': event['confidence'],
            })
        
        return formatted_events
    
    def _get_events_for_frame(self, event_timeline: List[Dict], current_frame: int) -> List[Dict]:
        """
        Get events that should be visible on a specific frame.
        
        Shows the most recent events (up to max_event_log_size) that occurred
        before or at the current frame.
        
        Args:
            event_timeline: Pre-computed list of all events
            current_frame: Current frame index
            
        Returns:
            List of events to display (most recent first)
        """
        # Get events that occurred at or before current frame
        past_events = [e for e in event_timeline if e['frame'] <= current_frame]
        
        # Take the most recent ones
        recent_events = past_events[-self.max_event_log_size:]
        
        # Add age info and return in reverse order (most recent first)
        result = []
        for event in reversed(recent_events):
            event_copy = event.copy()
            event_copy['age_frames'] = current_frame - event['frame']
            result.append(event_copy)
        
        return result
    
    def draw_annotations(
        self,
        frame: np.ndarray,
        frame_data: Dict,
        width: int,
        height: int,
        visible_events: List[Dict] = None,
    ) -> np.ndarray:
        """
        Draw CVAT-style annotations on frame.
        
        Args:
            frame: BGR image
            frame_data: Frame analysis data
            width: Frame width
            height: Frame height
            visible_events: Pre-computed list of events to display (optional, for video annotation)
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw persons - only RED and BLUE fighters (skip unknown/noise)
        for person in frame_data.get('persons', []):
            corner = person.get('corner', 'unknown')
            
            # Skip unknown boxes - these are false positives or referee (Story 8)
            # Only draw assigned fighters (red/blue corners)
            # Referee detections have corner='unknown' and role='referee', so they are filtered here
            if corner not in ['red', 'blue']:
                continue
            
            x1, y1, x2, y2 = [int(c) for c in person['box']]
            color = COLORS.get(corner, COLORS['unknown'])
            is_interpolated = person.get('interpolated', False)
            
            # Draw bounding box (dashed for interpolated/carried forward)
            if is_interpolated:
                # Draw dashed box for interpolated positions
                dash_length = 10
                for i in range(x1, x2, dash_length * 2):
                    cv2.line(annotated, (i, y1), (min(i + dash_length, x2), y1), color, 2)
                    cv2.line(annotated, (i, y2), (min(i + dash_length, x2), y2), color, 2)
                for i in range(y1, y2, dash_length * 2):
                    cv2.line(annotated, (x1, i), (x1, min(i + dash_length, y2)), color, 2)
                    cv2.line(annotated, (x2, i), (x2, min(i + dash_length, y2)), color, 2)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Build label
            guard = person.get('guard', 'unknown')
            stance = person.get('stance', 'unknown')
            label = f"{corner.upper()} | {guard} | {stance}"
            if is_interpolated:
                label += " [?]"  # Indicate interpolated
            
            # Draw label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Label background
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            # Label text
            cv2.putText(annotated, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness)
            
            # Confidence below box (or "interp" for interpolated)
            if is_interpolated:
                conf_text = "interp"
            else:
                conf_text = f"{person.get('confidence', 0):.0%}"
            cv2.putText(annotated, conf_text, (x1, y2 + 15), font, 0.4, color, 1)
        
        # Draw persistent event log at top left (replaces flickering action tags)
        current_frame = frame_data.get('frame', 0)
        
        # Use passed events if available, otherwise get from internal state
        if visible_events is None:
            visible_events = self._get_visible_events(current_frame)
        
        if visible_events:
            # Draw semi-transparent background for event log
            log_height = len(visible_events) * 28 + 10
            log_width = 350
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 5), (5 + log_width, 5 + log_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            
            # Draw header
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated, "RECENT EVENTS", (10, 22), font, 0.5, (200, 200, 200), 1)
            
            # Draw each event
            event_y = 48
            for event in visible_events:
                text = event['text']
                color = event['color']
                age = event.get('age_frames', 0)
                
                # Fade older events slightly (but keep them readable)
                if age > 60:  # After 2 seconds, start fading
                    fade_factor = max(0.5, 1.0 - (age - 60) / 90)  # Fade over 3 more seconds
                    color = tuple(int(c * fade_factor) for c in color)
                
                # Frame indicator
                frame_text = f"[{event['frame']}]"
                cv2.putText(annotated, frame_text, (10, event_y), font, 0.4, (150, 150, 150), 1)
                
                # Event text
                cv2.putText(annotated, text, (60, event_y), font, 0.5, color, 1)
                event_y += 28
        
        # Frame number (bottom right)
        frame_text = f"Frame: {frame_data.get('frame', 0)}"
        cv2.putText(annotated, frame_text, (width - 120, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def save_annotated_frames(
        self,
        video_path: str,
        results: Dict,
        output_dir: str,
        progress: bool = True,
    ):
        """Save individual annotated frames as images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Pre-compute events using peak detection on stored results
        event_timeline = self._build_event_timeline(results)
        
        frame_idx = 0
        total_frames = len(results['analysis'])
        iterator = range(total_frames)
        if progress:
            iterator = tqdm(iterator, desc="Saving frames")
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break
            
            frame_data = results['analysis'][frame_idx]
            visible_events = self._get_events_for_frame(event_timeline, frame_idx)
            annotated = self.draw_annotations(frame, frame_data, width, height, visible_events)
            
            frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
            
            frame_idx += 1
            if progress:
                iterator.update(1) if hasattr(iterator, 'update') else None
        
        cap.release()
        print(f"Saved {frame_idx} annotated frames to: {output_dir}")


# =============================================================================
# EVENT COMPUTATION - SINGLE SOURCE OF TRUTH
# =============================================================================
# This is the ONLY function that determines what counts as a discrete event
# (punch, defense, footwork, clinch). Both video annotation and summary.json
# use this same logic to ensure consistency.

def compute_all_events(
    results: Dict,
    action_confidence: float = 0.6,
    min_separation: int = 3,
    window_size: int = 10,
    defense_anchor_window: int = 3,
    gap_threshold: Optional[int] = None,
) -> List[Dict]:
    """
    Compute all discrete action events from analysis results.
    
    SINGLE SOURCE OF TRUTH for event counting. Both the annotated video
    (RECENT EVENTS panel) and summary.json use this function with the same
    parameters, so counts always match. The returned events list is written
    to summary.json as summary['events']; UIs and reports must use that
    (and summary['actions']) and must NOT recompute events from analysis.json.
    
    Logic:
    - When gap_threshold is None (default): peak detection for all non-defense
      action types (local max in window_size frames, min_separation between
      events from same fighter).
    - When gap_threshold is not None: gap-based grouping via
      _group_action_events() for all non-defense action types; **peak detection
      is disabled** for those types.
    - Defense: PUNCH-ANCHORED in both modes. A defense event only counts if
      it's within ±defense_anchor_window frames of a punch event.
    
    Args:
        results: Analysis results with frame-by-frame data
        action_confidence: Minimum confidence to be an event candidate (default 0.6)
        min_separation: Minimum frames between events from same fighter (default 3)
        window_size: Sliding window for peak detection (default 10)
        defense_anchor_window: Defense must be within ±this many frames of a punch (default 3)
        gap_threshold: When not None, use gap-based grouping instead of peak
            detection for non-defense types. Derived from action model
            window_size (~0.75 * window). Default None = peak detection.
    
    Returns:
        List of events sorted by frame, each with:
        'frame', 'type', 'fighter', 'confidence', 'action' (full action dict)
    """
    events = []
    
    # Non-defense action types (from config)
    non_defense_types = [at for at in ACTION_TYPES if at != 'defense']
    
    if gap_threshold is not None:
        # =================================================================
        # GAP-BASED GROUPING PATH (peak detection disabled for non-defense)
        # =================================================================
        for action_type in non_defense_types:
            grouped = _group_action_events(
                results,
                action_type,
                KEY_ATTRIBUTES_MAP.get(action_type, []),
                gap_threshold=gap_threshold,
                min_event_duration=5,
            )
            for ge in grouped:
                fighter_field = FIGHTER_FIELDS_MAP.get(action_type)
                full_action = ge.get('_full_action', {})
                fighter = full_action.get(fighter_field, 'unknown') if fighter_field else 'unknown'
                if fighter not in ['red', 'blue']:
                    fighter = 'unknown'
                events.append({
                    'frame': ge['best_frame'],
                    'type': action_type,
                    'fighter': fighter,
                    'confidence': ge.get('max_confidence', 0),
                    'action': full_action,
                })
        
        punch_event_frames = [e['frame'] for e in events if e['type'] == 'punch']
    else:
        # =================================================================
        # PEAK-DETECTION PATH (default)
        # =================================================================
        
        last_event_frame = {at: {'red': -999, 'blue': -999, 'unknown': -999} for at in non_defense_types}
        confidence_windows = {at: {'red': [], 'blue': [], 'unknown': []} for at in non_defense_types}
        punch_event_frames = []
        
        for frame_idx, frame_data in enumerate(results['analysis']):
            actions = frame_data.get('actions', {})
            
            for action_type in non_defense_types:
                action = actions.get(action_type, {})
                
                if not action.get('detected'):
                    for fighter in ['red', 'blue', 'unknown']:
                        confidence_windows[action_type][fighter].append((frame_idx, 0))
                        confidence_windows[action_type][fighter] = [
                            (f, c) for f, c in confidence_windows[action_type][fighter] 
                            if frame_idx - f <= window_size
                        ]
                    continue
                
                fighter_field = FIGHTER_FIELDS_MAP.get(action_type)
                fighter = action.get(fighter_field, 'unknown') if fighter_field else 'unknown'
                if fighter not in ['red', 'blue']:
                    fighter = 'unknown'
                
                conf_keys = CONFIDENCE_KEYS_MAP.get(action_type, ['type_confidence'])
                confidences = [action.get(key, 0) for key in conf_keys]
                confidence = max(confidences) if confidences else 0
                
                confidence_windows[action_type][fighter].append((frame_idx, confidence))
                confidence_windows[action_type][fighter] = [
                    (f, c) for f, c in confidence_windows[action_type][fighter] 
                    if frame_idx - f <= window_size
                ]
                
                if confidence < action_confidence:
                    continue
                
                if frame_idx - last_event_frame[action_type][fighter] < min_separation:
                    continue
                
                is_peak = True
                for f, c in confidence_windows[action_type][fighter]:
                    if f != frame_idx and c > confidence:
                        is_peak = False
                        break
                
                if is_peak:
                    last_event_frame[action_type][fighter] = frame_idx
                    events.append({
                        'frame': frame_idx,
                        'type': action_type,
                        'fighter': fighter,
                        'confidence': confidence,
                        'action': action,
                    })
                    
                    if action_type == 'punch':
                        punch_event_frames.append(frame_idx)
    
    # =========================================================================
    # STEP 2: Compute defense events - PUNCH-ANCHORED
    # Defense only counts if within ±defense_anchor_window of a punch event
    # =========================================================================
    
    if not punch_event_frames:
        # No punches = no defense events
        return sorted(events, key=lambda e: e['frame'])
    
    # Build set of frames that are "near a punch"
    punch_adjacent_frames = set()
    for punch_frame in punch_event_frames:
        for f in range(punch_frame - defense_anchor_window, punch_frame + defense_anchor_window + 1):
            if 0 <= f < len(results['analysis']):
                punch_adjacent_frames.add(f)
    
    # For each punch event, find the best defense detection in its window
    used_defense_frames = set()
    
    for punch_frame in punch_event_frames:
        start_frame = max(0, punch_frame - defense_anchor_window)
        end_frame = min(len(results['analysis']), punch_frame + defense_anchor_window + 1)
        
        # Find defense detections in this window
        best_defense = None
        best_conf = -1
        best_frame = None
        
        for frame_idx in range(start_frame, end_frame):
            if frame_idx in used_defense_frames:
                continue
            
            frame_data = results['analysis'][frame_idx]
            defense = frame_data.get('actions', {}).get('defense', {})
            
            if not defense.get('detected'):
                continue
            
            # Get confidence
            confidences = [defense.get(key, 0) for key in CONFIDENCE_KEYS_MAP.get('defense', ['type_confidence'])]
            conf = max(confidences) if confidences else 0
            
            if conf >= action_confidence and conf > best_conf:
                best_conf = conf
                best_defense = defense
                best_frame = frame_idx
        
        if best_defense is not None:
            # Mark this frame as used
            used_defense_frames.add(best_frame)
            
            # Get fighter
            fighter = best_defense.get('defender', 'unknown')
            if fighter not in ['red', 'blue']:
                fighter = 'unknown'
            
            events.append({
                'frame': best_frame,
                'type': 'defense',
                'fighter': fighter,
                'confidence': best_conf,
                'action': best_defense,
                'punch_frame': punch_frame,  # Link to associated punch
            })
    
    return sorted(events, key=lambda e: e['frame'])


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_json(results: Dict, output_path: str):
    """Save results as JSON. Creates parent directory if it does not exist."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Analysis saved: {output_path}")


def save_csv(results: Dict, output_dir: str):
    """Save results as multiple CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Persons CSV
    persons_path = output_dir / 'persons.csv'
    with open(persons_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'person_id', 'x1', 'y1', 'x2', 'y2', 
                         'confidence', 'corner', 'guard', 'stance', 'lead_hand'])
        
        for frame_data in results['analysis']:
            for person in frame_data.get('persons', []):
                box = person.get('box', [0, 0, 0, 0])
                writer.writerow([
                    frame_data['frame'],
                    person.get('id', 0),
                    box[0], box[1], box[2], box[3],
                    person.get('confidence', 0),
                    person.get('corner', ''),
                    person.get('guard', ''),
                    person.get('stance', ''),
                    person.get('lead_hand', ''),
                ])
    
    # Punches CSV
    punches_path = output_dir / 'punches.csv'
    with open(punches_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'attacker', 'hand', 'type', 'target', 'result', 'impact'])
        
        for frame_data in results['analysis']:
            punch = frame_data.get('actions', {}).get('punch', {})
            if punch.get('detected'):
                writer.writerow([
                    frame_data['frame'],
                    punch.get('attacker', ''),
                    punch.get('hand', ''),
                    punch.get('type', ''),
                    punch.get('target', ''),
                    punch.get('result', ''),
                    punch.get('impact', ''),
                ])
    
    # Defense CSV
    defense_path = output_dir / 'defenses.csv'
    with open(defense_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'defender', 'type', 'against', 'success'])
        
        for frame_data in results['analysis']:
            defense = frame_data.get('actions', {}).get('defense', {})
            if defense.get('detected'):
                writer.writerow([
                    frame_data['frame'],
                    defense.get('defender', ''),
                    defense.get('type', ''),
                    defense.get('against', ''),
                    defense.get('success', ''),
                ])
    
    # Footwork CSV
    footwork_path = output_dir / 'footwork.csv'
    with open(footwork_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'fighter', 'type', 'purpose', 'pressure'])
        
        for frame_data in results['analysis']:
            footwork = frame_data.get('actions', {}).get('footwork', {})
            if footwork.get('detected'):
                writer.writerow([
                    frame_data['frame'],
                    footwork.get('fighter', ''),
                    footwork.get('type', ''),
                    footwork.get('purpose', ''),
                    footwork.get('pressure', ''),
                ])
    
    # Clinch CSV
    clinch_path = output_dir / 'clinches.csv'
    with open(clinch_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'initiator', 'state', 'work_in_clinch', 'ref_action'])
        
        for frame_data in results['analysis']:
            clinch = frame_data.get('actions', {}).get('clinch', {})
            if clinch.get('detected'):
                writer.writerow([
                    frame_data['frame'],
                    clinch.get('initiator', ''),
                    clinch.get('state', ''),
                    clinch.get('work_in_clinch', ''),
                    clinch.get('ref_action', ''),
                ])
    
    print(f"CSV files saved to: {output_dir}")


def _group_action_events(
    results: Dict,
    action_type: str,
    key_attrs: List[str],
    gap_threshold: int = 12,
    min_event_duration: int = 5
) -> List[Dict]:
    """
    Group consecutive action detections into discrete events using gap-based grouping.
    
    This prevents over-counting where a single punch is detected across multiple frames
    due to the sliding temporal window (typically 16 frames). Uses gap detection as
    the primary method, with attribute changes only considered if they persist.
    
    Threshold rationale (window_size=16 at ~30fps):
    - gap_threshold=12: ~0.75 * window_size; allows some dropped frames within an
      action while still splitting distinct actions that are close together.
    - min_event_duration=5: Filters spurious 1-4 frame noise; real actions span
      multiple frames due to the sliding window overlap.
    - attr_persist=15: Attribute changes must persist 15+ frames to trigger a split;
      filters noisy predictions that flicker between values.
    
    Args:
        results: Analysis results with frame-by-frame data
        action_type: Type of action (punch, defense, footwork, clinch)
        key_attrs: Attributes to track (for stats, not grouping)
        gap_threshold: Minimum frames gap to start new event (~window_size * 0.75)
        min_event_duration: Minimum frames for an event to be valid (filters noise)
        
    Returns:
        List of event dicts, each representing a unique action
    """
    events = []
    current_event = None
    last_detected_frame = -999  # Track frame gaps
    attribute_change_frames = {}  # Track when attributes change
    
    for frame_idx, frame_data in enumerate(results['analysis']):
        actions = frame_data.get('actions', {})
        action = actions.get(action_type, {})
        
        if not action.get('detected'):
            # No detection - check if we should finalize current event
            if current_event is not None:
                # If gap is large enough, finalize event
                if frame_idx - last_detected_frame > gap_threshold:
                    # Only add if event lasted long enough (filters noise)
                    if current_event['end_frame'] - current_event['start_frame'] >= min_event_duration:
                        events.append(current_event)
                    current_event = None
            continue
        
        # Get key attributes for this detection
        current_attrs = {attr: action.get(attr, 'unknown') for attr in key_attrs}
        confidence = action.get(f'{key_attrs[0]}_confidence', 0.5) if key_attrs else 0.5
        
        # Determine if this is a new event
        is_new_event = False
        
        if current_event is None:
            # First detection - start new event
            is_new_event = True
        elif frame_idx - last_detected_frame > gap_threshold:
            # Large gap detected - definitely a new event
            is_new_event = True
        else:
            # Check if key attributes changed AND the change persists
            # (This handles fast combos while ignoring single-frame flicker)
            for attr in key_attrs:
                if current_attrs[attr] != current_event.get(attr):
                    # Attribute changed - track when it changed
                    change_key = f"{attr}_{current_attrs[attr]}"
                    if change_key not in attribute_change_frames:
                        attribute_change_frames[change_key] = frame_idx
                    
                    # Only split if change has persisted for 15+ frames (filters attribute flicker)
                    # Higher threshold reduces over-counting from noisy attribute predictions
                    if frame_idx - attribute_change_frames[change_key] >= 15:
                        is_new_event = True
                        break
        
        if is_new_event:
            # Finalize previous event
            if current_event is not None:
                # Only add if event lasted long enough
                if current_event['end_frame'] - current_event['start_frame'] >= min_event_duration:
                    events.append(current_event)
            
            # Start new event
            current_event = {
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'max_confidence': confidence,
                'best_frame': frame_idx,
                **current_attrs,  # Store all key attributes
                '_full_action': action,  # Store full action data for best frame
            }
            # Reset attribute change tracking for new event
            attribute_change_frames = {}
        else:
            # Continue current event
            current_event['end_frame'] = frame_idx
            if confidence > current_event.get('max_confidence', 0):
                current_event['max_confidence'] = confidence
                current_event['best_frame'] = frame_idx
                current_event['_full_action'] = action
        
        last_detected_frame = frame_idx
    
    # Don't forget the last event
    if current_event is not None:
        if current_event['end_frame'] - current_event['start_frame'] >= min_event_duration:
            events.append(current_event)
    
    return events


def compute_summary(
    results: Dict,
    action_confidence: float = 0.6,
    min_separation: int = 3,
    gap_threshold: Optional[int] = None,
) -> Dict:
    """
    Compute summary statistics from analysis results.
    
    Uses compute_all_events() - the SINGLE SOURCE OF TRUTH for event counting.
    This ensures summary.json matches the annotated video exactly.
    
    Output (summary.json) includes:
    - actions: aggregate counts and breakdowns (punch, defense, footwork, clinch)
    - events: list of discrete events with frame, type, fighter, text, confidence, action.
      UIs and reports must use this list (and actions) from summary.json; they must
      NOT recompute events from analysis.json.
    
    Args:
        results: Analysis results with frame-by-frame data
        action_confidence: Minimum confidence threshold for events (default 0.6)
        min_separation: Minimum frames between events from same fighter (default 3)
        gap_threshold: When not None, passed to compute_all_events to use gap-based
            grouping instead of peak detection for non-defense types. Default None
            preserves existing behaviour.
    
    Returns:
        Summary dict suitable for JSON serialization (summary.json).
    """
    summary = {
        'video': results['video'],
        'total_frames': results['total_frames'],
        'fps': results['fps'],
        'duration_seconds': results['total_frames'] / results['fps'] if results['fps'] > 0 else 0,
        
        'detection': {
            'total_person_detections': 0,
            'avg_persons_per_frame': 0,
            'frames_with_detections': 0,
        },
        
        'fighters': {
            'red': {'frames_visible': 0, 'guards': Counter(), 'stances': Counter()},
            'blue': {'frames_visible': 0, 'guards': Counter(), 'stances': Counter()},
        },
        
        # Actions structure built from config (KEY_ATTRIBUTES_MAP + FIGHTER_FIELDS_MAP).
        # Each action type gets 'count' and 'by_<attr>' Counters for each key_attribute
        # plus 'by_<fighter_field>' if the type has a fighter.
        'actions': {}
    }
    
    # Build the breakdown attributes list per action type and initialize structure.
    # This replaces the old hardcoded per-type dicts with a config-driven structure.
    breakdown_attrs_map = {}
    for at in ACTION_TYPES:
        attrs = list(KEY_ATTRIBUTES_MAP.get(at, []))
        ff = FIGHTER_FIELDS_MAP.get(at)
        if ff and ff not in attrs:
            attrs.append(ff)
        breakdown_attrs_map[at] = attrs
        # Initialize: count + by_<attr> Counter for each breakdown attribute
        entry = {'count': 0}
        for attr in attrs:
            entry[f'by_{attr}'] = Counter()
        summary['actions'][at] = entry
    
    # =========================================================================
    # PERSON/DETECTION STATS (per-frame counting)
    # =========================================================================
    for frame_data in results['analysis']:
        persons = frame_data.get('persons', [])
        
        # Detection stats
        if persons:
            summary['detection']['frames_with_detections'] += 1
            summary['detection']['total_person_detections'] += len(persons)
        
        # Per-fighter stats
        for person in persons:
            corner = person.get('corner', 'unknown')
            if corner in ['red', 'blue']:
                summary['fighters'][corner]['frames_visible'] += 1
                summary['fighters'][corner]['guards'][person.get('guard', 'unknown')] += 1
                summary['fighters'][corner]['stances'][person.get('stance', 'unknown')] += 1
    
    # =========================================================================
    # ACTION STATS - uses compute_all_events() (single source of truth)
    # =========================================================================
    
    # Get all events using the shared function
    all_events = compute_all_events(
        results,
        action_confidence=action_confidence,
        min_separation=min_separation,
        window_size=10,
        gap_threshold=gap_threshold,
    )
    
    # Aggregate stats from events (config-driven, no per-type if/elif)
    for event in all_events:
        action_type = event['type']
        action = event['action']
        
        if action_type not in summary['actions']:
            continue  # Skip unknown types (safety)
        
        summary['actions'][action_type]['count'] += 1
        for attr in breakdown_attrs_map.get(action_type, []):
            breakdown_key = f'by_{attr}'
            if breakdown_key in summary['actions'][action_type]:
                # Use str() so Counter key is always hashable (model may output non-string)
                val = action.get(attr, 'unknown')
                summary['actions'][action_type][breakdown_key][str(val) if val is not None else 'unknown'] += 1
    
    # =========================================================================
    # EVENTS LIST - for frontend timeline (no recomputation needed)
    # =========================================================================
    
    # Include computed events for frontend to consume directly
    # This is the SINGLE SOURCE OF TRUTH - frontend should NOT recompute
    summary['events'] = []
    for event in all_events:
        action = event['action']
        action_type = event['type']
        
        # Build display text from config (same helper as _detect_and_log_events)
        text = _format_event_text(action_type, action)
        
        summary['events'].append({
            'frame': event['frame'],
            'type': action_type,
            'fighter': event['fighter'],
            'text': text,
            'confidence': event['confidence'],
            'action': action,
        })
    
    # =========================================================================
    # FINALIZE
    # =========================================================================
    
    # Calculate averages
    if summary['total_frames'] > 0:
        summary['detection']['avg_persons_per_frame'] = (
            summary['detection']['total_person_detections'] / summary['total_frames']
        )
    
    # Convert Counters to dicts for JSON serialization
    for corner in ['red', 'blue']:
        summary['fighters'][corner]['guards'] = dict(summary['fighters'][corner]['guards'])
        summary['fighters'][corner]['stances'] = dict(summary['fighters'][corner]['stances'])
    
    for action in summary['actions'].values():
        for key in action:
            if isinstance(action[key], Counter):
                action[key] = dict(action[key])
    
    return summary


def print_summary(summary: Dict):
    """Print summary to console."""
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nVideo: {summary['video']}")
    print(f"Duration: {summary['duration_seconds']:.1f}s ({summary['total_frames']} frames @ {summary['fps']:.1f} FPS)")
    
    print(f"\n--- Detections ---")
    print(f"Total person detections: {summary['detection']['total_person_detections']}")
    print(f"Avg persons/frame: {summary['detection']['avg_persons_per_frame']:.2f}")
    print(f"Frames with detections: {summary['detection']['frames_with_detections']}")
    
    print(f"\n--- Fighters ---")
    for corner in ['red', 'blue']:
        fighter = summary['fighters'][corner]
        print(f"  {corner.upper()}:")
        print(f"    Frames visible: {fighter['frames_visible']}")
        if fighter['guards']:
            print(f"    Guards: {fighter['guards']}")
        if fighter['stances']:
            print(f"    Stances: {fighter['stances']}")
    
    print(f"\n--- Actions ---")
    for action_name, action_data in summary['actions'].items():
        if action_data['count'] > 0:
            print(f"  {action_name.upper()}: {action_data['count']} detected")
            for key, value in action_data.items():
                if key != 'count' and isinstance(value, dict) and value:
                    label = key.replace('by_', '').replace('_', ' ').title()
                    print(f"    {label}: {value}")
    
    print(f"{'='*60}\n")


def run_main(args, backend='pytorch'):
    """Run the full batch video analysis pipeline.
    
    Args:
        args: Parsed argparse namespace with video, output, confidence, etc.
        backend: 'pytorch' or 'onnx'
    """
    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    
    if args.output is None:
        args.output = Path('results') / args.video.stem
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VISTRIKE Video Analysis ({backend.upper()} backend)")
    print(f"{'='*60}")
    
    analyzer = BoxingAnalyzer(
        models_dir=str(args.models),
        device=args.device,
        confidence=args.confidence,
        attr_confidence=args.attr_confidence,
        action_confidence=args.action_confidence,
        min_event_separation=args.min_separation,
        assign_single_fighter=args.assign_single_fighter,
        side_confidence_min=args.side_confidence_min,
        stable_side_frames=args.stable_side_frames,
        use_gap_grouping=args.use_gap_grouping,
        backend=backend,
    )
    
    start_time = time.time()
    results = analyzer.analyze_video(str(args.video))
    analysis_time = time.time() - start_time
    
    print(f"\nAnalysis completed in {analysis_time:.1f}s")
    print(f"  Speed: {results['total_frames'] / analysis_time:.1f} FPS")
    
    if args.output_format == 'json':
        save_json(results, args.output / 'analysis.json')
    else:
        save_csv(results, args.output / 'csv')
    
    summary = compute_summary(
        results,
        action_confidence=args.action_confidence,
        min_separation=args.min_separation,
        gap_threshold=analyzer.get_gap_threshold(),
    )
    save_json(summary, args.output / 'summary.json')
    print_summary(summary)
    
    if args.save_video.lower() == 'true':
        output_video = args.output / f"{args.video.stem}_annotated.mp4"
        analyzer.create_annotated_video(str(args.video), results, str(output_video))
    
    if args.save_frames:
        frames_dir = args.output / 'frames'
        analyzer.save_annotated_frames(str(args.video), results, str(frames_dir))
    
    print(f"\nAll outputs saved to: {args.output}")
    print(f"{'='*60}\n")
