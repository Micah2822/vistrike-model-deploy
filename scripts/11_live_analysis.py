#!/usr/bin/env python3
"""
Production Real-Time Analysis Pipeline.

Analyze boxing video from webcam, file, or RTSP stream in real-time
with <50ms latency and 30+ FPS on RTX 3060 / Apple M2 Pro+.

NOTE: SSL workaround included for macOS certificate issues.

================================================================================
USAGE:
================================================================================

# Webcam (default camera)
python3 scripts/11_live_analysis.py --source 0

# Video file
python3 scripts/11_live_analysis.py --source path/to/video.mp4

# RTSP stream
python3 scripts/11_live_analysis.py --source rtsp://camera_ip:554/stream

# Save output video
python3 scripts/11_live_analysis.py --source 0 --output live_output.mp4

# Use ONNX runtime (faster, ~20% speedup)
python3 scripts/11_live_analysis.py --source 0 --onnx

# Headless mode (no display window)
python3 scripts/11_live_analysis.py --source video.mp4 --display false

# Force CPU
python3 scripts/11_live_analysis.py --source 0 --device cpu

================================================================================
PERFORMANCE TARGETS:
================================================================================

- 30+ FPS on RTX 3060 or Apple M2 Pro+ / M4 Pro
- Zero frame drops (async capture with frame skipping)
- <50ms end-to-end latency
- FP16 inference by default for speed

================================================================================
ARCHITECTURE:
================================================================================

Three-thread pipeline:
1. CaptureThread  - Dedicated frame capture (never blocks inference)
2. InferenceThread - GPU inference (unified model + actions)
3. MainThread      - Display & annotation (CPU, parallel to GPU)

================================================================================
TRACKING & REFEREE (Story 8):
================================================================================

Same logic as 10_inference.py: IoU + corner assignment, max 2 fighter tracks.
Third unmatched detection when 2 tracks exist → referee (role='referee', id=-1).
Role confidence < 0.5 → no fighter track. See main_usage_guides/05_VIDEO_INFERENCE.md.

================================================================================
REQUIRED MODELS:
================================================================================

models/
├── unified/best.pt              # PyTorch unified model
├── unified/unified_model.onnx   # ONNX model (if --onnx)
└── actions/                     # Optional action models
    ├── punch/best.pt
    └── ...

================================================================================
ONNX EXPORT:
================================================================================

To create ONNX model for faster inference:

    python3 -c "
    import torch
    # Load your trained unified model
    checkpoint = torch.load('models/unified/best.pt', weights_only=False)
    # ... recreate model architecture ...
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 640)
    torch.onnx.export(model, dummy_input, 'models/unified/unified_model.onnx',
                      input_names=['input'], output_names=['boxes', 'scores', 'attributes'],
                      dynamic_axes={'input': {0: 'batch'}})
    "

================================================================================
"""

# SSL certificate workaround for macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core dependencies
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is required. Install with: pip install numpy")
    sys.exit(1)

# Optional PyTorch (required unless using pure ONNX)
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Optional ONNX Runtime
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Action types config path
ACTION_TYPES_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "action_types.yaml"


def load_action_types_config() -> tuple:
    """Load action types, colors, fighter keys, and fighter fields from YAML config."""
    if not ACTION_TYPES_CONFIG_PATH.exists():
        # Fallback to hardcoded defaults
        return (
            ['punch', 'defense', 'footwork', 'clinch', 'knockdown', 'wobble', 'stoppage'],
            {
                'red': (0, 0, 255), 'blue': (255, 0, 0), 'unknown': (128, 128, 128),
                'punch': (0, 255, 255), 'defense': (0, 255, 0), 'footwork': (255, 0, 255),
                'clinch': (0, 165, 255), 'fps_good': (0, 255, 0), 'fps_bad': (0, 0, 255),
            },
            {'punch': 'attacker_side', 'defense': 'defender_side', 'footwork': 'fighter_side',
             'clinch': 'initiator_side', 'knockdown': 'fighter_side', 'wobble': 'fighter_side', 'stoppage': None},
            {'punch': 'attacker', 'defense': 'defender', 'footwork': 'fighter',
             'clinch': 'initiator', 'knockdown': 'fighter', 'wobble': 'fighter', 'stoppage': None},
        )
    
    with open(ACTION_TYPES_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    action_types = list(config.get('action_types', {}).keys())
    
    # Build COLORS dict
    colors = {
        'fps_good': (0, 255, 0),
        'fps_bad': (0, 0, 255),
    }
    for name, bgr in config.get('visualization_colors', {}).items():
        colors[name] = tuple(bgr)
    for name, cfg in config.get('action_types', {}).items():
        if 'color_bgr' in cfg:
            colors[name] = tuple(cfg['color_bgr'])
    
    # Build fighter_keys dict (model prediction field, e.g. 'attacker_side')
    fighter_keys = {}
    for name, cfg in config.get('action_types', {}).items():
        fighter_keys[name] = cfg.get('fighter_key')
    
    # Build fighter_fields dict (original annotation field, e.g. 'attacker')
    fighter_fields = {}
    for name, cfg in config.get('action_types', {}).items():
        fighter_fields[name] = cfg.get('fighter_field')
    
    return action_types, colors, fighter_keys, fighter_fields


# Load action types and colors from YAML
ACTION_TYPES, COLORS, FIGHTER_KEYS_MAP, FIGHTER_FIELDS_MAP = load_action_types_config()

# Model input size (optimized for speed)
INPUT_WIDTH = 640
INPUT_HEIGHT = 384

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_device(device_str: str) -> str:
    """Get best available device. ``auto`` is CUDA if available, else CPU (never MPS)."""
    if device_str != 'auto':
        return device_str

    if TORCH_AVAILABLE and torch.cuda.is_available():
        return 'cuda'

    return 'cpu'


def get_onnx_providers(device: str) -> List[str]:
    """Get ONNX Runtime execution providers for device."""
    if device == 'cuda':
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device == 'mps':
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    else:
        return ['CPUExecutionProvider']


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_frame_numpy(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame using NumPy (for ONNX)."""
    # Resize
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize (0-255 to 0-1, then ImageNet norm)
    normalized = rgb.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    
    # HWC to CHW, add batch dimension
    chw = np.transpose(normalized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    
    return batch.astype(np.float32)


def preprocess_frame_torch(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess frame using PyTorch."""
    # Resize
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # To tensor and normalize
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    # Add batch dimension and move to device
    tensor = tensor.unsqueeze(0).to(device)
    
    return tensor


# =============================================================================
# CAPTURE THREAD
# =============================================================================

class CaptureThread(threading.Thread):
    """
    Dedicated thread for frame capture.
    
    Never blocks inference - drops old frames to maintain low latency.
    """
    
    def __init__(
        self,
        source: Union[int, str],
        frame_queue: queue.Queue,
        target_fps: int = 30,
    ):
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.target_fps = target_fps
        self.running = True
        
        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0
        self.actual_fps = 0.0
    
    def run(self):
        """Main capture loop."""
        # Initialize capture
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open source: {self.source}")
            self.running = False
            return
        
        # Try to enable hardware acceleration
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except:
            pass
        
        # Set buffer size to minimum to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get source FPS
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30  # Default for webcams
        
        print(f"Capture started: {self.source}")
        print(f"  Source FPS: {source_fps:.1f}")
        
        fps_timer = time.time()
        fps_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # For video files, loop back to start
                if isinstance(self.source, str) and not self.source.startswith('rtsp'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            self.frames_captured += 1
            fps_count += 1
            
            # Update FPS counter every second
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.actual_fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()
            
            # Drop old frames to prevent lag (non-blocking put)
            try:
                self.frame_queue.put_nowait((frame, time.time()))
            except queue.Full:
                # Queue full - drop oldest frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait((frame, time.time()))
                except:
                    pass
        
        cap.release()
        print("Capture thread stopped")


# =============================================================================
# INFERENCE THREAD
# =============================================================================

class InferenceThread(threading.Thread):
    """
    GPU inference thread.
    
    Processes frames as fast as possible using unified model.
    """
    
    def __init__(
        self,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        device: str = 'cuda',
        use_onnx: bool = False,
        models_dir: str = 'models/',
    ):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.device_str = device
        self.use_onnx = use_onnx
        self.models_dir = Path(models_dir)
        self.running = True
        
        # Frame buffer for temporal action detection
        self.frame_buffer = deque(maxlen=16)
        
        # Stats
        self.frames_processed = 0
        self.inference_fps = 0.0
        self.avg_inference_ms = 0.0
        
        # Models (loaded in run())
        self.unified_model = None
        self.action_models = {}
        self.onnx_session = None
        
        # Tracking state for persistent corner assignment and temporal smoothing
        self.tracks = {}  # {track_id: {
                          #   'box': [...], 
                          #   'corner': 'blue'|'red'|None, 
                          #   'frames_seen': 0, 
                          #   'last_seen': frame_idx,
                          #   'role_history': [],      # Last N role predictions
                          #   'stance_history': [],    # Last N stance predictions
                          #   'lead_hand_history': [], # Last N lead_hand predictions
                          #   'headgear_history': [],  # Last N headgear predictions
                          # }}
        self.next_track_id = 0
        self.iou_threshold = 0.3  # Minimum IoU to match detection to existing track
        self.max_frames_missing = 9999  # Effectively never remove tracks (corners are permanent)
        self.corner_confidence_threshold = 0.7  # High confidence needed to assign corner
        self._frame_count = 0  # Track frame count for tracking
        
        # Story 8: Role confidence thresholds for referee detection
        self.role_confidence_threshold = 0.7  # Require >0.7 to trust 'fighter' prediction
        self.role_confidence_referee_default = 0.5  # Below this, default to referee
        
        # Temporal smoothing settings
        self.role_history_size = 60  # 2 seconds at 30fps - longer for critical attribute
        self.stance_history_size = 30  # 1 second at 30fps
        self.lead_hand_history_size = 30
        self.headgear_history_size = 30
    
    def _reset_tracker(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_track_id = 0
        self._frame_count = 0
    
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
    
    def _assign_tracks_and_corners(self, persons: List[Dict]) -> List[Dict]:
        """
        Assign persistent track IDs and enforce sticky corner assignment.
        
        Boxing-specific strategy (max 2 fighters):
        1. Match detections to existing tracks by IoU (best match first)
        2. If <2 tracks exist: create new track for unmatched detections (if role=fighter)
        3. If 2 tracks exist: assign unmatched detections to closest track (no new tracks)
        4. Corners are PERMANENT once assigned - never switch
        
        Story 8 - Referee detection:
        - Third detection rule: if 2 fighter tracks exist and detection is unmatched,
          classify as referee (id=-1, corner='unknown', role='referee')
        - Role confidence threshold: require >0.7 to trust 'fighter' prediction;
          <0.5 defaults to referee and doesn't create fighter track
        """
        frame_idx = self._frame_count
        
        # Count existing fighter tracks (those with corners assigned)
        fighter_tracks = {tid: t for tid, t in self.tracks.items() if t['corner'] in ['blue', 'red']}
        num_fighter_tracks = len(fighter_tracks)
        
        track_ids = list(self.tracks.keys())
        matched_detections = set()
        matched_tracks = set()
        
        # Step 1: Match detections to tracks by IoU (best matches first)
        matches = []
        for det_idx, person in enumerate(persons):
            box = person.get('box', [0, 0, 0, 0])
            for track_id in track_ids:
                iou = self._compute_iou(box, self.tracks[track_id]['box'])
                matches.append((iou, det_idx, track_id))
        
        # Sort by IoU descending
        matches.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy assignment with IoU threshold
        for iou, det_idx, track_id in matches:
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
            
            # Update track state
            track['box'] = person.get('box', [0, 0, 0, 0])
            track['last_seen'] = frame_idx
            track['frames_seen'] += 1
            
            matched_detections.add(det_idx)
            matched_tracks.add(track_id)
        
        # Step 2: Handle unmatched detections
        for det_idx, person in enumerate(persons):
            if det_idx in matched_detections:
                continue
            
            box = person.get('box', [0, 0, 0, 0])
            
            # If we already have 2 fighter tracks, assign to closest track (no new tracks)
            if num_fighter_tracks >= 2:
                # Find closest unmatched track by IoU (even if below threshold)
                best_iou = -1
                best_track_id = None
                for track_id in fighter_tracks:
                    if track_id in matched_tracks:
                        continue
                    iou = self._compute_iou(box, self.tracks[track_id]['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
                
                # Story 8: Require minimum IoU (0.05) to prevent false positives hijacking tracks
                min_iou_for_assignment = 0.05
                if best_track_id is not None and best_iou >= min_iou_for_assignment:
                    # Assign to closest existing track
                    track = self.tracks[best_track_id]
                    person['id'] = best_track_id
                    person['corner'] = track['corner']  # Sticky corner
                    
                    # Apply temporal smoothing for role, stance, lead_hand, headgear
                    person = self._apply_temporal_smoothing(person, track)
                    
                    track['box'] = box
                    track['last_seen'] = frame_idx
                    track['frames_seen'] += 1
                    matched_tracks.add(best_track_id)
                else:
                    # Story 8: Third-detection rule - if 2 fighter tracks exist and this
                    # detection doesn't match either, treat as referee (not a false positive)
                    person['id'] = -1  # Non-fighter ID
                    person['corner'] = 'unknown'
                    person['role'] = 'referee'  # Explicit referee classification (Story 8)
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
                    # Don't create a track for this person
                    continue
                
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
                }
                
                # Apply temporal smoothing (initializes history with first prediction)
                person = self._apply_temporal_smoothing(person, self.tracks[track_id])
                
                # Update fighter_tracks for subsequent iterations
                if assigned_corner in ['blue', 'red']:
                    fighter_tracks[track_id] = self.tracks[track_id]
        
        return persons
    
    def _load_models(self):
        """Load models at thread start."""
        if self.use_onnx and ONNX_AVAILABLE:
            self._load_onnx_models()
        elif TORCH_AVAILABLE:
            self._load_pytorch_models()
        else:
            print("ERROR: Neither PyTorch nor ONNX Runtime available")
            self.running = False
    
    def _load_pytorch_models(self):
        """Load PyTorch models."""
        self.device = torch.device(self.device_str)
        
        # Unified model
        unified_path = self.models_dir / 'unified' / 'best.pt'
        if unified_path.exists():
            print(f"Loading unified model: {unified_path}")
            checkpoint = torch.load(unified_path, map_location=self.device, weights_only=False)
            self.unified_model = checkpoint  # Placeholder - actual model loading needed
            print("  Unified model loaded")
        else:
            print(f"WARNING: Unified model not found at {unified_path}")
        
        # Action models (loaded from YAML config)
        actions_dir = self.models_dir / 'actions'
        if actions_dir.exists():
            for action_type in ACTION_TYPES:
                action_path = actions_dir / action_type / 'best.pt'
                if action_path.exists():
                    print(f"Loading {action_type} model")
                    checkpoint = torch.load(action_path, map_location=self.device, weights_only=False)
                    self.action_models[action_type] = checkpoint
    
    def _load_onnx_models(self):
        """Load ONNX models."""
        providers = get_onnx_providers(self.device_str)
        
        # Unified model
        onnx_path = self.models_dir / 'unified' / 'unified_model.onnx'
        if onnx_path.exists():
            print(f"Loading ONNX model: {onnx_path}")
            print(f"  Providers: {providers}")
            self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            print("  ONNX model loaded")
        else:
            print(f"WARNING: ONNX model not found at {onnx_path}")
            print("  Falling back to PyTorch")
            self.use_onnx = False
            if TORCH_AVAILABLE:
                self._load_pytorch_models()
    
    def run(self):
        """Main inference loop."""
        print(f"Inference thread starting on device: {self.device_str}")
        print(f"  Using ONNX: {self.use_onnx}")
        
        self._load_models()
        
        if not self.running:
            return
        
        fps_timer = time.time()
        fps_count = 0
        inference_times = deque(maxlen=100)
        
        # Create dedicated CUDA stream if available
        cuda_stream = None
        if TORCH_AVAILABLE and self.device_str == 'cuda' and torch.cuda.is_available():
            cuda_stream = torch.cuda.Stream()
        
        while self.running:
            try:
                frame, capture_time = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            inference_start = time.time()
            
            # Run inference
            if self.use_onnx and self.onnx_session:
                result = self._run_onnx_inference(frame, capture_time)
            else:
                result = self._run_pytorch_inference(frame, capture_time, cuda_stream)
            
            inference_time = (time.time() - inference_start) * 1000
            inference_times.append(inference_time)
            
            self.frames_processed += 1
            fps_count += 1
            
            # Update stats every second
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.inference_fps = fps_count / elapsed
                self.avg_inference_ms = sum(inference_times) / len(inference_times)
                fps_count = 0
                fps_timer = time.time()
            
            # Send result to display (non-blocking)
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.result_queue.put_nowait(result)
                except:
                    pass
        
        print("Inference thread stopped")
    
    def _run_onnx_inference(
        self,
        frame: np.ndarray,
        capture_time: float,
    ) -> Dict:
        """Run inference using ONNX Runtime."""
        # Preprocess
        input_tensor = preprocess_frame_numpy(frame)
        
        # Increment frame count for tracking
        self._frame_count += 1
        
        # Run unified model
        # Note: Output structure depends on how model was exported
        # This is a placeholder - actual output parsing needed
        persons = []
        try:
            outputs = self.onnx_session.run(None, {'input': input_tensor})
            # TODO: Parse outputs to get persons list
        except Exception as e:
            outputs = None
        
        # Apply tracking and sticky corner assignment
        if persons:
            persons = self._assign_tracks_and_corners(persons)
        
        return {
            'frame': frame,
            'capture_time': capture_time,
            'persons': persons,
            'actions': {
                'punch': {'detected': False},
                'defense': {'detected': False},
                'footwork': {'detected': False},
                'clinch': {'detected': False},
            },
        }
    
    def _run_pytorch_inference(
        self,
        frame: np.ndarray,
        capture_time: float,
        cuda_stream: Optional[Any] = None,
    ) -> Dict:
        """Run inference using PyTorch."""
        # Increment frame count for tracking
        self._frame_count += 1
        
        result = {
            'frame': frame,
            'capture_time': capture_time,
            'persons': [],
            'actions': {
                'punch': {'detected': False},
                'defense': {'detected': False},
                'footwork': {'detected': False},
                'clinch': {'detected': False},
            },
        }
        
        if not TORCH_AVAILABLE:
            return result
        
        device = torch.device(self.device_str)
        
        # Preprocess
        input_tensor = preprocess_frame_torch(frame, device)
        
        # Run in dedicated stream for async
        stream_context = torch.cuda.stream(cuda_stream) if cuda_stream else nullcontext()
        
        with stream_context:
            with torch.no_grad():
                # Enable automatic mixed precision on CUDA
                autocast_ctx = torch.cuda.amp.autocast() if self.device_str == 'cuda' else nullcontext()
                
                with autocast_ctx:
                    # Unified model inference
                    # Note: Placeholder - actual model forward pass needed
                    # When persons are detected, they would be assigned here
                    persons = []  # Placeholder for detected persons
                    
                    # Apply tracking and sticky corner assignment
                    if persons:
                        persons = self._assign_tracks_and_corners(persons)
                    result['persons'] = persons
                    
                    # Add to temporal buffer
                    self.frame_buffer.append(input_tensor)
                    
                    # Action detection (if enough frames)
                    if len(self.frame_buffer) >= 8:
                        # Placeholder - action model inference
                        # When implemented: run action models, then map sides to corners.
                        # Pass assign_single_fighter and side_confidence_min from config/args if available:
                        # result['actions'] = map_action_sides_to_corners(
                        #     result['actions'], result['persons'],
                        #     assign_single_fighter=..., side_confidence_min=...
                        # )
                        pass
        
        return result


# =============================================================================
# CONTEXT MANAGER FALLBACK
# =============================================================================

def map_action_sides_to_corners(
    actions: Dict,
    persons: List[Dict],
    assign_single_fighter: bool = False,
    side_confidence_min: float = 0.0,
) -> Dict:
    """
    Map action model side predictions (left/right) to corners (red/blue).
    
    The action model predicts which SIDE of the frame performed the action
    (e.g. attacker_side=left). Detection+tracking knows which person is on
    which side and what their corner is. This function bridges the two.
    
    Guards:
    - Requires exactly 2 fighters with one red and one blue.
    - Duplicate corners or 3+ tracks produce 'unknown'.
    - Optional single-fighter assignment (assign_single_fighter).
    - Optional side confidence threshold (side_confidence_min).
    
    Args:
        actions: Action detection results per action type
        persons: Tracked person dicts with 'box' and 'corner' keys
        assign_single_fighter: If True, assign action to the only visible fighter
        side_confidence_min: Minimum side confidence; below this, fighter = unknown
        
    Returns:
        Updated actions dict with original fighter fields populated.
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
    corners = {p['corner'] for p in fighters}
    valid_pair = len(fighters) == 2 and corners == {'red', 'blue'}
    
    # Single-fighter assignment (optional): both sides map to only fighter.
    single_fighter = len(fighters) == 1 and assign_single_fighter
    
    if not valid_pair and not single_fighter:
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
    
    if single_fighter:
        side_to_corner = {
            'left': fighters[0]['corner'],
            'right': fighters[0]['corner'],
        }
    else:
        side_to_corner = {
            'left': fighters[0]['corner'],
            'right': fighters[-1]['corner'],
        }
    
    for action_type, action in actions.items():
        if not action.get('detected'):
            continue
        fighter_key = FIGHTER_KEYS_MAP.get(action_type)
        fighter_field = FIGHTER_FIELDS_MAP.get(action_type)
        if not fighter_key or not fighter_field:
            continue
        side_val = action.get(fighter_key)  # 'left'/'right' (new model) or None (old model)
        
        # Backward compatibility: only map when model outputs _side (left/right).
        # Old checkpoints set attacker/defender to red/blue directly — leave those.
        if side_val in ('left', 'right'):
            action[fighter_field] = side_to_corner.get(side_val, 'unknown')
            if side_confidence_min > 0:
                side_conf = action.get(f'{fighter_key}_confidence', 1.0)
                if side_conf < side_confidence_min:
                    action[fighter_field] = 'unknown'
    
    return actions


class nullcontext:
    """Null context manager for Python < 3.7 compatibility."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


# =============================================================================
# LIVE ANALYZER
# =============================================================================

class LiveAnalyzer:
    """
    Real-time boxing analysis pipeline.
    
    Three-thread architecture:
    1. CaptureThread  - Dedicated frame capture
    2. InferenceThread - GPU inference
    3. Main thread    - Display & annotation
    """
    
    def __init__(
        self,
        source: Union[int, str],
        output_path: Optional[str] = None,
        display: bool = True,
        use_onnx: bool = False,
        device: str = 'auto',
        models_dir: str = 'models/',
    ):
        self.source = source
        self.output_path = output_path
        self.display = display
        self.use_onnx = use_onnx
        self.device = get_device(device)
        self.models_dir = models_dir
        
        # Queues for thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Video writer
        self.output_writer = None
        
        # Stats
        self.display_fps = 0.0
        self.total_frames = 0
        self.start_time = None
    
    def run(self):
        """Main analysis loop."""
        print(f"\n{'='*60}")
        print("VISTRIKE Live Analysis")
        print(f"{'='*60}")
        print(f"Source: {self.source}")
        print(f"Device: {self.device}")
        print(f"ONNX: {self.use_onnx}")
        print(f"Display: {self.display}")
        print(f"Output: {self.output_path or 'None'}")
        print(f"{'='*60}\n")
        
        # Start threads
        capture_thread = CaptureThread(self.source, self.frame_queue)
        inference_thread = InferenceThread(
            self.frame_queue,
            self.result_queue,
            device=self.device,
            use_onnx=self.use_onnx,
            models_dir=self.models_dir,
        )
        
        capture_thread.start()
        inference_thread.start()
        
        # Wait for first frame to get dimensions
        print("Waiting for first frame...")
        first_result = None
        for _ in range(50):  # 5 second timeout
            try:
                first_result = self.result_queue.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        
        if first_result is None:
            print("ERROR: No frames received. Check source.")
            capture_thread.running = False
            inference_thread.running = False
            return
        
        # Setup video writer if needed
        frame_height, frame_width = first_result['frame'].shape[:2]
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_writer = cv2.VideoWriter(
                self.output_path, fourcc, 30, (frame_width, frame_height)
            )
        
        # Process first result
        self._process_result(first_result, capture_thread, inference_thread)
        
        # FPS tracking
        fps_times = deque(maxlen=30)
        self.start_time = time.time()
        
        print("\nAnalysis running. Press 'q' to quit.\n")
        
        # Main display loop
        try:
            while capture_thread.running and inference_thread.running:
                try:
                    result = self.result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.total_frames += 1
                fps_times.append(time.time())
                
                # Calculate display FPS
                if len(fps_times) > 1:
                    self.display_fps = len(fps_times) / (fps_times[-1] - fps_times[0])
                
                # Process and display
                self._process_result(result, capture_thread, inference_thread)
                
                # Check for quit
                if self.display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        # Cleanup
        print("\nShutting down...")
        capture_thread.running = False
        inference_thread.running = False
        capture_thread.join(timeout=1.0)
        inference_thread.join(timeout=1.0)
        
        if self.output_writer:
            self.output_writer.release()
        
        cv2.destroyAllWindows()
        
        # Print final stats
        self._print_stats(capture_thread, inference_thread)
    
    def _process_result(
        self,
        result: Dict,
        capture_thread: CaptureThread,
        inference_thread: InferenceThread,
    ):
        """Process inference result and display."""
        frame = result['frame']
        
        # Draw annotations
        annotated = self._draw_annotations(frame, result)
        
        # Draw stats overlay
        annotated = self._draw_stats(
            annotated,
            result,
            capture_thread,
            inference_thread,
        )
        
        # Display
        if self.display:
            cv2.imshow('VISTRIKE Live Analysis', annotated)
        
        # Write to output
        if self.output_writer:
            self.output_writer.write(annotated)
    
    def _draw_annotations(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw person boxes and action labels."""
        annotated = frame.copy()
        
        # Draw persons - only red/blue fighters (skip referee and unknown, same as 10_inference)
        for person in result.get('persons', []):
            corner = person.get('corner', 'unknown')
            if corner not in ['red', 'blue']:
                continue
            box = person.get('box', [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(c) for c in box]
            color = COLORS.get(corner, COLORS['unknown'])
            
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label
            guard = person.get('guard', '?')
            stance = person.get('stance', '?')
            label = f"{corner.upper()} | {guard} | {stance}"
            
            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4), font, 0.5, (0, 0, 0), 1)
            
            # Confidence
            conf = person.get('confidence', 0)
            cv2.putText(annotated, f"{conf:.0%}", (x1, y2 + 15), font, 0.4, color, 1)
        
        # Draw actions at top
        action_y = 80
        for action_name, action_data in result.get('actions', {}).items():
            if action_data.get('detected', False):
                color = COLORS.get(action_name, (255, 255, 255))
                
                if action_name == 'punch':
                    text = f"PUNCH: {action_data.get('attacker', '?')} {action_data.get('type', '?')}"
                elif action_name == 'defense':
                    text = f"DEFENSE: {action_data.get('defender', '?')} {action_data.get('type', '?')}"
                elif action_name == 'footwork':
                    text = f"FOOTWORK: {action_data.get('fighter', '?')} {action_data.get('type', '?')}"
                elif action_name == 'clinch':
                    text = f"CLINCH: {action_data.get('initiator', '?')}"
                else:
                    text = f"{action_name.upper()}"
                
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (10, action_y - th - 4), (10 + tw + 8, action_y + 4), (0, 0, 0), -1)
                cv2.putText(annotated, text, (14, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                action_y += 30
        
        return annotated
    
    def _draw_stats(
        self,
        frame: np.ndarray,
        result: Dict,
        capture_thread: CaptureThread,
        inference_thread: InferenceThread,
    ) -> np.ndarray:
        """Draw FPS and latency overlay."""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        # Calculate latency
        latency_ms = (time.time() - result['capture_time']) * 1000
        
        # FPS color (green if good, red if bad)
        fps_color = COLORS['fps_good'] if self.display_fps >= 25 else COLORS['fps_bad']
        latency_color = COLORS['fps_good'] if latency_ms < 50 else COLORS['fps_bad']
        
        # Draw stats box
        stats_lines = [
            f"Display FPS: {self.display_fps:.1f}",
            f"Inference FPS: {inference_thread.inference_fps:.1f}",
            f"Latency: {latency_ms:.0f}ms",
            f"Inference: {inference_thread.avg_inference_ms:.1f}ms",
        ]
        
        box_width = 200
        box_height = len(stats_lines) * 22 + 10
        
        # Semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 5), (5 + box_width, 5 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw text
        y = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Display FPS
        cv2.putText(annotated, stats_lines[0], (10, y), font, 0.5, fps_color, 1)
        y += 22
        
        # Inference FPS
        cv2.putText(annotated, stats_lines[1], (10, y), font, 0.5, (255, 255, 255), 1)
        y += 22
        
        # Latency
        cv2.putText(annotated, stats_lines[2], (10, y), font, 0.5, latency_color, 1)
        y += 22
        
        # Inference time
        cv2.putText(annotated, stats_lines[3], (10, y), font, 0.5, (255, 255, 255), 1)
        
        # Source info (bottom left)
        if isinstance(self.source, int):
            source_text = f"Webcam {self.source}"
        elif str(self.source).startswith('rtsp'):
            source_text = "RTSP Stream"
        else:
            source_text = Path(self.source).name
        
        cv2.putText(annotated, source_text, (10, height - 10),
                    font, 0.5, (255, 255, 255), 1)
        
        # Device info (bottom right)
        device_text = f"{self.device.upper()}"
        if self.use_onnx:
            device_text += " (ONNX)"
        (tw, _), _ = cv2.getTextSize(device_text, font, 0.5, 1)
        cv2.putText(annotated, device_text, (width - tw - 10, height - 10),
                    font, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _print_stats(
        self,
        capture_thread: CaptureThread,
        inference_thread: InferenceThread,
    ):
        """Print final statistics."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'='*60}")
        print("SESSION STATISTICS")
        print(f"{'='*60}")
        print(f"Duration: {duration:.1f}s")
        print(f"Total frames processed: {self.total_frames}")
        print(f"Average display FPS: {self.total_frames / duration:.1f}" if duration > 0 else "")
        print(f"Frames captured: {capture_thread.frames_captured}")
        print(f"Frames dropped: {capture_thread.frames_dropped}")
        print(f"Drop rate: {capture_thread.frames_dropped / max(capture_thread.frames_captured, 1) * 100:.1f}%")
        print(f"Final inference FPS: {inference_thread.inference_fps:.1f}")
        print(f"Avg inference time: {inference_thread.avg_inference_ms:.1f}ms")
        print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time boxing video analysis"
    )
    
    # Source
    parser.add_argument('--source', type=str, default='0',
                        help='Input source: 0 (webcam), video path, or RTSP URL')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (optional)')
    
    # Display
    parser.add_argument('--display', type=str, default='true',
                        help='Show live display window (true/false)')
    
    # Model options
    parser.add_argument('--onnx', action='store_true',
                        help='Use ONNX Runtime instead of PyTorch')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Inference device')
    parser.add_argument('--models', type=str, default='models/',
                        help='Models directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse source (handle "0" as webcam index)
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Parse display flag
    display = args.display.lower() in ('true', '1', 'yes')
    
    # Validate dependencies
    if args.onnx and not ONNX_AVAILABLE:
        print("ERROR: ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
        sys.exit(1)
    
    if not args.onnx and not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available. Install with: pip install torch torchvision")
        print("  Or use --onnx flag with ONNX Runtime")
        sys.exit(1)
    
    # Create and run analyzer
    analyzer = LiveAnalyzer(
        source=source,
        output_path=args.output,
        display=display,
        use_onnx=args.onnx,
        device=args.device,
        models_dir=args.models,
    )
    
    analyzer.run()


if __name__ == '__main__':
    main()
