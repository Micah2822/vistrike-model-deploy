# VISTRIKE Model Testing UI

A simple web-based UI for testing VISTRIKE models on video files. Upload videos, configure inference settings, and view annotated outputs with detailed statistics.

---

## Features

- **Video Upload**: Drag-and-drop or click to upload video files
- **Configuration Options**:
  - Detection confidence threshold
  - Attribute confidence threshold
  - Device selection (CPU, CUDA, MPS, Auto)
  - Save annotated video toggle
- **Annotated Video Output**: View video with bounding boxes, attributes, and action tags
- **Attributes & Actions Display**: Clear visualization of detected attributes and actions
- **Detailed Statistics Report**: Comprehensive breakdown including:
  - Total punches thrown and landed
  - Punch types breakdown
  - Defense statistics
  - Footwork analysis
  - Clinch events
  - Fighter-specific stats (guards, stances)
  - Interactive pie charts for visual data representation

---

## Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install flask flask-cors
   ```

2. **Trained Models**: Ensure you have trained models in the **VISTRIKE-AI-Official** project root `models/` directory (testing-ui runs inference from the parent repo):
   ```
   VISTRIKE-AI-Official/models/
   ├── unified/best.pt          # OR unified_mps/best.pt
   └── actions/                  # Optional
       ├── punch/best.pt
       ├── defense/best.pt
       ├── footwork/best.pt
       └── clinch/best.pt
   ```

3. **Inference Script**: The UI wraps `scripts/10_inference.py` in the **VISTRIKE-AI-Official** project root.

---

## Launching the UI

### Step 1: Start the Server

From the **Vistrike-Main-UI** directory (testing-ui lives inside it):

```bash
cd testing-ui
python3 app.py
```

The server will start on `http://localhost:5001`

### Step 2: Open in Browser

Navigate to:
```
http://localhost:5001
```

---

## Usage Guide

### 1. Upload Video

- **Click** the upload area or **drag and drop** a video file
- Supported formats: MP4, AVI, MOV, MKV, WebM
- Maximum file size: 500MB
- A preview will appear once a file is selected

### 2. Configure Settings

- **Detection Confidence** (0.01 - 1.0): 
  - Lower values = more detections (may include false positives)
  - Higher values = fewer, high-confidence detections only
  - Default: 0.5

- **Attribute Confidence** (0.0 - 1.0):
  - Filters low-confidence attribute predictions
  - 0.0 = show all predictions
  - 0.5+ = hide uncertain attributes
  - Default: 0.0

- **Device**:
  - **CPU**: Default, works on all systems
  - **CUDA**: For NVIDIA GPUs
  - **MPS**: For Apple Silicon (M1/M2/M3)
  - **Auto**: Automatically selects best available

- **Save Annotated Video**: Toggle to generate/output annotated video

### 3. Analyze

Click the **"Analyze Video"** button. The UI will:
- Upload the video to the server
- Run inference using the configured settings
- Display progress during processing
- Show results when complete

### 4. View Results

Results are displayed in three sections:

#### Annotated Video
- Video with bounding boxes, corner labels (RED/BLUE), guard/stance info
- Action tags overlaid (PUNCH, DEFENSE, FOOTWORK, CLINCH)
- Frame numbers and confidence scores

#### Attributes & Actions
- Quick overview of key metrics:
  - Fighter visibility (frames detected)
  - Total actions detected (punches, defenses, footwork, clinches)

#### Detailed Statistics Report
- **Video Information**: Duration, frame count, FPS, detection coverage
- **Fighter Statistics**: Per-corner stats (red/blue) including:
  - Frames visible
  - Primary guard used
  - Primary stance
- **Action Breakdowns**: For each action type (punch, defense, footwork, clinch):
  - Total count
  - Type distribution (pie charts)
  - Results/outcomes (pie charts)
  - Per-fighter breakdowns (pie charts)

---

## Expected Outputs

### Annotated Video
- Location: `results/{video_name}_{timestamp}/{video_name}_annotated.mp4`
- Format: MP4 with overlays
- Contains: Bounding boxes, labels, action tags

### JSON Files
- **`summary.json`** – **Single source of truth** for action counts and discrete events. Contains `actions` (counts/breakdowns) and `events` (list with frame, type, fighter, text). The UI reads from this for live stats, timeline, and report; it does **not** recompute events from `analysis.json`.
- `analysis.json`: Frame-by-frame raw detections; used only for per-frame display (e.g. current frame attributes during playback).

### Storage Notes
- **Uploaded videos**: Stored in system temp directory, automatically deleted after processing
- **Results**: Stored in project root `results/` folder (not in `ui/`)
- **No permanent storage**: Uploaded videos are not saved to disk permanently

### File Structure
```
Vistrike-Main-UI/testing-ui/
├── app.py                 # Flask server
└── static/               # Frontend files (HTML, CSS, JS)

results/                  # Analysis outputs (VISTRIKE-AI-Official project root)
└── {video_name}_{timestamp}/
    ├── analysis.json
    ├── summary.json
    └── {video_name}_annotated.mp4

Note: Uploaded videos are stored in system temp directory and automatically deleted after processing.
```

---

## Troubleshooting

### Server Won't Start
- **Check Python version**: Requires Python 3.7+
- **Install dependencies**: `pip install flask flask-cors`
- **Check port**: Default port is 5001. If in use, set `PORT` environment variable: `PORT=8080 python3 app.py`

### Analysis Fails
- **Check models**: Ensure models exist in `models/` directory
- **Check video format**: Supported formats listed above
- **Check file size**: Maximum 500MB
- **Check device**: If using CUDA/MPS, ensure hardware is available
- **View server logs**: Error details appear in terminal

### No Annotated Video
- **Check "Save Annotated Video"**: Must be enabled
- **Check file permissions**: Ensure write access to `ui/results/`
- **Check video codec**: Some codecs may not be supported

### Charts Not Displaying
- **Check browser console**: JavaScript errors may prevent rendering
- **Check Chart.js**: Ensure internet connection for CDN
- **Check data**: Charts only appear if action data exists

### Slow Performance
- **Use CPU**: If GPU unavailable, CPU is slower but works
- **Reduce video resolution**: Smaller videos process faster
- **Lower confidence thresholds**: Faster but less accurate

---

## API Endpoints

The UI uses these backend endpoints:

- `GET /` - Serve main UI page
- `POST /api/analyze` - Analyze uploaded video
- `GET /api/results/<filename>` - Download result files
- `GET /api/status` - Health check

---

## Technical Details

### Backend
- **Framework**: Flask (Python)
- **Inference**: Wraps `scripts/10_inference.py` (same logic as CLI)
- **File Handling**: Temporary storage in `testing-ui/uploads/`
- **Results**: Stored in project root `results/` with timestamped directories
- **Source of truth**: `summary.json` (actions + events); frontend consumes it directly; no event recomputation from `analysis.json`

### Frontend
- **Framework**: Vanilla JavaScript (no build step required)
- **Styling**: Custom CSS using VISTRIKE brand guidelines
- **Charts**: Chart.js (CDN) for pie charts
- **No Dependencies**: Pure HTML/CSS/JS for simplicity

### Performance
- **Speed**: Same as command-line inference (no overhead)
- **Accuracy**: Identical results to `scripts/10_inference.py`
- **Memory**: Depends on video size and model complexity

---

## Notes

- The UI is a **wrapper** around existing inference scripts
- **No modifications** are made to core script files
- Results are **identical** to command-line usage
- All processing happens **server-side** (Python)
- Frontend is **static** (HTML/CSS/JS only)

---

## Support

For issues or questions:
1. Check server terminal for error messages
2. Verify models are trained and available
3. Check browser console for JavaScript errors
4. Review `scripts/10_inference.py` documentation

---

**Built for everyday fighters. Powered by AI.**
