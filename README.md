# Real-Time Object Detection on Video using YOLOv8 and OpenCV

A complete, production-quality pipeline for real-time object detection on video files using the state-of-the-art YOLOv8 model.

## What This Project Demonstrates
- **YOLOv8 Inference**: Utilizing deep learning models for fast and accurate object recognition.
- **OpenCV Video I/O**: Robustly reading from and writing to video streams.
- **Real-Time Bounding Box Rendering**: Dynamically drawing informative annotations (boxes, text, filled backgrounds) over video frames.
- **CSV Logging**: Saving detailed, frame-by-frame detection data (timestamps, coordinates, confidence) for later analysis.
- **Report Generation**: Automatically compiling summary statistics of processing performance and object counts.

## Prerequisites
- Python 3.8 or higher installed on your system.
- An input video file named `input_video.mp4` placed in the same directory as the script.
- (Optional but recommended) A CUDA-compatible NVIDIA GPU for faster processing.

## Installation Instructions
1. Clone or download this repository.
2. Open a terminal or command prompt in the project directory.
3. (Optional) Create and activate a virtual environment.
4. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## How to Run
Ensure you have an `input_video.mp4` file in the project folder, then execute the main script:

```bash
python detect.py
```

While running, a window will pop up showing the live processing. You can press the `q` key while focused on this window to stop the processing early.

## Changing the Input Video
To use a different video, you can either:
1. Rename your video to `input_video.mp4` and place it in the folder.
2. Open `config.py` and change the `INPUT_VIDEO` variable to match your file's name (e.g., `INPUT_VIDEO = "my_custom_clip.avi"`).

## Configuration Options (`config.py`)
You can tweak the behavior of the application by modifying `config.py`.

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| `INPUT_VIDEO` | Path to the source video to process. | `"input_video.mp4"` |
| `OUTPUT_VIDEO` | Path where the annotated video will be saved. | `"output_video.mp4"` |
| `MODEL_NAME` | The YOLOv8 model weights file to load. | `"yolov8n.pt"` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence score to register a detection. | `0.4` |
| `IOU_THRESHOLD` | IoU threshold for overlapping box suppression (NMS). | `0.45` |
| `DISPLAY_WINDOW` | Whether to show the OpenCV live preview window. | `True` |
| `FRAME_SKIP` | Number of frames to skip to speed up processing. | `0` |
| `BOX_COLOR` | Color of bounding boxes in BGR format. | `(0, 255, 0)` |
| `TEXT_COLOR` | Color of text labels in BGR format. | `(255, 255, 255)` |
| `BOX_THICKNESS` | Pixel thickness of the bounding box lines. | `2` |
| `FONT_SCALE` | Size of the text in labels. | `0.6` |
| `LOG_CSV` | Filename for the frame-by-frame detection data. | `"detection_log.csv"` |
| `REPORT_FILE` | Filename for the final summary text report. | `"detection_report.txt"` |

## Using a Larger Model
By default, the project uses the "nano" model (`yolov8n.pt`) which is optimized for speed. If you need higher accuracy, you can change `MODEL_NAME` in `config.py` to a larger model:
- `yolov8s.pt` (Small): Better accuracy, slightly slower.
- `yolov8m.pt` (Medium): Good balance of accuracy and speed.
- `yolov8l.pt` (Large): High accuracy, requires a good GPU.
- `yolov8x.pt` (Extra Large): Maximum accuracy, very slow on CPU.

The script will automatically download the required model file from Ultralytics on the first run.

## Output Files
After a successful run, three new files will be generated:
1. **`output_video.mp4`**: The fully processed video containing drawn bounding boxes and labels over the original footage.
2. **`detection_log.csv`**: A spreadsheet listing every single detection, including the frame number, timestamp, class name, confidence score, and exact pixel coordinates.
3. **`detection_report.txt`**: A cleanly formatted text file summarizing the run, showing video resolution, FPS, total processing time, and a breakdown of how many times each object class was detected.

## Example Console Output
```
[*] Loading YOLOv8 model: yolov8n.pt...
[+] Model loaded successfully.
[+] CUDA is available. Using GPU for faster processing.
[*] Opening input video: input_video.mp4...

--- Video Details ---
Resolution: 1920x1080
FPS:        30.0
Total Frames: 450
Duration:   15.00 seconds

[*] Setting up outputs...
[*] Starting video processing...
[~] Processing: Frame 450/450 (100.0%)
[*] End of video stream reached.

[*] Processing complete. Cleaning up...
[*] Generating report at detection_report.txt...

=========================================
           PIPELINE FINISHED             
=========================================
Output Video: output_video.mp4
CSV Log:      detection_log.csv
Report:       detection_report.txt
-----------------------------------------
Top Detections:
  - person: 1245
  - car: 302
  - bicycle: 87
=========================================
```

## Technologies Used
- **Python**: Core programming language.
- **YOLOv8 (Ultralytics)**: State-of-the-art deep learning model for real-time object detection.
- **OpenCV**: Industry-standard computer vision library for video handling and frame manipulation.
- **PyTorch**: Deep learning framework powering YOLOv8.
