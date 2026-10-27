"""
Configuration settings for the YOLOv8 object detection project.
"""

# The path to the input video file.
INPUT_VIDEO = "input_video.mp4"

# The path where the processed output video will be saved.
OUTPUT_VIDEO = "output_video.mp4"

# The YOLOv8 model to use.
# "yolov8n.pt" is the nano model, which is the fastest but least accurate.
# Larger models like "yolov8s.pt" (small), "yolov8m.pt" (medium), "yolov8l.pt" (large),
# and "yolov8x.pt" (extra large) exist. They offer higher accuracy at the cost of processing speed.
MODEL_NAME = "yolov8n.pt"

# Only show detections that have a confidence score above this threshold.
# A threshold of 0.4 means we only consider predictions the model is at least 40% sure about.
# This helps filter out false positives and low-confidence guesses.
CONFIDENCE_THRESHOLD = 0.4

# Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).
# NMS is a technique used to eliminate redundant overlapping bounding boxes for the same object.
# If multiple boxes overlap by more than this threshold, the lower confidence ones are discarded.
IOU_THRESHOLD = 0.45

# Whether to show a live video window during processing.
DISPLAY_WINDOW = True

# Number of frames to skip between processed frames.
# 0 means process every frame. Setting this to 2 means process every 2nd frame,
# which can significantly speed up processing if real-time analysis of every single frame is not needed.
FRAME_SKIP = 0

# The color of the bounding boxes.
# OpenCV uses BGR (Blue, Green, Red) format instead of the standard RGB (Red, Green, Blue).
# So (0, 255, 0) represents pure Green.
BOX_COLOR = (0, 255, 0)

# The color of the text used for labels. (255, 255, 255) is white in BGR.
TEXT_COLOR = (255, 255, 255)

# The thickness of the bounding box lines in pixels.
BOX_THICKNESS = 2

# The scale (size) of the font used for labels.
FONT_SCALE = 0.6

# The path to the CSV file where frame-by-frame detections will be logged.
LOG_CSV = "detection_log.csv"

# The path to the text file where the final summary report will be saved.
REPORT_FILE = "detection_report.txt"
