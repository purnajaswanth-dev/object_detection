"""
Main Script for Real-Time Object Detection using YOLOv8 and OpenCV.

This script reads an input video, processes it frame-by-frame using a pre-trained YOLOv8 model
to detect objects, draws bounding boxes and labels on the detected objects, logs the detections
to a CSV file, displays the live processing, and saves the annotated frames to a new output video.
Finally, it generates a summary report text file.
"""

# ==============================================================================
# Section A — Imports and Setup
# ==============================================================================
import cv2
import time
import os
import sys
import torch
from ultralytics import YOLO

# Import custom utility functions for drawing, logging, and reporting
from utils import (
    draw_bounding_box, get_video_info, initialize_video_writer,
    initialize_csv_log, log_detection_to_csv, save_report
)

# Import all configuration constants
from config import *


def main():
    # ==============================================================================
    # Section B — Model Loading
    # ==============================================================================
    print(f"[*] Loading YOLOv8 model: {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"[-] Error loading model: {e}")
        sys.exit(1)

    # CUDA is Nvidia's parallel computing platform. If available, PyTorch can use the GPU
    # to perform matrix operations much faster than the CPU.
    if torch.cuda.is_available():
        print("[+] CUDA is available. Using GPU for faster processing.")
    else:
        print("[-] CUDA is not available. Using CPU. Processing will be slower.")

    # ==============================================================================
    # Section C — Video Input
    # ==============================================================================
    print(f"[*] Opening input video: {INPUT_VIDEO}...")
    if not os.path.exists(INPUT_VIDEO):
        print(f"[-] Error: Input video file '{INPUT_VIDEO}' not found in the current directory.")
        sys.exit(1)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video file '{INPUT_VIDEO}'.")
        sys.exit(1)

    # Extract and display video metadata
    video_info = get_video_info(cap)
    print("\n--- Video Details ---")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"FPS:        {video_info['fps']}")
    print(f"Total Frames: {video_info['total_frames']}")
    print(f"Duration:   {video_info['duration_seconds']:.2f} seconds\n")

    # ==============================================================================
    # Section D — Output Setup
    # ==============================================================================
    print("[*] Setting up outputs...")
    # Setup video writer for the output file
    video_writer = initialize_video_writer(
        OUTPUT_VIDEO, video_info['fps'], video_info['width'], video_info['height']
    )
    
    # Setup CSV logging
    csv_file_handle, csv_writer = initialize_csv_log(LOG_CSV)
    
    # Dictionary to track the total number of times each class is detected
    detection_counts = {}

    # ==============================================================================
    # Section E — Main Processing Loop
    # ==============================================================================
    print("[*] Starting video processing...")
    start_time = time.time()
    
    frame_number = 0
    frames_processed = 0

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        # If ret is False, we have reached the end of the video or there's an error reading
        if not ret:
            print("\n[*] End of video stream reached.")
            break
            
        frame_number += 1

        # Frame skipping logic: use modulo operator to only process 1 out of every (FRAME_SKIP + 1) frames.
        # e.g., if FRAME_SKIP = 2, we process frame 1, skip 2, 3, process 4... (frame_number % 3 == 1)
        if FRAME_SKIP > 0 and frame_number % (FRAME_SKIP + 1) != 1:
            continue
            
        frames_processed += 1

        # Run inference on the current frame. verbose=False suppresses per-frame console spam from YOLO
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        # Extract bounding boxes from the results.
        # results[0] contains the predictions for the first (and only) image passed.
        # .boxes contains the BoundingBox object.
        boxes = results[0].boxes

        for box in boxes:
            # .cls contains the class ID tensor. We convert it to a python integer.
            class_id = int(box.cls[0].item())
            
            # .conf contains the confidence score tensor. We convert it to a python float.
            confidence = float(box.conf[0].item())
            
            # .xyxy contains the [x1, y1, x2, y2] coordinates tensor. We map to int for OpenCV drawing.
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Lookup the human-readable class name from the model's dictionary
            class_name = model.names[class_id]
            
            # Update our running tally of detections
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            # Draw the box and label on the current frame
            draw_bounding_box(
                frame, (x1, y1, x2, y2), class_name, confidence, 
                BOX_COLOR, BOX_THICKNESS, FONT_SCALE, TEXT_COLOR
            )
            
            # Log this specific detection to our CSV file
            log_detection_to_csv(
                csv_writer, frame_number, video_info['fps'], 
                class_name, confidence, (x1, y1, x2, y2)
            )

        # Write the processed (annotated) frame to our output video file
        video_writer.write(frame)

        # Live preview handling
        if DISPLAY_WINDOW:
            # cv2.imshow displays an image in a window. We update it every frame to create a video player effect.
            cv2.imshow("YOLOv8 Real-Time Detection", frame)
            
            # cv2.waitKey(1) waits for 1 millisecond for a keyboard event.
            # It's necessary to allow OpenCV to refresh the GUI window.
            # If the user presses 'q' (ASCII 113), we break the loop to stop processing early.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[*] Processing interrupted by user.")
                break

        # Print progress update to console every 30 frames
        if frame_number % 30 == 0:
            percent_complete = (frame_number / video_info['total_frames']) * 100
            print(f"\r[~] Processing: Frame {frame_number}/{video_info['total_frames']} ({percent_complete:.1f}%)", end="")

    # ==============================================================================
    # Section F — Cleanup and Report
    # ==============================================================================
    print("\n[*] Processing complete. Cleaning up...")
    
    # Calculate the total wall-clock time spent in the processing loop
    processing_time = time.time() - start_time
    
    # Release file handles and system resources
    cap.release()
    video_writer.release()
    csv_file_handle.close()
    
    # Close all graphical windows created by OpenCV
    cv2.destroyAllWindows()
    
    print(f"[*] Generating report at {REPORT_FILE}...")
    save_report(
        REPORT_FILE, video_info, detection_counts, 
        frames_processed, processing_time
    )

    # Print final summary to console
    print("\n=========================================")
    print("           PIPELINE FINISHED             ")
    print("=========================================")
    print(f"Output Video: {OUTPUT_VIDEO}")
    print(f"CSV Log:      {LOG_CSV}")
    print(f"Report:       {REPORT_FILE}")
    print("-----------------------------------------")
    
    # Show top 3 detections if any occurred
    if detection_counts:
        print("Top Detections:")
        sorted_counts = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_counts[:3]:
            print(f"  - {cls}: {count}")
    else:
        print("No objects were detected.")
    print("=========================================\n")


if __name__ == "__main__":
    main()
