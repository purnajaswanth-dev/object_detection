"""
Utility functions for the YOLOv8 object detection project.
"""

import cv2
import csv

def draw_bounding_box(frame, box, label, confidence, color, thickness, font_scale, text_color):
    """
    Draws a bounding box and a filled label on the given video frame.

    Args:
        frame (numpy.ndarray): The current video frame.
        box (tuple): A tuple containing the (x1, y1, x2, y2) coordinates of the bounding box.
        label (str): The class name of the detected object.
        confidence (float): The confidence score of the detection.
        color (tuple): The BGR color for the bounding box and label background.
        thickness (int): The thickness of the bounding box lines.
        font_scale (float): The scale of the font for the text label.
        text_color (tuple): The BGR color for the text label.
    """
    x1, y1, x2, y2 = box
    
    # Draw the main bounding box rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Format the text to show class name and confidence percentage
    text = f"{label} {confidence * 100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate the width and height of the text string to create a fitted background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
    
    # Coordinates for the background rectangle (placed slightly above the bounding box)
    bg_y1 = max(0, y1 - text_height - 10)
    
    # Draw the filled background rectangle for the text to ensure readability on any background
    cv2.rectangle(frame, (x1, bg_y1), (x1 + text_width + 10, y1), color, -1)
    
    # Draw the text over the filled background
    cv2.putText(frame, text, (x1 + 5, y1 - 5), font, font_scale, text_color, 1, cv2.LINE_AA)

def get_video_info(cap):
    """
    Extracts metadata from a cv2.VideoCapture object.

    Args:
        cap (cv2.VideoCapture): The opened video capture object.

    Returns:
        dict: A dictionary containing video width, height, fps, total_frames, and duration_seconds.
    """
    # cv2.CAP_PROP_FRAME_WIDTH: Constant to get the width of the frames in the video stream.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # cv2.CAP_PROP_FRAME_HEIGHT: Constant to get the height of the frames in the video stream.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # cv2.CAP_PROP_FPS: Constant to get the frame rate (Frames Per Second) of the video.
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # cv2.CAP_PROP_FRAME_COUNT: Constant to get the total number of frames in the video.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate duration by dividing total frames by frames per second. Handle zero division just in case.
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds
    }

def initialize_video_writer(output_path, fps, width, height):
    """
    Creates and returns a cv2.VideoWriter object for saving the processed video.

    A VideoWriter is an OpenCV class that takes individual frames and encodes them into a video file.
    The codec determines how the video data is compressed.

    Args:
        output_path (str): The path where the output video will be saved.
        fps (float): The frame rate of the output video.
        width (int): The width of the output video frames.
        height (int): The height of the output video frames.

    Returns:
        cv2.VideoWriter: The initialized video writer object.
    """
    # fourcc stands for "Four-Character Code" and is used to specify the video codec.
    # 'mp4v' is a widely compatible codec for writing .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Initialize the VideoWriter with the path, codec, fps, and frame dimensions
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer

def calculate_stats(detection_counts):
    """
    Calculates summary statistics from a dictionary of detection counts.

    Args:
        detection_counts (dict): A dictionary mapping class names to their total detection counts.

    Returns:
        tuple: A tuple containing (total_detections, most_detected_class, least_detected_class).
    """
    if not detection_counts:
        return 0, "None", "None"
        
    total_detections = sum(detection_counts.values())
    
    # Find the class with the maximum count
    most_detected = max(detection_counts, key=detection_counts.get)
    
    # Find the class with the minimum count
    least_detected = min(detection_counts, key=detection_counts.get)
    
    return total_detections, most_detected, least_detected

def save_report(report_path, video_info, detection_counts, total_frames_processed, processing_time_seconds):
    """
    Writes a formatted summary report to a text file.

    Args:
        report_path (str): The path to save the report file.
        video_info (dict): Video metadata from get_video_info().
        detection_counts (dict): A dictionary mapping class names to detection counts.
        total_frames_processed (int): The number of frames actually processed (accounting for skips).
        processing_time_seconds (float): Total time taken for the processing loop.
    """
    total_detections, most_detected, least_detected = calculate_stats(detection_counts)
    
    # Calculate processing FPS to measure how fast the code ran
    processing_fps = total_frames_processed / processing_time_seconds if processing_time_seconds > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("========================================\n")
        f.write("       YOLOv8 DETECTION REPORT          \n")
        f.write("========================================\n\n")
        
        f.write("--- Video Metadata ---\n")
        f.write(f"Resolution:       {video_info['width']}x{video_info['height']}\n")
        f.write(f"Original FPS:     {video_info['fps']:.2f}\n")
        f.write(f"Total Duration:   {video_info['duration_seconds']:.2f} seconds\n")
        f.write(f"Total Frames:     {video_info['total_frames']}\n\n")
        
        f.write("--- Processing Stats ---\n")
        f.write(f"Frames Processed: {total_frames_processed}\n")
        f.write(f"Processing Time:  {processing_time_seconds:.2f} seconds\n")
        f.write(f"Processing FPS:   {processing_fps:.2f} fps\n\n")
        
        f.write("--- Detection Summary ---\n")
        f.write(f"Total Detections: {total_detections}\n")
        f.write(f"Most Detected:    {most_detected} ({detection_counts.get(most_detected, 0)})\n")
        f.write(f"Least Detected:   {least_detected} ({detection_counts.get(least_detected, 0)})\n\n")
        
        f.write("--- Per-Class Counts ---\n")
        # Sort classes alphabetically for a cleaner report
        for cls, count in sorted(detection_counts.items()):
            f.write(f"- {cls.ljust(15)}: {count}\n")
            
        f.write("\n========================================\n")

def initialize_csv_log(csv_path):
    """
    Creates a CSV file for logging detections and writes the header row.

    Args:
        csv_path (str): The path to the CSV file to create.

    Returns:
        tuple: A tuple containing the open file handle and the csv.writer object.
               The caller is responsible for closing the file handle.
    """
    # Open the file in write mode, with newline='' to prevent blank lines in Windows
    f = open(csv_path, mode='w', newline='')
    writer = csv.writer(f)
    
    # Write the column headers for our tracking data
    writer.writerow([
        "frame_number", 
        "timestamp_seconds", 
        "class_name", 
        "confidence", 
        "x1", "y1", "x2", "y2"
    ])
    
    return f, writer

def log_detection_to_csv(writer, frame_number, fps, class_name, confidence, box):
    """
    Writes a single detection record as a row in the CSV file.

    Args:
        writer (csv.writer): The CSV writer object.
        frame_number (int): The current frame number.
        fps (float): The frame rate of the video (used to calculate timestamp).
        class_name (str): The predicted class name.
        confidence (float): The confidence score.
        box (tuple): The bounding box coordinates (x1, y1, x2, y2).
    """
    # Calculate the timestamp in seconds based on frame number and original FPS
    timestamp = frame_number / fps if fps > 0 else 0
    x1, y1, x2, y2 = box
    
    # Write the row data corresponding to the headers defined in initialize_csv_log
    writer.writerow([
        frame_number,
        f"{timestamp:.3f}",
        class_name,
        f"{confidence:.4f}",
        x1, y1, x2, y2
    ])
