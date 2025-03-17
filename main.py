from typing import List, Callable
import argparse
import os

import cv2
from ultralytics.models.yolo import detect

from model_funcs import load_model, pose_and_detect, segment, pose_detect_track, PlotConfig, AppSettings


def render_settings(frame, settings: AppSettings):
    """Render the current settings on the frame"""
    if not settings.show_controls:
        return frame
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Background for text
    cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
    
    # Add status text
    status_lines = settings.get_status_text()
    for i, line in enumerate(status_lines):
        cv2.putText(overlay, line, (20, 35 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Add controls text if help is shown
    y_offset = len(status_lines) * 25 + 50
    cv2.putText(overlay, "Controls:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for i, (key, desc) in enumerate(settings.control_descriptions.items()):
        cv2.putText(overlay, f"{key}: {desc}", (20, y_offset + 30 + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Apply the overlay with transparency
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def crop_to_portrait(frame, ratio=9/16):
    """
    Crop a landscape frame to portrait orientation by cutting off the sides.
    
    Args:
        frame: Input landscape frame
        ratio: Width to height ratio (default: 9/16 for portrait)
        
    Returns:
        Portrait cropped frame
    """
    h, w = frame.shape[:2]  # Get height and width
    
    # Calculate the target width for the portrait aspect ratio
    target_width = int(h * ratio)
    
    # If the original frame is already narrower than our target, return as is
    if w <= target_width:
        return frame
    
    # Calculate the starting x-coordinate to crop from center
    x_start = (w - target_width) // 2
    
    # Crop the frame
    cropped_frame = frame[:, x_start:x_start+target_width]
    
    return cropped_frame


def run_cam(settings: AppSettings, funcs: List[Callable]):
    """
    Run the camera and apply specified functions to each frame.

    Args:
        settings (AppSettings): Application settings including camera sources.
        funcs (list): List of functions to apply to each frame.

    Returns:
        None
    """
    # print control buttons
    print("Control keys:")
    for key, desc in settings.control_descriptions.items():
        print(f"{key}: {desc}")
    
    # Initialize camera with the first source
    cap = cv2.VideoCapture(settings.current_camera())
    if not cap.isOpened():
        print(f"Error: Could not open camera {settings.current_camera()}")
        return
    
    print("Streaming...")
    print(f"Camera: {settings.camera_names[settings.cam_index]}")
    print("Started with tracking" if settings.tracking_enabled else "Started without tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Frame not captured from {settings.camera_names[settings.cam_index]}.")
            # Try to reopen the camera or move to the next one
            cap.release()
            settings.next_camera()
            cap = cv2.VideoCapture(settings.current_camera())
            if not cap.isOpened():
                print(f"Error: Could not open camera {settings.camera_names[settings.cam_index]}")
                continue
            print(f"Switched to camera: {settings.camera_names[settings.cam_index]}")
            continue
        
        # Mirror the frame for webcams, but not for IP cameras or video files
        if isinstance(settings.current_camera(), int):
            frame = cv2.flip(frame, 1)
        # crop to portrait
        frame = crop_to_portrait(frame)
        
        # Apply the function to the frame
        func = funcs[settings.func_index]  # Get the current function to apply
        try:
            # Process the frame with the selected function
            if settings.func_index == 0:  # pose_and_detect function
                frame = func(frame, plot_config=settings.plot_config, 
                           threshold=settings.threshold, )
            else:  # segment function
                frame = func(frame, plot_config=settings.plot_config, 
                           threshold=settings.threshold)
        except Exception as e:
            print(f"Error processing frame with function {settings.func_index}: {e}")
            # If there's an error, skip processing for this frame
            settings.func_index = (settings.func_index + 1) % len(funcs)
            print(f"Switching to next function: {settings.func_index}")
            continue

        

        # Render settings on the frame if enabled
        frame = render_settings(frame, settings)

        # Display the processed frame
        cv2.imshow('', frame)
        
        # Handle key events for control buttons
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('1'):
            settings.func_index = (settings.func_index - 1) % len(funcs)
            print(f"Switched to function: {settings.func_index}")
        elif key == ord('2'):
            settings.func_index = (settings.func_index + 1) % len(funcs)
            print(f"Switched to function: {settings.func_index}")
        elif key == ord('l'):
            settings.plot_config.labels = not settings.plot_config.labels
            print("Toggled labels")
        elif key == ord('b'):
            settings.plot_config.boxes = not settings.plot_config.boxes
            print("Toggled boxes")
        elif key == ord('p'):
            settings.plot_config.probs = not settings.plot_config.probs
            print("Toggled probabilities")
        elif key == ord('c'):
            settings.plot_config.color_mode = 'instance' if settings.plot_config.color_mode == 'class' else 'class'
            print(f"Toggled color mode to {settings.plot_config.color_mode}")
        elif key == ord('t'):
            settings.tracking_enabled = not settings.tracking_enabled
            print(f"Toggled tracking to {settings.tracking_enabled}")
        elif key == ord('='):
            settings.threshold = min(settings.threshold + 0.01, 1.0)
            print(f"Increased threshold to {settings.threshold:.2f}")
        elif key == ord('-'):
            settings.threshold = max(settings.threshold - 0.01, 0.3)
            print(f"Decreased threshold to {settings.threshold:.2f}")
        elif key == ord('h'):
            settings.show_controls = not settings.show_controls
            print(f"{'Showing' if settings.show_controls else 'Hiding'} controls")
        elif key == ord(','):  # Switch to previous camera
            cap.release()
            prev_cam = settings.prev_camera()
            print(f"Switching to camera: {settings.camera_names[settings.cam_index]}")
            cap = cv2.VideoCapture(prev_cam)
            if not cap.isOpened():
                                print(f"Error: Could not open camera {settings.camera_names[settings.cam_index]}")
        elif key == ord('.'):  # Switch to next camera
            cap.release()
            next_cam = settings.next_camera()
            print(f"Switching to camera: {settings.camera_names[settings.cam_index]}")
            cap = cv2.VideoCapture(next_cam)
            if not cap.isOpened():
                print(f"Error: Could not open camera {settings.camera_names[settings.cam_index]}")
            
    cap.release()
    cv2.destroyAllWindows()


def run_cam_old(cam, funcs: List[Callable], plot_config: PlotConfig, threshold=0.5, track=False):
    """
    Run the camera and apply specified functions to each frame.

    Args:
        cam: Camera object to capture frames.
        funcs (list): List of functions to apply to each frame.
        plot_config (PlotConfig): Configuration for plotting results.
        device (str): Device to run the models on (e.g., 'cpu' or 'cuda').
        threshold (float): Confidence threshold for model predictions.

    Returns:
        None
    """
    # print control buttons
    print("Control keys:")
    print("Number keys to switch functions")
    print("l to toggle labels")
    print("b to toggle boxes")
    print("p to toggle probabilities")
    print("c to change color mode")
    print("+ to increase threshold")
    print("- to decrease threshold")
    print("ESC to quit")
    # Define control values
    func_index = 0
    tracking_enabled = track
    cap = cv2.VideoCapture(cam)
    print("Streaming...")
    print("Started with tracking" if tracking_enabled else "Started without tracking")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured.")
            break
        
        # Mirror the frame for a more natural view
        frame = cv2.flip(frame, 1)
        # Apply the function to the frame
        func = funcs[func_index]  # Get the current function to apply
        try:
            # Process the frame with the selected function
            if func_index == 0:  # pose_and_detect function
                frame = func(frame, plot_config=plot_config, threshold=threshold, track=tracking_enabled)
            else:  # segment function
                frame = func(frame, plot_config=plot_config, threshold=threshold)
        except Exception as e:
            print(f"Error processing frame with function {func_index}: {e}")
            # If there's an error, skip processing for this frame
            # change the function index to avoid repeated errors
            if func_index >= len(funcs) - 1:
                func_index = 0
            else:
                func_index += 1
            # Log the error and switch to the next function
            # This helps to avoid infinite loops in case of persistent errors
            print(f"Switching to next function: {func_index}")
            continue

        # Display the processed frame
        cv2.imshow('', frame)
        
        # Handle key events for control buttons
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('1'):
            func_index = (func_index - 1) % len(funcs)
            if func_index < 0:
                func_index = len(funcs) - 1
            print(f"Switched to function: {func_index}")
        elif key == ord('2'):
            func_index = (func_index + 1) % len(funcs)
            if func_index >= len(funcs):
                func_index = 0
            print(f"Switched to function: {func_index}")
        elif key == ord('l'):
            plot_config.labels = not plot_config.labels
            print("Toggled labels")
        elif key == ord('b'):
            plot_config.boxes = not plot_config.boxes
            print("Toggled boxes")
        elif key == ord('p'):
            plot_config.probs = not plot_config.probs
            print("Toggled probabilities")
        elif key == ord('c'):
            plot_config.color_mode = 'instance' if plot_config.color_mode == 'class' else 'class'
            print(f"Toggled color mode to {plot_config.color_mode}")
        elif key == ord('t'):
            tracking_enabled = not tracking_enabled
            print(f"Toggled tracking to {tracking_enabled}")
        elif key == ord('='):
            threshold = min(threshold + 0.01, 1.0)
            print(f"Increased threshold to {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(threshold - 0.01, 0.3)
            print(f"Decreased threshold to {threshold:.2f}")
    cap.release()
    cv2.destroyAllWindows()


def main(webcams=None, ip_cams=None, videos=None, cam_source=0, pose_model='tiny', detect_model='tiny', segment_model='tiny', device='cpu') -> None:
    """
    Main function to load models, set up camera, and run the processing loop.
    
    Args:
        webcams (list): List of webcam indices
        ip_cams (list): List of IP camera URLs
        videos (list): List of video file paths
        cam_source (int): Default camera source (used if webcams is None)
        pose_model (str): Pose model size
        detect_model (str): Detection model size
        segment_model (str): Segmentation model size
        device (str): Device to run models on
    """
    # Handle camera sources
    if webcams is None and ip_cams is None and videos is None:
        # If no lists are provided, use the single cam_source
        camera_sources = [cam_source]
        camera_names = ["Default Camera"]
    else:
        # Initialize as empty lists if None
        webcams = webcams or []
        ip_cams = ip_cams or []
        videos = videos or []
        
        # Build the camera sources and names lists
        camera_sources = []
        camera_names = []
        
        # Add webcams
        for i, webcam in enumerate(webcams):
            camera_sources.append(webcam)
            camera_names.append(f"Webcam {i}")
        
        # Add IP cameras
        for i, ip_cam in enumerate(ip_cams):
            camera_sources.append(ip_cam)
            camera_names.append(f"IP Camera {i}")
        
        # Add video files
        for i, video in enumerate(videos):
            if os.path.exists(video):
                camera_sources.append(video)
                camera_names.append(f"Video {i}: {os.path.basename(video)}")
            else:
                print(f"Warning: Video file not found: {video}")
    # Load models
    model_dict = {
        'pose': {
            'tiny': 'yolo11n-pose.pt',
            'small': 'yolo11s-pose.pt',
            'medium': 'yolo11m-pose.pt',
            'large': 'yolo11l-pose.pt'
        },
        'detect': {
            'tiny': 'yolo11n.pt',
            'small': 'yolo11s.pt',
            'medium': 'yolo11m.pt',
            'large': 'yolo11l.pt'
        },
        'segment': {
            'tiny': 'yolo11n-seg.pt',
            'small': 'yolo11s-seg.pt',
            'medium': 'yolo11m-seg.pt',
            'large': 'yolo11l-seg.pt'
        },
    }
    
    # Initialize our settings class with the camera sources
    settings = AppSettings()
    settings.camera_sources = camera_sources
    settings.camera_names = camera_names
    
    if not camera_sources:
        print("Error: No camera sources available.")
        return

    # Load the models based on the specified sizes
    pose_model_obj = load_model(model_dict['pose'][pose_model])
    detect_model_obj = load_model(model_dict['detect'][detect_model])
    segment_model_obj = load_model(model_dict['segment'][segment_model])
    pose_track_model_obj = load_model(model_dict['pose'][pose_model])  # For pose and detect tracking
    detect_track_model_obj = load_model(model_dict['detect'][detect_model])  # For pose and detect tracking

    # Define the functions to apply to each frame
    def pose_and_detect_func(img, plot_config, threshold):
        return pose_and_detect(img, pose_model_obj, detect_model_obj, device, plot_config, threshold)
    def segment_func(img, plot_config, threshold):
        return segment(img, segment_model_obj, device, plot_config, threshold)
    def pose_detect_track_func(img, plot_config, threshold):
        return pose_detect_track(img, pose_track_model_obj, detect_track_model_obj, device, plot_config, threshold)
    funcs = [pose_and_detect_func, pose_detect_track_func, segment_func]

    # Run the camera with the defined functions
    run_cam(settings, funcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLO models on camera input.')
    parser.add_argument('--cam_source', type=int, default=0, help='Camera source (default: 0)')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file with camera sources')
    parser.add_argument('--pose_model', type=str, choices=['tiny', 'small', 'medium', 'large'], default='tiny', help='Pose model size (default: tiny)')
    parser.add_argument('--detect_model', type=str, choices=['tiny', 'small', 'medium', 'large'], default='tiny', help='Detection model size (default: tiny)')
    parser.add_argument('--segment_model', type=str, choices=['tiny', 'small', 'medium', 'large'], default='tiny', help='Segmentation model size (default: tiny)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the models on (default: cpu)')
    args = parser.parse_args()
    
    # Load configuration file if specified
    if args.config:
        with open(args.config, 'r') as f:
            import json
            config = json.load(f)
            
        # Run with config file settings
        main(
            webcams=config.get('webcams', [0]), 
            ip_cams=config.get('ip_cams', []), 
            videos=config.get('videos', []), 
            pose_model=args.pose_model,
            detect_model=args.detect_model, 
            segment_model=args.segment_model, 
            device=args.device
        )
    else:
        # Run with command line arguments
        main(
            cam_source=args.cam_source, 
            pose_model=args.pose_model, 
            detect_model=args.detect_model, 
            segment_model=args.segment_model, 
            device=args.device
        )