from typing import List, Union


from dataclasses import dataclass, field

from ultralytics import YOLO


@dataclass
class PlotConfig:
    """
    Configuration for plotting results.
    """
    labels: bool = True  # Whether to display labels on the plot
    boxes: bool = True  # Whether to display bounding boxes on the plot
    probs: bool = True  # Whether to display probabilities on the plot
    color_mode: str = 'class' # 'class' or 'instance'


@dataclass
class AppSettings:
    # Display settings
    plot_config: PlotConfig = field(default_factory=PlotConfig)
    # Processing settings
    tracking_enabled: bool = False
    threshold: float = 0.5
    # Application state
    func_index: int = 0
    cam_index: int = 0
    show_controls: bool = False
    
    # Camera sources list (will be populated in main)
    camera_sources: List[Union[int, str]] = field(default_factory=list)
    camera_names: List[str] = field(default_factory=list)
    
    # Controls description for display
    control_descriptions = {
        '1/2': 'Switch functions',
        'l': 'Toggle labels',
        'b': 'Toggle boxes',
        'p': 'Toggle probabilities',
        'c': 'Change color mode',
        't': 'Toggle tracking',
        '+/-': 'Adjust threshold',
        'h': 'Show/hide this help',
        '</>': 'Switch cameras',
        'ESC': 'Quit'
    }
    
    def get_status_text(self):
        """Generate text describing current settings"""
        status_lines = [
            f"Camera: {self.camera_names[self.cam_index]}",
            f"Function: {self.func_index}",
            f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}",
            f"Threshold: {self.threshold:.2f}",
            f"Labels: {'ON' if self.plot_config.labels else 'OFF'}",
            f"Boxes: {'ON' if self.plot_config.boxes else 'OFF'}",
            f"Probabilities: {'ON' if self.plot_config.probs else 'OFF'}",
            f"Color Mode: {self.plot_config.color_mode}"
        ]
        return status_lines
    
    def next_camera(self):
        """Switch to the next camera in the list"""
        self.cam_index = (self.cam_index + 1) % len(self.camera_sources)
        return self.camera_sources[self.cam_index]
        
    def prev_camera(self):
        """Switch to the previous camera in the list"""
        self.cam_index = (self.cam_index - 1) % len(self.camera_sources)
        return self.camera_sources[self.cam_index]
    
    def current_camera(self):
        """Get the current camera source"""
        return self.camera_sources[self.cam_index]
    

def load_model(model_path: str):
    """
    Load a YOLO model from the specified path.

    Args:
        model_path (str): Path to the YOLO model file.

    Returns:
        YOLO: Loaded YOLO model.
    """
    model = YOLO(model_path)
    model.fuse()  # Fuse Conv2d + BatchNorm2d layers for faster inference

    return model


def pose_and_detect(img, pose_model: YOLO, detect_model: YOLO, device, plot_config: PlotConfig, threshold=0.5, ):
    """
    Perform pose estimation and object detection on the given image.

    Args:
        pose_model: Loaded pose estimation model.
        detect_model: Loaded object detection model.
        img: Input image for processing.

    Returns:
        tuple: Pose keypoints and detected objects.
    """
    

    
    pose_results = pose_model.predict(source=img, stream=True, conf=threshold, device=device, verbose=False)
    detect_results = detect_model.predict(source=img, stream=True, conf=threshold, device=device, verbose=False)

        # Filter out detections with the "person" label
    filtered_detect_results = []
    for result in detect_results:
        filtered_boxes = [box for box in result.boxes if box.cls != 0]  # Assuming class 0 is 'person'
        if filtered_boxes:
            result.boxes = filtered_boxes
            filtered_detect_results.append(result)

    # Plot the pose results
    for result in pose_results:
        annotated_frame = result.plot(img=img,
                                      font='Avenir Next Condensed.ttc',
                                      labels=plot_config.labels, 
                                      boxes=plot_config.boxes, 
                                      probs=plot_config.probs, 
                                      color_mode=plot_config.color_mode
                                      )

    # Plot the filtered detection results
    for result in filtered_detect_results:
        annotated_frame = result.plot(img=annotated_frame, 
                                      labels=plot_config.labels, 
                                      boxes=plot_config.boxes, 
                                      probs=plot_config.probs, 
                                      color_mode=plot_config.color_mode
                                      )

    return annotated_frame


def pose_detect_track(img, pose_model: YOLO, detect_model: YOLO, device, plot_config: PlotConfig, threshold=0.5):
        # Get model results

    # Use tracking if specified
    pose_results = pose_model.track(source=img, stream=True, conf=threshold, device=device, verbose=False)
    detect_results = detect_model.track(source=img, stream=True, conf=threshold, device=device, verbose=False)
    
    filtered_detect_results = []
    for result in detect_results:
        filtered_boxes = [box for box in result.boxes if box.cls != 0]  # Assuming class 0 is 'person'
        if filtered_boxes:
            result.boxes = filtered_boxes
            filtered_detect_results.append(result)

    # Plot the pose results
    for result in pose_results:
        annotated_frame = result.plot(img=img,
                                      font='Avenir Next Condensed.ttc',
                                      labels=plot_config.labels, 
                                      boxes=plot_config.boxes, 
                                      probs=plot_config.probs, 
                                      color_mode=plot_config.color_mode
                                      )

    # Plot the filtered detection results
    for result in filtered_detect_results:
        annotated_frame = result.plot(img=annotated_frame, 
                                      labels=plot_config.labels, 
                                      boxes=plot_config.boxes, 
                                      probs=plot_config.probs, 
                                      color_mode=plot_config.color_mode
                                      )
        
    return annotated_frame

def segment(img, model: YOLO, device, plot_config: PlotConfig, threshold=0.5):

    """
    Perform image segmentation using the specified YOLO model.

    Args:
        img: Input image for segmentation.
        model: Loaded YOLO segmentation model.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        threshold (float): Confidence threshold for segmentation.

    Returns:
        numpy.ndarray: Segmented image.
    """
    results = model.predict(source=img, stream=True, conf=threshold, device=device, verbose=False)
    
    # Plot the segmentation results
    for result in results:
        segmented_frame = result.plot(img=img,
                                      labels=plot_config.labels, 
                                      boxes=plot_config.boxes, 
                                      probs=plot_config.probs, 
                                      color_mode=plot_config.color_mode
                                      )

    return segmented_frame
