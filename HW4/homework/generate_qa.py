import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)
        min_box_size: Minimum size for bounding boxes to be considered

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
        - bounding_box: (x1, y1, x2, y2) coordinates of the bounding box
    """
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return []

    # Calculate scaling factors for the current image size
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Extract kart character names from info
    kart_names = {}
    if "kart_names" in info:
        kart_names = info["kart_names"]

    # Image center coordinates
    image_center_x = img_width / 2
    image_center_y = img_height / 2

    # Extract kart objects
    karts = []
    min_center_distance = float('inf')
    center_kart_index = -1

    for i, detection in enumerate(frame_detections):
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        # Only consider objects of type "Kart"
        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        # Skip if bounding box is out of the image
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Calculate center of the bounding box
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        # Calculate distance to image center
        distance_to_center = ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5

        # Get kart name from the kart_names dict, or use a default name
        kart_name = kart_names.get(str(track_id), f"Kart {track_id}")
        
        # Identify ego car (typically track_id 0)
        is_ego = (track_id == 0)

        # Create kart object
        kart = {
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_center_kart": False,  # Will update this later
            "is_ego": is_ego,
            "bounding_box": (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
            "distance_to_center": distance_to_center
        }
        
        karts.append(kart)

        # Check if this is the kart closest to the center
        if distance_to_center < min_center_distance:
            min_center_distance = distance_to_center
            center_kart_index = len(karts) - 1

    # Mark the center kart
    if center_kart_index >= 0:
        karts[center_kart_index]["is_center_kart"] = True

    return karts

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract track name from the info
    if "track_name" in info:
        return info["track_name"]
    
    # If track_name is not available, try to get it from filename or return default
    info_file = Path(info_path)
    parent_dir = info_file.parent
    
    # Try to get track info from parent directory name as fallback
    if parent_dir.name.lower() != "valid" and parent_dir.name.lower() != "train":
        return parent_dir.name
        
    return "Unknown Track"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # Extract kart objects and track info
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    # If no karts are detected, return a minimal set of QA pairs
    if not karts:
        return [
            {
                "question": "How many karts are visible in this image?",
                "answer": "0"
            },
            {
                "question": "What track is shown in this image?",
                "answer": track_name
            }
        ]
    
    # Find ego car
    ego_car = None
    for kart in karts:
        if kart["is_ego"]:
            ego_car = kart
            break
    
    # If no ego car is found, use the center kart as reference
    if ego_car is None:
        for kart in karts:
            if kart["is_center_kart"]:
                ego_car = kart
                break
    
    # If still no reference kart, use the first one
    if ego_car is None and karts:
        ego_car = karts[0]
    
    qa_pairs = []
    
    # 1. Ego car question (if we have one)
    if ego_car and ego_car["is_ego"]:
        qa_pairs.append({
            "question": "What kart is the ego car (player's kart)?",
            "answer": ego_car["kart_name"]
        })
    
    # 2. Total karts question
    qa_pairs.append({
        "question": "How many karts are visible in this image?",
        "answer": str(len(karts))
    })
    
    # 3. Track information questions
    qa_pairs.append({
        "question": "What track is shown in this image?",
        "answer": track_name
    })
    
    # If we have an ego car, we can ask positional questions
    if ego_car:
        # Calculate the relative positions of other karts
        karts_left = 0
        karts_right = 0
        karts_front = 0
        karts_behind = 0
        
        # Get ego car's center position
        ego_x, ego_y = ego_car["center"]
        
        # Assuming y-axis points downward (higher y-value means lower on screen)
        # and x-axis points rightward (higher x-value means more to the right)
        for kart in karts:
            if kart is ego_car:
                continue
                
            kart_x, kart_y = kart["center"]
            
            # 4. Relative position questions for each kart
            # Left/Right position
            if kart_x < ego_x:
                position_x = "to the left of"
                karts_left += 1
            else:
                position_x = "to the right of"
                karts_right += 1
                
            # Front/Behind position (in racing games, usually lower y means further ahead)
            if kart_y < ego_y:
                position_y = "in front of"
                karts_front += 1
            else:
                position_y = "behind"
                karts_behind += 1
                
            qa_pairs.append({
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": position_x.split(" ")[2] if "to the" in position_x else position_x
            })
            
            qa_pairs.append({
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": position_y if position_y != "in front of" else "in front"
            })
        
        # 5. Counting questions
        qa_pairs.append({
            "question": "How many karts are to the left of the ego car?",
            "answer": str(karts_left)
        })
        
        qa_pairs.append({
            "question": "How many karts are to the right of the ego car?",
            "answer": str(karts_right)
        })
        
        qa_pairs.append({
            "question": "How many karts are in front of the ego car?",
            "answer": str(karts_front)
        })
        
        qa_pairs.append({
            "question": "How many karts are behind the ego car?",
            "answer": str(karts_behind)
        })
    
    # 6. Positional questions (not dependent on ego car)
    leftmost_kart = min(karts, key=lambda k: k["center"][0])
    rightmost_kart = max(karts, key=lambda k: k["center"][0])
    
    qa_pairs.append({
        "question": "Which kart is the leftmost in the image?",
        "answer": leftmost_kart["kart_name"]
    })
    
    qa_pairs.append({
        "question": "Which kart is the rightmost in the image?",
        "answer": rightmost_kart["kart_name"]
    })
    
    # 7. Distance questions
    if len(karts) > 1:
        # Find the two closest karts
        min_distance = float('inf')
        closest_pair = None
        
        for i in range(len(karts)):
            for j in range(i+1, len(karts)):
                kart1 = karts[i]
                kart2 = karts[j]
                
                # Calculate Euclidean distance between the two karts
                x1, y1 = kart1["center"]
                x2, y2 = kart2["center"]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (kart1["kart_name"], kart2["kart_name"])
        
        if closest_pair:
            qa_pairs.append({
                "question": "Which two karts are closest to each other?",
                "answer": f"{closest_pair[0]} and {closest_pair[1]}"
            })
    
    return qa_pairs




def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def process_dataset(data_folder: str, output_file: str, max_samples: int = None):
    """
    Process the entire dataset and generate QA pairs for all info files.
    
    Args:
        data_folder: Path to the folder containing the data
        output_file: Path to output JSON file to save the QA pairs
        max_samples: Maximum number of samples to process (None for all)
    """
    data_path = Path(data_folder)
    info_files = list(data_path.glob("**/*_info.json"))
    
    # Limit number of samples if specified
    if max_samples:
        info_files = info_files[:max_samples]
    
    qa_dataset = []
    
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        
        # For each info file, find all associated image files
        image_files = list(info_file.parent.glob(f"{base_name}_*_im.jpg"))
        
        for image_file in image_files:
            # Extract view index from image filename
            _, view_index = extract_frame_info(str(image_file))
            
            # Generate QA pairs for this view
            qa_pairs = generate_qa_pairs(str(info_file), view_index)
            
            # Add to dataset
            qa_dataset.append({
                "image_path": str(image_file),
                "info_path": str(info_file),
                "qa_pairs": qa_pairs
            })
            
            print(f"Processed {image_file.name}")
    
    # Save dataset to output file
    with open(output_file, 'w') as f:
        json.dump(qa_dataset, f, indent=2)
    
    print(f"Generated {len(qa_dataset)} QA samples saved to {output_file}")


def generate_balanced_qa_dataset(data_folder: str, output_file: str, samples_per_track: int = 100):
    """
    Generate a balanced QA dataset with equal representation from all available tracks.
    
    Args:
        data_folder: Path to the folder containing the data
        output_file: Path to output JSON file to save the QA pairs
        samples_per_track: Number of samples to include per track
    """
    data_path = Path(data_folder)
    info_files = list(data_path.glob("**/*_info.json"))
    
    # Group info files by track
    track_info_files = {}
    for info_file in info_files:
        track_name = extract_track_info(str(info_file))
        if track_name not in track_info_files:
            track_info_files[track_name] = []
        track_info_files[track_name].append(info_file)
    
    qa_dataset = []
    
    # Process each track
    for track_name, files in track_info_files.items():
        print(f"Processing track: {track_name} ({len(files)} info files)")
        
        # Limit samples per track
        track_samples = 0
        
        for info_file in files:
            base_name = info_file.stem.replace("_info", "")
            
            # For each info file, find all associated image files
            image_files = list(info_file.parent.glob(f"{base_name}_*_im.jpg"))
            
            for image_file in image_files:
                if track_samples >= samples_per_track:
                    break
                    
                # Extract view index from image filename
                _, view_index = extract_frame_info(str(image_file))
                
                # Generate QA pairs for this view
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
                
                # Add to dataset
                qa_dataset.append({
                    "image_path": str(image_file),
                    "info_path": str(info_file),
                    "qa_pairs": qa_pairs
                })
                
                track_samples += 1
                print(f"Processed {image_file.name} - {track_samples}/{samples_per_track} for {track_name}")
                
            if track_samples >= samples_per_track:
                break
    
    # Save dataset to output file
    with open(output_file, 'w') as f:
        json.dump(qa_dataset, f, indent=2)
    
    print(f"Generated balanced dataset with {len(qa_dataset)} QA samples saved to {output_file}")


def format_for_vlm_training(qa_dataset_path: str, output_file: str):
    """
    Format the QA dataset into a format suitable for VLM training.
    
    Args:
        qa_dataset_path: Path to the QA dataset JSON file
        output_file: Path to output JSON file for VLM training
    """
    # Load QA dataset
    with open(qa_dataset_path) as f:
        qa_dataset = json.load(f)
    
    # Format for VLM training
    vlm_dataset = []
    
    for sample in qa_dataset:
        image_path = sample["image_path"]
        
        for qa_pair in sample["qa_pairs"]:
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            
            vlm_dataset.append({
                "image": image_path,
                "question": question,
                "answer": answer
            })
    
    # Save formatted dataset
    with open(output_file, 'w') as f:
        json.dump(vlm_dataset, f, indent=2)
    
    print(f"Formatted {len(vlm_dataset)} QA pairs for VLM training, saved to {output_file}")




"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "process": process_dataset,
        "balanced": generate_balanced_qa_dataset,
        "format": format_for_vlm_training
    })

if __name__ == "__main__":
    main()
