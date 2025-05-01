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
    # Format is: XXXXX_YY_im.jpg where XXXXX is frame_id and YY is view_index
    # Example: 00000_00_im.jpg, 00000_01_im.jpg, etc.
    
    parts = filename.split("_")
    if len(parts) >= 2:
        try:
            # For files like 00000_00_im.jpg, parts[0] is '00000' and parts[1] is '00'
            frame_id = int(parts[0])  # Already in decimal, no need for hex conversion
            view_index = int(parts[1])
            return frame_id, view_index
        except ValueError:
            # Handle the case where conversion fails
            print(f"Warning: Could not extract frame info from {filename}")
            return 0, 0
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
    try:
        pil_image = Image.open(image_path)
        if pil_image is None:
            raise ValueError(f"Could not read image at {image_path}")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        # Return a blank image
        return np.zeros((400, 600, 3), dtype=np.uint8)

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    try:
        with open(info_path) as f:
            info = json.load(f)
    except Exception as e:
        print(f"Error reading info file {info_path}: {e}")
        return np.array(pil_image)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if "detections" in info and view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections or no detections found")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        if len(detection) < 6:
            print(f"Warning: Invalid detection format: {detection}")
            continue
            
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
    try:
        with open(info_path) as f:
            info = json.load(f)
    except Exception as e:
        print(f"Error reading info file {info_path}: {e}")
        return []

    # Get the correct detection frame based on view index
    if "detections" in info and view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections or no detections found")
        return []

    # Calculate scaling factors for the current image size
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Extract kart character names from info
    kart_names = []
    if "karts" in info:
       kart_names = info["karts"]

    # Image center coordinates
    image_center_x = img_width / 2
    image_center_y = img_height / 2

    # Extract kart objects
    karts = []
    min_center_distance = float('inf')
    center_kart_index = -1

    for i, detection in enumerate(frame_detections):
        if len(detection) < 6:
            print(f"Warning: Invalid detection format: {detection}")
            continue
            
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

        # Later in the code...
        # Get kart name using the track_id as an index into the kart_names list
        if isinstance(kart_names, list) and 0 <= track_id < len(kart_names):
            kart_name = kart_names[track_id]
        else:
            kart_name = f"Kart {track_id}"
        
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
    try:
        with open(info_path) as f:
            info = json.load(f)
    except Exception as e:
        print(f"Error reading info file {info_path}: {e}")
        return "Unknown Track"

    # Extract track name from the info
    if "track" in info:
        return info["track"]
    
    # If track_name is not available, try to get it from filename or return default
    info_file = Path(info_path)
    parent_dir = info_file.parent
    
    # Try to get track info from parent directory name as fallback
    if parent_dir.name.lower() != "valid" and parent_dir.name.lower() != "train":
        return parent_dir.name
        
    return "Unknown Track"


def generate_question_variations(question_template, replacements):
    """
    Generate variations of a question using different phrasings.
    
    Args:
        question_template: Template with placeholders
        replacements: List of replacement dictionaries for the placeholders
    
    Returns:
        List of question variations
    """
    variations = []
    for replacement in replacements:
        variations.append(question_template.format(**replacement))
    return variations


def multiply_qa_variations(qa_pairs):
    """
    Create variations of existing QA pairs to increase dataset size.
    
    Args:
        qa_pairs: Original list of QA pairs
    
    Returns:
        Extended list of QA pairs with variations
    """
    extended_qa_pairs = []
    
    for qa in qa_pairs:
        # Add original QA pair
        extended_qa_pairs.append(qa)
        
        # Add variations with different phrasings
        question = qa["question"]
        answer = qa["answer"]
        
        # For counting questions
        if question.startswith("How many"):
            variations = [
                f"Count the number of {question.split('How many ')[1]}",
                f"What is the total count of {question.split('How many ')[1]}"
            ]
            
            for var in variations:
                extended_qa_pairs.append({
                    "question": var,
                    "answer": answer
                })
        
        # For positional questions
        if "leftmost" in question or "rightmost" in question:
            if "leftmost" in question:
                var = question.replace("leftmost", "furthest to the left")
            else:
                var = question.replace("rightmost", "furthest to the right")
                
            extended_qa_pairs.append({
                "question": var,
                "answer": answer
            })
        
        # For yes/no questions, add inverse question
        if "Is " in question and (" to the " in question or " in front " in question or " behind " in question):
            parts = question.split(" is ")
            if len(parts) > 1:
                subject = parts[0].replace("Is ", "")
                predicate = parts[1]
                
                options = []
                if "to the left or right" in predicate:
                    options = ["left", "right"]
                elif "in front of or behind" in predicate:
                    options = ["in front", "behind"]
                
                if options and len(options) == 2:
                    inverse_answer = options[0] if answer == options[1] else options[1]
                    inverse_question = question.replace(options[0], "XXX").replace(options[1], options[0]).replace("XXX", options[1])
                    
                    extended_qa_pairs.append({
                        "question": inverse_question,
                        "answer": inverse_answer
                    })
                    
        # For track name questions
        if "What track is shown" in question:
            variations = [
                "Which track is this race taking place on?",
                "Can you identify the racing track in this image?",
                "What is the name of this racing track?"
            ]
            
            for var in variations:
                extended_qa_pairs.append({
                    "question": var,
                    "answer": answer
                })
                
        # For ego car questions
        if "What kart is the ego car" in question:
            variations = [
                "Which character's kart is the player controlling?",
                "Who is the player character in this race?",
                "Which racer is the player controlling?"
            ]
            
            for var in variations:
                extended_qa_pairs.append({
                    "question": var,
                    "answer": answer
                })
    
    return extended_qa_pairs


def generate_multi_view_qa_pairs(info_path: str, view_indices: list) -> list:
    """
    Generate questions that reference multiple views of the same scene.
    
    Args:
        info_path: Path to the info.json file
        view_indices: List of view indices to analyze
    
    Returns:
        List of QA pairs comparing views
    """
    multi_view_qa = []
    
    # Get karts in each view
    views_karts = {}
    for view_index in view_indices:
        views_karts[view_index] = extract_kart_objects(info_path, view_index)
    
    # Compare karts across views
    for i, view1 in enumerate(view_indices):
        for view2 in view_indices[i+1:]:
            karts1 = views_karts[view1]
            karts2 = views_karts[view2]
            
            # Get kart IDs in each view
            kart_ids1 = [k["instance_id"] for k in karts1]
            kart_ids2 = [k["instance_id"] for k in karts2]
            
            # Find karts that appear in both views
            common_karts = []
            for kart1 in karts1:
                if kart1["instance_id"] in kart_ids2:
                    kart2 = next(k for k in karts2 if k["instance_id"] == kart1["instance_id"])
                    common_karts.append((kart1, kart2))
            
            # Questions about common karts
            if common_karts:
                multi_view_qa.append({
                    "question": f"How many karts appear in both view {view1} and view {view2}?",
                    "answer": str(len(common_karts))
                })
                
                for kart1, kart2 in common_karts:
                    # Compare positions
                    x1, y1 = kart1["center"]
                    x2, y2 = kart2["center"]
                    
                    x_diff = "moved right" if x2 > x1 else "moved left" if x2 < x1 else "didn't move horizontally"
                    y_diff = "moved down" if y2 > y1 else "moved up" if y2 < y1 else "didn't move vertically"
                    
                    multi_view_qa.append({
                        "question": f"How did {kart1['kart_name']} move from view {view1} to view {view2} horizontally?",
                        "answer": x_diff
                    })
                    
                    multi_view_qa.append({
                        "question": f"How did {kart1['kart_name']} move from view {view1} to view {view2} vertically?",
                        "answer": y_diff
                    })
                    
                    # Variations of movement questions
                    multi_view_qa.append({
                        "question": f"Did {kart1['kart_name']} move left or right from view {view1} to view {view2}?",
                        "answer": x_diff.replace("moved ", "") if x_diff != "didn't move horizontally" else "neither"
                    })
                    
                    multi_view_qa.append({
                        "question": f"Did {kart1['kart_name']} move up or down from view {view1} to view {view2}?",
                        "answer": y_diff.replace("moved ", "") if y_diff != "didn't move vertically" else "neither"
                    })
    
    return multi_view_qa


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
        if kart.get("is_ego", False):
            ego_car = kart
            break
    
    # If no ego car is found, use the center kart as reference
    if ego_car is None:
        for kart in karts:
            if kart.get("is_center_kart", False):
                ego_car = kart
                break
    
    # If still no reference kart, use the first one
    if ego_car is None and karts:
        ego_car = karts[0]
    
    qa_pairs = []
    
    # 1. Ego car question (if we have one)
    if ego_car and ego_car.get("is_ego", True):
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
    if len(karts) > 0:
        leftmost_kart = min(karts, key=lambda k: k["center"][0])
        rightmost_kart = max(karts, key=lambda k: k["center"][0])
        topmost_kart = min(karts, key=lambda k: k["center"][1])
        bottommost_kart = max(karts, key=lambda k: k["center"][1])
        
        qa_pairs.append({
            "question": "Which kart is the leftmost in the image?",
            "answer": leftmost_kart["kart_name"]
        })
        
        qa_pairs.append({
            "question": "Which kart is the rightmost in the image?",
            "answer": rightmost_kart["kart_name"]
        })
        
        qa_pairs.append({
            "question": "Which kart is highest in the image?",
            "answer": topmost_kart["kart_name"]
        })
        
        qa_pairs.append({
            "question": "Which kart is lowest in the image?",
            "answer": bottommost_kart["kart_name"]
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
            
        # Find the two furthest karts
        max_distance = 0
        furthest_pair = None
        
        for i in range(len(karts)):
            for j in range(i+1, len(karts)):
                kart1 = karts[i]
                kart2 = karts[j]
                
                # Calculate Euclidean distance between the two karts
                x1, y1 = kart1["center"]
                x2, y2 = kart2["center"]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                
                if distance > max_distance:
                    max_distance = distance
                    furthest_pair = (kart1["kart_name"], kart2["kart_name"])
        
        if furthest_pair:
            qa_pairs.append({
                "question": "Which two karts are furthest from each other?",
                "answer": f"{furthest_pair[0]} and {furthest_pair[1]}"
            })
            
        # Compare distances of pairs of karts
        for i in range(len(karts)):
            for j in range(i+1, len(karts)):
                for k in range(j+1, len(karts)):
                    kart1 = karts[i]
                    kart2 = karts[j]
                    kart3 = karts[k]
                    
                    x1, y1 = kart1["center"]
                    x2, y2 = kart2["center"]
                    x3, y3 = kart3["center"]
                    
                    dist12 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    dist13 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
                    dist23 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
                    
                    if dist12 < dist13 and dist12 < dist23:
                        closest = f"{kart1['kart_name']} and {kart2['kart_name']}"
                    elif dist13 < dist12 and dist13 < dist23:
                        closest = f"{kart1['kart_name']} and {kart3['kart_name']}"
                    else:
                        closest = f"{kart2['kart_name']} and {kart3['kart_name']}"
                    
                    qa_pairs.append({
                        "question": f"Among {kart1['kart_name']}, {kart2['kart_name']}, and {kart3['kart_name']}, which two karts are closest to each other?",
                        "answer": closest
                    })
            
    # 8. Size comparison questions
    if len(karts) > 1:
        # Sort karts by bounding box size (area)
        karts_by_size = sorted(karts, key=lambda k: 
                          (k["bounding_box"][2] - k["bounding_box"][0]) * 
                          (k["bounding_box"][3] - k["bounding_box"][1]))
        
        smallest_kart = karts_by_size[0]
        largest_kart = karts_by_size[-1]
        
        qa_pairs.append({
            "question": "Which kart appears largest in the image?",
            "answer": largest_kart["kart_name"]
        })
        
        qa_pairs.append({
            "question": "Which kart appears smallest in the image?",
            "answer": smallest_kart["kart_name"]
        })
        
        # Compare sizes between pairs of karts
        for i in range(len(karts)):
            for j in range(i+1, len(karts)):
                kart1 = karts[i]
                kart2 = karts[j]
                
                # Calculate areas
                area1 = ((kart1["bounding_box"][2] - kart1["bounding_box"][0]) * 
                         (kart1["bounding_box"][3] - kart1["bounding_box"][1]))
                area2 = ((kart2["bounding_box"][2] - kart2["bounding_box"][0]) * 
                         (kart2["bounding_box"][3] - kart2["bounding_box"][1]))
                
                larger = kart1["kart_name"] if area1 > area2 else kart2["kart_name"]
                
                qa_pairs.append({
                    "question": f"Which appears larger in the image: {kart1['kart_name']} or {kart2['kart_name']}?",
                    "answer": larger
                })
                
    # 9. Questions about kart quadrants (dividing image into 4 sections)
    if len(karts) > 0:
        # Define quadrants (top-left, top-right, bottom-left, bottom-right)
        mid_x = img_width / 2
        mid_y = img_height / 2
        
        quadrants = {
            "top-left": [],
            "top-right": [],
            "bottom-left": [],
            "bottom-right": []
        }
        
        for kart in karts:
            x, y = kart["center"]
            if x < mid_x and y < mid_y:
                quadrants["top-left"].append(kart["kart_name"])
            elif x >= mid_x and y < mid_y:
                quadrants["top-right"].append(kart["kart_name"])
            elif x < mid_x and y >= mid_y:
                quadrants["bottom-left"].append(kart["kart_name"])
            else:
                quadrants["bottom-right"].append(kart["kart_name"])
        
        for quadrant, karts_in_quadrant in quadrants.items():
            qa_pairs.append({
                "question": f"How many karts are in the {quadrant} quadrant of the image?",
                "answer": str(len(karts_in_quadrant))
            })
            
            if karts_in_quadrant:
                qa_pairs.append({
                    "question": f"Which karts are in the {quadrant} quadrant of the image?",
                    "answer": ", ".join(karts_in_quadrant) if karts_in_quadrant else "None"
                })
    
    # 10. Compositional questions
    if len(karts) > 1:
        # Compositional questions combining position and other attributes
        for kart in karts:
            # Questions about karts relative to this one
            karts_to_left = [k for k in karts if k["center"][0] < kart["center"][0] and k is not kart]
            karts_to_right = [k for k in karts if k["center"][0] > kart["center"][0] and k is not kart]
            
            if karts_to_left:
                closest_left = max(karts_to_left, key=lambda k: k["center"][0])
                qa_pairs.append({
                    "question": f"Which kart is immediately to the left of {kart['kart_name']}?",
                    "answer": closest_left["kart_name"]
                })
                
            if karts_to_right:
                closest_right = min(karts_to_right, key=lambda k: k["center"][0])
                qa_pairs.append({
                    "question": f"Which kart is immediately to the right of {kart['kart_name']}?",
                    "answer": closest_right["kart_name"]
                })
    
    # Generate multiple variations for each question
    return multiply_qa_variations(qa_pairs)


def find_image_file(info_file: str, view_index: int) -> str:
    """
    Find the corresponding image file for an info file and view index.
    Handles different file extensions and naming patterns.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze

    Returns:
        Path to the image file if found, None otherwise
    """
    info_path = Path(info_file)
    # Get the base name without the "_info" suffix
    # For example, for "00000_info.json", base_name will be "00000"
    base_name = info_path.stem.replace("_info", "")
    
    # Based on the file structure shown, the pattern is XXXXX_YY_im.jpg
    # Where XXXXX is the base name and YY is the view index with 2-digit padding
    
    # Try with the exact pattern matching the file structure
    pattern = f"{base_name}_{view_index:02d}_im.jpg"
    files = list(info_path.parent.glob(pattern))
    if files:
        return str(files[0])
    
    # If the pattern fails, try to be more flexible with extensions
    extensions = ['.jpg', '.png']
    for ext in extensions:
        pattern = f"{base_name}_{view_index:02d}_im{ext}"
        files = list(info_path.parent.glob(pattern))
        if files:
            return str(files[0])
    
    # Try with 1-digit (no padding) as a fallback
    for ext in extensions:
        pattern = f"{base_name}_{view_index}_im{ext}"
        files = list(info_path.parent.glob(pattern))
        if files:
            return str(files[0])
    
    # List all available image files for this base name
    image_files = []
    for ext in extensions:
        pattern = f"{base_name}_*_im{ext}"
        files = list(info_path.parent.glob(pattern))
        if files:
            image_files.extend(files)
    
    if image_files:
        print(f"Warning: Could not find exact view index {view_index}, available view indices are:")
        for img in image_files:
            _, img_view_index = extract_frame_info(str(img))
            print(f"  - {img.name} (view index: {img_view_index})")
        
        # Use the first available image as a last resort
        return str(image_files[0])
    
    # No image found
    return None


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    print(f"Checking QA pairs for {info_file} with view index {view_index}")
    
    # Find corresponding image file
    image_file = find_image_file(info_file, view_index)
    
    if not image_file:
        print(f"Error: No image file found for {info_file} with view index {view_index}")
        # List available view indices
        info_path = Path(info_file)
        base_name = info_path.stem.replace("_info", "")
        all_images = []
        for ext in ['.jpg', '.png']:
            pattern = f"{base_name}_*_im{ext}"
            all_images.extend(list(info_path.parent.glob(pattern)))
        
        if all_images:
            print(f"Available images for this info file:")
            for img in all_images:
                frame_id, img_view_index = extract_frame_info(str(img))
                print(f"  - {img.name} (view index: {img_view_index})")
            print("\nPlease specify one of these view indices.")
        else:
            print("No image files found at all for this info file.")
        return

    print(f"Found image file: {image_file}")

    try:
        # Read the info.json file - verify it exists and is valid
        with open(info_file) as f:
            info = json.load(f)
            
        # Verify the detections exist and have data for the specified view index
        if "detections" not in info:
            print(f"Warning: No 'detections' field found in {info_file}")
        elif view_index >= len(info["detections"]):
            print(f"Warning: View index {view_index} out of range for detections (max: {len(info['detections'])-1})")
            print(f"Available view indices: 0-{len(info['detections'])-1}")
            return
    except Exception as e:
        print(f"Error reading or parsing info file: {e}")
        return

    # Visualize detections
    try:
        print("Drawing detections...")
        annotated_image = draw_detections(image_file, info_file)

        # Display the image
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"Frame {extract_frame_info(image_file)[0]}, View {view_index}")
        plt.show()
    except Exception as e:
        print(f"Error visualizing detections: {e}")
    
    # Generate QA pairs
    try:
        print("Generating QA pairs...")
        qa_pairs = generate_qa_pairs(info_file, view_index)

        # Print QA pairs
        print("\nQuestion-Answer Pairs:")
        print("-" * 50)
        for qa in qa_pairs:
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}")
            print("-" * 50)
    except Exception as e:
        print(f"Error generating QA pairs: {e}")


def list_available_views(info_file: str):
    """
    List all available view indices for a given info file.
    
    Args:
        info_file: Path to the info.json file
    """
    info_path = Path(info_file)
    if not info_path.exists():
        print(f"Error: Info file {info_file} not found")
        return
        
    # Get the base name without the "_info" suffix
    base_name = info_path.stem.replace("_info", "")
    
    # Find all image files matching the pattern
    all_images = []
    for ext in ['.jpg', '.png']:
        pattern = f"{base_name}_*_im{ext}"
        all_images.extend(list(info_path.parent.glob(pattern)))
    
    if not all_images:
        print(f"No image files found for {info_file}")
        return
        
    print(f"Available view indices for {info_file}:")
    for img in sorted(all_images, key=lambda x: extract_frame_info(str(x))[1]):
        frame_id, view_index = extract_frame_info(str(img))
        print(f"  - {img.name} (view index: {view_index})")
    
    # Also check the detections in the info.json file
    try:
        with open(info_file) as f:
            info = json.load(f)
            
        if "detections" in info:
            num_views = len(info["detections"])
            print(f"\nInfo file has detections for {num_views} views (indices 0-{num_views-1})")
        else:
            print("\nWarning: No 'detections' field found in info file")
    except Exception as e:
        print(f"\nError reading info file: {e}")


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
        image_files = []
        for ext in ['.jpg', '.png']:
            pattern = f"{base_name}_*_im{ext}"
            image_files.extend(list(info_file.parent.glob(pattern)))
        
        for image_file in image_files:
            # Extract view index from image filename
            _, view_index = extract_frame_info(str(image_file))
            
            # Generate QA pairs for this view
            try:
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
                
                # Add to dataset
                qa_dataset.append({
                    "image_path": str(image_file),
                    "info_path": str(info_file),
                    "qa_pairs": qa_pairs
                })
                
                print(f"Processed {image_file.name}")
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
    
    # Save dataset to output file
    with open(output_file, 'w') as f:
        json.dump(qa_dataset, f, indent=2)
    
    print(f"Generated {len(qa_dataset)} QA samples saved to {output_file}")


def generate_balanced_qa_dataset(data_folder: str, output_file: str, samples_per_track: int = 1000):
    """
    Generate a balanced QA dataset with equal representation from all available tracks.
    
    Args:
        data_folder: Path to the folder containing the data
        output_file: Path to output JSON file to save the QA pairs
        samples_per_track: Number of samples to include per track (default: 1000)
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
            image_files = []
            for ext in ['.jpg', '.png']:
                pattern = f"{base_name}_*_im{ext}"
                image_files.extend(list(info_file.parent.glob(pattern)))
            
            for image_file in image_files:
                if track_samples >= samples_per_track:
                    break
                    
                # Extract view index from image filename
                _, view_index = extract_frame_info(str(image_file))
                
                # Generate QA pairs for this view
                try:
                    qa_pairs = generate_qa_pairs(str(info_file), view_index)
                    
                    # Add to dataset
                    qa_dataset.append({
                        "image_path": str(image_file),
                        "info_path": str(info_file),
                        "qa_pairs": qa_pairs
                    })
                    
                    track_samples += 1
                    print(f"Processed {image_file.name} - {track_samples}/{samples_per_track} for {track_name}")
                except Exception as e:
                    print(f"Error processing {image_file.name}: {e}")
                
            if track_samples >= samples_per_track:
                break
    
    # Save dataset to output file
    with open(output_file, 'w') as f:
        json.dump(qa_dataset, f, indent=2)
    
    print(f"Generated balanced dataset with {len(qa_dataset)} QA samples saved to {output_file}")


def process_all_views_dataset(data_folder: str, output_file: str):
    """
    Process all views for every info file and generate extensive QA pairs.
    
    Args:
        data_folder: Path to the folder containing the data
        output_file: Path to output JSON file to save the QA pairs
    """
    data_path = Path(data_folder)
    info_files = list(data_path.glob("**/*_info.json"))
    
    qa_dataset = []
    
    for info_file in info_files:
        print(f"Processing {info_file}")
        
        # Get the info data to determine how many views are available
        try:
            with open(info_file) as f:
                info = json.load(f)
            
            if "detections" in info:
                num_views = len(info["detections"])
                
                # Process each view
                for view_index in range(num_views):
                    # Find corresponding image file
                    image_file = find_image_file(str(info_file), view_index)
                    
                    if image_file:
                        # Generate basic QA pairs
                        basic_qa_pairs = generate_qa_pairs(str(info_file), view_index)
                        
                        # Generate additional questions with variations
                        extended_qa_pairs = multiply_qa_variations(basic_qa_pairs)
                        
                        # Add to dataset
                        qa_dataset.append({
                            "image_path": image_file,
                            "info_path": str(info_file),
                            "qa_pairs": extended_qa_pairs
                        })
                        
                        print(f"  - Generated {len(extended_qa_pairs)} QA pairs for view {view_index}")
                
                # If multiple views are available, generate multi-view questions
                if num_views > 1:
                    multi_view_qa = generate_multi_view_qa_pairs(str(info_file), list(range(num_views)))
                    
                    # For multi-view questions, duplicate them for each view's image
                    for view_index in range(num_views):
                        image_file = find_image_file(str(info_file), view_index)
                        
                        if image_file:
                            qa_dataset.append({
                                "image_path": image_file,
                                "info_path": str(info_file),
                                "qa_pairs": multi_view_qa
                            })
                            
                            print(f"  - Added {len(multi_view_qa)} multi-view QA pairs for view {view_index}")
        
        except Exception as e:
            print(f"Error processing {info_file}: {e}")
    
    # Save dataset to output file
    with open(output_file, 'w') as f:
        json.dump(qa_dataset, f, indent=2)
    
    # Count total QA pairs
    total_qa_pairs = sum(len(sample["qa_pairs"]) for sample in qa_dataset)
    
    print(f"Generated {len(qa_dataset)} samples with a total of {total_qa_pairs} QA pairs")
    print(f"Saved to {output_file}")


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


def main():
    """
    Main entry point for the script. Provides a command-line interface using the Fire library.
    
    Available commands:
    - check: Check QA pairs for a specific file and view index
    - process: Process the entire dataset and generate QA pairs
    - balanced: Generate a balanced QA dataset with equal representation from all tracks
    - process_all: Process all views for every info file and generate extensive QA pairs
    - format: Format the QA dataset for VLM training
    - list_views: List all available view indices for a given info file
    
    Examples:
    - python generate_qa.py check --info_file=../data/valid/00000_info.json --view_index=0
    - python generate_qa.py process --data_folder=../data/valid --output_file=qa_dataset.json
    - python generate_qa.py balanced --data_folder=../data/valid --output_file=balanced_qa_dataset.json --samples_per_track=1000
    - python generate_qa.py process_all --data_folder=../data/valid --output_file=all_views_qa_dataset.json
    - python generate_qa.py format --qa_dataset_path=qa_dataset.json --output_file=vlm_dataset.json
    - python generate_qa.py list_views --info_file=../data/valid/00000_info.json
    """
    try:
        fire.Fire({
            "check": check_qa_pairs,
            "process": process_dataset,
            "balanced": generate_balanced_qa_dataset,
            "process_all": process_all_views_dataset,
            "format": format_for_vlm_training,
            "list_views": list_available_views
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()