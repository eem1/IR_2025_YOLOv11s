import cv2
import numpy as np
import os
import csv
import time
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# --- CONFIGURATION ---
# ==========================================
MANIFEST_FILE = "ir_manifest.txt"
MODEL_PATH = r"Y:\NMML_Polar\Analytics\IceSeal_DetectionModel\Yolov11_IR_2025\train\runs\detect\seals_run1\weights\best.pt"
STATS_LOG_FILE = "pipeline_run_stats.txt"

CONFIDENCE_THRESHOLD = 0.1 

# Normalization Settings
DIFF_THRESHOLD = 500 
LOWER_PERCENTILE = 0.001
UPPER_PERCENTILE = 99.999

# ==========================================
# --- IMAGE PROCESSING FUNCTIONS ---
# ==========================================
def despeckle_16bit(image, threshold):
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(-1) #strips trailing channel dimensions
    median_blurred = cv2.medianBlur(image, 3)
    diff = cv2.absdiff(image, median_blurred)
    bad_pixel_mask = diff > threshold
  # stabilize fix for the shape mismatch crash on early flight folders
    cleaned_img = np.where(bad_pixel_mask, median_blurred, image)
    return cleaned_img

def robust_normalize_to_8bit(image, low_p, high_p):
    p_low = np.nanpercentile(image, low_p)
    p_high = np.nanpercentile(image, high_p)
    img_clipped = np.clip(image, p_low, p_high)
    img_norm = (img_clipped - p_low) / (p_high - p_low + 1e-5)
    img_8bit = (img_norm * 255).astype(np.uint8)
    # YOLO expects 3 channels (RGB/BGR), so we duplicate the grayscale channel
    img_8bit_3c = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    return img_8bit_3c

# ==========================================
# --- MAIN PIPELINE ---
# ==========================================
def run_pipeline():
    # 1. Load the Model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 2. Check for Manifest
    if not os.path.exists(MANIFEST_FILE):
        print(f"Error: Manifest file '{MANIFEST_FILE}' not found.")
        return

    # Tracking stats across the entire run
    total_lists_processed = 0
    total_images_attempted = 0
    total_images_success = 0
    total_detections = 0
    failed_images = []

    # Read the manifest to get all image list files
    with open(MANIFEST_FILE, 'r', encoding='utf-8') as mf:
        list_files = [line.strip() for line in mf if line.strip()]

    print(f"Found {len(list_files)} image list files in the manifest.")

    # 3. Process Each Image List
    for list_file_path in list_files:
        list_path_obj = Path(list_file_path)
        
        if not list_path_obj.exists():
            print(f"Warning: List file not found, skipping -> {list_file_path}")
            continue
            
        total_lists_processed += 1
        
        # Determine CSV output path (replace _ir_images.txt or .txt with _ir_detections.csv)
        if list_path_obj.name.endswith('_ir_images.txt'):
            csv_name = list_path_obj.name.replace('_ir_images.txt', '_ir_detections.csv')
        else:
            csv_name = list_path_obj.stem + '_ir_detections.csv'
            
        csv_output_path = list_path_obj.parent / csv_name
        
        print(f"\nProcessing list: {list_path_obj.name}")
        print(f"Outputting detections to: {csv_output_path.name}")

        # Read the image paths from the current list file
        with open(list_path_obj, 'r', encoding='utf-8') as lf:
            image_paths = [line.strip() for line in lf if line.strip()]

        # Setup CSV for this specific list
        with open(csv_output_path, mode='w', newline='') as csv_file:
            # Write VIAME headers
            csv_file.write("# 1: Detection or Track-id, 2: Video or Image Identifier, 3: Unique Frame Identifier, 4-7: Img-bbox(TL_x, TL_y, BR_x, BR_y), 8: Detection or Length Confidence, 9: Target Length (0 or -1 if invalid), 10-11+: Repeated Species, Confidence Pairs or Attributes\n")
            current_time = time.ctime()
            csv_file.write(f"# metadata,exec_time: 0,exported_by: python_yolo_script,exported_at: {current_time},,,,,,,,\n")
            
            writer = csv.writer(csv_file)
            global_detection_id = 1

            # 4. Process Each Image in the List
            for frame_id, img_path_str in enumerate(image_paths):
                total_images_attempted += 1
                img_path = Path(img_path_str)
                
                # Load 16-bit thermal image
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    failed_images.append(str(img_path))
                    continue
                
                total_images_success += 1

                # --- In-Memory Preprocessing ---
                clean_16bit = despeckle_16bit(img, DIFF_THRESHOLD)
                final_8bit_3c = robust_normalize_to_8bit(clean_16bit, LOWER_PERCENTILE, UPPER_PERCENTILE)

                # --- In-Memory Inference ---
                # Pass the numpy array directly; save=False prevents disk writes; quiet processing with verbose=false eliminates per/frame reporting; force GPU useage with device=0
                # Dynamically turn verbose ON only every 50 frames
                show_details = (frame_id + 1) % 50 == 0
                results = model.predict(source=final_8bit_3c, save=False, conf=CONFIDENCE_THRESHOLD, verbose=show_details, device=0)
                result = results[0] # Single image prediction
                
                # Get the full absolute pathway to each image recorded in the dettections csv file
                full_image_path = str(img_path)

                # --- Log Detections ---
                for box in result.boxes:
                    total_detections += 1
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id] 
                    conf = float(box.conf[0])       
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    
                    writer.writerow([
                        global_detection_id,    
                        full_image_path,               
                        frame_id,               
                        f"{x_min:.3f}",         
                        f"{y_min:.3f}",         
                        f"{x_max:.3f}",         
                        f"{y_max:.3f}",         
                        f"{conf:.5f}",          
                        0,                      
                        cls_name,               
                        f"{conf:.5f}"           
                    ])
                    global_detection_id += 1

                # Optional: terminal progress indicator
                if (frame_id + 1) % 50 == 0:
                    print(f"  Processed {frame_id + 1}/{len(image_paths)} images...")

    # 5. Write Run Statistics Log
    with open(STATS_LOG_FILE, 'w', encoding='utf-8') as log_file:
        log_file.write("=== PIPELINE RUN STATISTICS ===\n")
        log_file.write(f"Date/Time: {time.ctime()}\n")
        log_file.write(f"Image Lists Processed: {total_lists_processed}\n")
        log_file.write(f"Total Images Attempted: {total_images_attempted}\n")
        log_file.write(f"Total Images Successfully Processed: {total_images_success}\n")
        log_file.write(f"Total Detections Logged: {total_detections}\n")
        log_file.write(f"Total Failed Images: {len(failed_images)}\n")
        
        if failed_images:
            log_file.write("\n--- FAILED IMAGES (Could not be read) ---\n")
            for fail_path in failed_images:
                log_file.write(f"{fail_path}\n")

    print(f"\nPipeline Complete!")
    print(f"Stats written to: {STATS_LOG_FILE}")
    if failed_images:
        print(f"Warning: {len(failed_images)} images failed to process. Check the stats log.")

if __name__ == "__main__":
    run_pipeline()