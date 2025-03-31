import csv
import json
import cv2
import os
from collections import defaultdict



def csv_to_coco(csv_file, json_file, image_folder, output_folder):
    images = []
    annotations = []
    categories = {}
    image_id_map = {}
    image_id_counter = 1
    annotation_id = 1
    
    os.makedirs(output_folder, exist_ok=True)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            frame_id = row['frame_id']
            x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
            label = row['label']
            
            image_path = os.path.join(image_folder, f"{frame_id}.jpg")
            if not os.path.exists(image_path):
                continue
            
            if frame_id not in image_id_map:
                image_id_map[frame_id] = image_id_counter
                img = cv2.imread(image_path)
                height, width, _ = img.shape
                
                images.append({
                    "id": image_id_counter,
                    "file_name": f"{frame_id}.jpg",
                    "width": width,
                    "height": height
                })
                image_id_counter += 1
            
            if label not in categories:
                categories[label] = len(categories) + 1
                
            annotations.append({
                "id": annotation_id,
                "image_id": image_id_map[frame_id],
                "category_id": categories[label],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            
            img = cv2.imread(image_path)
            cropped = img[y:y+h, x:x+w]
            cropped_path = os.path.join(output_folder, f"{frame_id}_{annotation_id}.jpg")
            cv2.imwrite(cropped_path, cropped)
            
            annotation_id += 1
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"COCO JSON saved to {json_file}")
    print(f"Cropped images saved to {output_folder}")




# Gọi hàm để chuyển đổi
# csv_to_coco('Bago_highway_1.csv', 'annotations_coco.json', 'Bago_highway_1', 'cropped_boxes')

import json
import cv2
import os

def crop_coco_images(coco_json_path, images_dir, output_dir, category_id=2):
    """
    Extracts and saves cropped images of objects with a specified category ID from a COCO dataset.
    
    :param coco_json_path: Path to the COCO JSON annotation file.
    :param images_dir: Directory containing the original images.
    :param output_dir: Directory to save cropped images.
    :param category_id: Category ID to filter objects (default: 4).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img["id"]: img["file_name"] for img in coco_data["images"]}
    
    for ann in coco_data["annotations"]:
        if ann["category_id"] == category_id:
            image_id = ann["image_id"]
            bbox = ann["bbox"]  # Format: [x, y, width, height]
            
            if image_id in images:
                img_path = os.path.join(images_dir, images[image_id])
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                x, y, w, h = map(int, bbox)
                cropped_img = img[y:y+h, x:x+w]
                
                output_path = os.path.join(output_dir, f"crop_{image_id}_{ann['id']}.jpg")
                cv2.imwrite(output_path, cropped_img)
                print(f"Saved: {output_path}")

# Example usage
# crop_coco_images("processing_data/annotations_coco.json", "processing_data/Bago_highway_1", "processing_data/cropped")
def list_files_and_folders(folder_path):
    try:
        items = os.listdir(folder_path)
        files = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]
        folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        
        return files, folders
    except FileNotFoundError:
        print("Thư mục không tồn tại!")
        return [], []
    except PermissionError:
        print("Không có quyền truy cập thư mục!")
        return [], []
def main():
    path_img_folder = "D:/2025/yolo/main_src/data/part_1/part_1"
    path_csv_folder = "D:/2025/yolo/annotation/annotation/"
    path_json_folder = "D:/2025/yolo/main_src/data/annotation/"
    _, list_img_folders = list_files_and_folders(path_img_folder)
    for file_name in list_img_folders:
        # path_file_csv = os.path.join(path_csv_folder, f"{file_name}.csv")
        crop_coco_images(os.path.join(path_json_folder, f"{file_name}.json"), 
                                       os.path.join(path_img_folder,f"{file_name}"), 
                                        "D:/2025/yolo/main_src/processing_data/2",
                                        category_id=1)
    
if __name__ == '__main__':
    main() 