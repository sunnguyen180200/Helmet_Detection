import csv
import json
from collections import defaultdict
import os
def csv_to_coco(csv_file, json_file):
    images = []
    annotations = []
    categories = {}
    labels = ["DNoHelmetP1NoHelmetP2NoHelmet", "DNoHelmet", "DNoHelmetP0NoHelmetP1NoHelmet", "DNoHelmetP0NoHelmet", "DHelmetP1Helmet", "DHelmet"]
    categories_dict = {
           "DNoHelmetP1NoHelmetP2NoHelmet": 0,
           "DNoHelmet": 0,
           "DNoHelmetP0NoHelmetP1NoHelmet": 0,
           "DNoHelmetP0NoHelmet": 0,
           "DHelmetP1Helmet": 1,
           "DHelmet": 1,
        }
    image_id_map = {}
    image_id_counter = 1
    annotation_id = 1
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            frame_id = row['frame_id']
            if frame_id in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                frame_id = "0"+frame_id
            x, y, w, h = map(float, [row['x'], row['y'], row['w'], row['h']])
            label = row['label']
            
            if frame_id not in image_id_map:
                image_id_map[frame_id] = image_id_counter
                images.append({
                    "id": image_id_counter,
                    "file_name": f"{frame_id}.jpg",
                    "width": 1920,
                    "height": 1080
                })
                image_id_counter += 1
            if label  in labels:
                if label not in categories:
                    categories[label] = categories_dict[label]
                    
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id_map[frame_id],
                    "category_id": categories[label],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1
        
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"COCO JSON saved to {json_file}")

# Gọi hàm để chuyển đổi
# csv_to_coco('Bago_highway_1.csv', 'annotations_coco.json')


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
    
def check_label(path_file_csv, categories_list):
    with open(path_file_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            if label not in categories_list:
                categories_list.append(label)
    return categories_list
    
# end def
def main():
    path_img_folder = "D:/2025/yolo/part_1/part_1/"
    path_csv_folder = "D:/2025/yolo/annotation/annotation/"
    path_json_folder = "D:/2025/yolo/main_src/data/annotation/"
    _, list_img_folders = list_files_and_folders(path_img_folder)
    categories_list = []
    for file_name in list_img_folders:
        path_file_csv = os.path.join(path_csv_folder, f"{file_name}.csv")
        csv_to_coco(path_file_csv, os.path.join(path_json_folder, f"{file_name}.json"))
        # categories_list = check_label(path_file_csv, categories_list)

    
if __name__ == '__main__':
    main() 