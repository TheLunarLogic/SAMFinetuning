# 🧠 Fine-Tuning SAM (Segment Anything Model) - Beginner Friendly Guide

## 📌 What is SAM?

The **Segment Anything Model (SAM)** is a powerful image segmentation model developed by Meta AI. It is designed to generalize across a wide range of segmentation tasks without requiring task-specific training. SAM uses a **promptable design**, where it can segment objects based on different types of inputs like:

- Points
- Boxes
- Text prompts

### 🔧 Technical Architecture

- **Backbone:** Vision Transformer (ViT)
- **Inputs:** Image + Prompt(optional) (e.g., point, box)
- **Outputs:** Segmentation mask(s)
- **Core Idea:** Predicts masks corresponding to prompts using learned visual embeddings and attention mechanisms.

SAM is trained on a large and diverse dataset with over **1 billion masks** and performs zero-shot segmentation. However, its general nature might not perform optimally on **domain-specific** or **fine-grained** tasks.

---

## ❓ Why Fine-Tune SAM?

Despite SAM's strong zero-shot capabilities, **fine-tuning SAM** is necessary when:

- Working with **specialized domains** (e.g., medical, industrial, aerial imagery).
- Needing **higher accuracy** on specific classes or fine-grained objects.
- Improving **IoU scores** on your custom dataset.

Fine-tuning adapts SAM’s powerful general features to your task-specific dataset, significantly boosting performance.

---

## 🎯 Goal
Fine-tune the SAM (Segment Anything Model) using your custom dataset annotated in **YOLO** or **COCO** format.

---

## 🔄 Step-by-Step Workflow

### 🔹 1. Prepare Dataset (YOLO or COCO Format)

#### ✅ A. Using YOLO Polygon Format

- **Script:** `process_yolo_to_classwise.py`
- **Input:**
  - `images/`: Your raw images  
  - `labels/`: YOLO polygon annotations  
- **Functionality:**
  - Converts normalized polygon coordinates to pixel coordinates  
  - Creates binary masks using OpenCV  
  - Crops objects from images and masks  
  - Saves them to:

```
processed_dataset/
  └── Class_<name>/
      ├── images/
      └── masks/
```

> ⚠️ Make sure the dataset directory contains `data.yaml` file otherwise refer kaggle file ( `SAM FINE TUNING (with YOLO-DATASET) `).

---

#### ✅ B. Using COCO Format

- **Script:** `process_coco_dataset.py`
- **Input:**
  - one COCO JSON annotation files (e.g., `coco.json`)  
  - Corresponding images  
- **Functionality:**  
  - Creates binary masks from polygons  
  - Crops objects using bounding boxes and padding  
  - Saves them to:

```
processed_dataset/
  └── Class_<id>_<name>/
      ├── images/
      └── masks/
```

---

### 🔹 2. Generate Representative Points

#### ✅ A. For COCO-Based Dataset

- **Script:** `Create_csv.py`
- **Functionality:**
  - skip classes with with no instances  
  - Picks 3 key points:
    - Center of the mask  
    - Two equidistant/opposite points  
  - **Output:**
    - `train.csv`  
    - `val.csv`   
    - `test.csv` 

#### ✅ B. For YOLO-Based Dataset

- **Script:** `Create_yolo_csv.py`
- **Functionality:**
  - Selects 3 key points:
    - One near the center  
    - Two farthest points  
  - **Output:** `output.csv` (can be split into train/val/test)

---

### 🔹 3. Format CSVs for Training

- **Script:** `reformated_csv.py`
- **Functionality:**
  - Converts separate point columns into a list of three [x, y] pairs  
  - **New CSV Format:**
    ```
    image, label, points
    "img.jpg", "mask.png", "[[x1,y1],[x2,y2],[x3,y3]]"
    ```
  - **Output:**
    - `train_reformatted.csv`  
    - `val_reformatted.csv`  
    - `test_reformatted.csv`

---

### 🔹 4. Train SAM

- **Optional Args:** `--batch_size`, `--learning_rate`, `--save_dir`, `--num_epochs`  etc
- **Output:**
  - Checkpoints: `./checkpoints/` (Top 3 models)  
  - Final Model: `sam_lora_final.ckpt`  
  - Logs: Training loss, IoU per epoch

---


### 🔹 5. Run Inference

- **Scripts:**
  - `run_inference.py` — for fine-tuned model  
  - `run_base_sam_inference.py` — for pretrained model  

- **Output:**
  ```
  finetuned_sam_results2/
    ├── predictions.jpg
    ├── image + prediction + ground truth
  ```
  - CSVs:
    - `finetuned_sam_results.csv` — Sample-wise results  
    - `finetuned_sam_class_iou_summary.csv` — Class-wise IoUs

---
## 📆 Summary Pipeline Flow

![SAM Pipeline](https://raw.githubusercontent.com/TheLunarLogic/image/main/SAM%20diagram.png)

---

