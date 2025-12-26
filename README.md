 # Huawei-Cloud-Retina-projects
## Retinal Disease Multi-Label Classification (Huawei ModelArts Version)
This repository contains a PyTorch-based deep learning pipeline designed to detect multiple eye conditions from retinal fundus images. The model is specifically optimized to run on Huawei Cloud ModelArts, handling local data pathing and manual weight loading required for isolated cloud environments.

### Model Overview:
The script performs multi-label classification (detecting one or more conditions simultaneously) for four specific categories:
* Cataract
* Glaucoma
* Diabetic Retinopathy
* Normal Fundus

### Technical Specifications:
* **Architecture:** EfficientNet-B4 (via the timm library).
* **Input Resolution:** $380 \times 380$ pixels (optimized for EfficientNet-B4 scaling).
* **Loss Function:** BCEWithLogitsLoss (ideal for multi-label tasks where classes are not mutually exclusive).
* **Optimization:** Adam Optimizer with a ReduceLROnPlateau scheduler monitoring Macro AUC.
* **Augmentation:** Includes horizontal flips and random rotations to improve model generalization.

### Performance Metric:
The model evaluates success based on the Macro-averaged ROC-AUC score, ensuring that the model performs well across all four disease categories regardless of class imbalance.

### CSV Structure Requirements:
The training script expects a CSV file named `labels_image_centric_4_class_clean.csv` located in your data directory. It must contain the following columns:
| Column Name | Description |
| :--- | :--- |
| **Image_File** | The name of the image file (e.g., `img_001.jpg`). |
| **cataract** | Binary label (1 for present, 0 for absent). |
| **glaucoma** | Binary label (1 for present, 0 for absent). |
| **diabetic_retinopathy** | Binary label (1 for present, 0 for absent). |
| **normal_fundus** | Binary label (1 for present, 0 for absent). |

## External Eye Disease Classification (Selfie Model)
### Model Overview:
Unlike fundus models that require specialized cameras to see the back of the eye, this model uses images of the external eye structures (eyelids, conjunctiva, iris, and lens). It performs multi-class classification for five categories:
* **Cataract:** Clouding of the eye's lens.
* **Conjunctivitis:** Inflammation of the outer layer of the eye (pink eye).
* **Eyelid Conditions:** Issues affecting the surrounding eye tissue (e.g., blepharitis or styes).
* **Normal:** Healthy external eye appearance.
* **Uveitis:** Inflammation of the middle layer of the eye, often visible as deep redness around the iris.

### Technical Improvements in V3:
* **Backbone:** Uses EfficientNet-B4 with input resolution $380 \times 380$.
* **Automated Labeling:** Uses datasets.ImageFolder to automatically map folder names (cataract, uveitis, etc.) to class labels, making it easier to add new data.
* **Comprehensive Metrics:** After every epoch, the script generates a detailed Scikit-Learn Classification Report, providing precision, recall, and F1-scores for every individual disease category.
* **ModelArts Optimization:** * Loads local weights from pytorch_model.bin in the data directory to bypass cloud download restrictions.
  * Saves the best-performing model as best_selfie_model_v3.pth.

### Dataset Structure: Selfie Model
The selfie model uses an `ImageFolder` structure. Organize your images into subfolders named after each category:

```text
/input_data_0/
├── training/
│   ├── cataract/          # Contains .jpg/.png images
│   ├── conjunctivitis/    # Contains .jpg/.png images
│   ├── eyelid/            # Contains .jpg/.png images
│   ├── normal/            # Contains .jpg/.png images
│   └── uveitis/           # Contains .jpg/.png images
└── validation/
    ├── cataract/
    ├── conjunctivitis/
    ├── eyelid/
    ├── normal/
    └── uveitis/
```
## Training on Huawei Cloud ModelArts
Both models are optimized for ModelArts using the Custom Algorithm training mode. Follow these steps to set up and execute your training job.

### 1. Data & Script Preparation (OBS):
Before creating the job, upload your assets to an Object Storage Service (OBS) bucket. Organize your folders as follows:
* **Code Folder:** Upload train.py, train_selfie_v3.py, and pytorch_model.bin (pre-trained weights).
* **Data Folder:** Upload your training/validation images and the CSV label file.
  
### 2. Creating the Training Job:
In the ModelArts console, navigate to Training Management > Training Jobs and click Create.
**A. Algorithm Configuration:**
Creation Mode: Select ```Custom algorithm```.
* Boot File: Select the script for the model you want to train:
  * **For Retina:** ```obs://your-bucket/code/train.py```
  * **For Selfie:** ```obs://your-bucket/code/train_selfie_v3.py```
* Code Directory: Select the folder in OBS containing your scripts and weights.
* Engine: Choose a preset PyTorch engine (e.g., ```PyTorch-1.10``` or higher).
**B. Input & Output Parameters:**
  ModelArts maps OBS paths to local container paths automatically. Ensure your job parameters match the script logic:
  
| Parameter Name | OBS Path | Local Container Path (Mapped) |
| :--- | :--- |:--- |
| **data_url** | ```obs://your-bucket/data/```. | ```/home/ma-user/modelarts/inputs/input_data_0/```. |
| **train_url** | ```obs://your-bucket/output/```. | ```/home/ma-user/modelarts/outputs/model_output_0/```. |

**[!IMPORTANT] The scripts are hardcoded to look for data in /home/ma-user/modelarts/inputs/input_data_0/. Ensure your OBS "Input" configuration points to the folder containing your images and labels.**
