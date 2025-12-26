
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import timm 

print("--- RUNNING SCRIPT VERSION V7 (Weights in DATA folder) ---")

# --- 1. CONFIGURATION FOR HUAWEI CLOUD MODELARTS ---
DATA_DIR = '/home/ma-user/modelarts/inputs/input_data_0/' 
MODEL_SAVE_DIR = '/home/ma-user/modelarts/outputs/model_output_0/' 

# --- CRITICAL FIX: Look for the weights file inside the DATA_DIR ---
# This path is now inside the data folder, which we know is downloading.
WEIGHTS_FILE = os.path.join(DATA_DIR, 'pytorch_model.bin')
# ---------------------------------------------------


# --- 2. DEFINE FILE PATHS BASED ON THE ABOVE ---
LABEL_FILE = os.path.join(DATA_DIR, 'labels_image_centric_4_class_clean.csv')
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'Training Images') 
IMAGE_DIR_TEST = os.path.join(DATA_DIR, 'Testing Images')   
# ---------------------------------------------------


# Label Columns (MUST match your preprocessed CSV)
LABEL_COLUMNS = ['cataract', 'glaucoma', 'diabetic_retinopathy', 'normal_fundus']
NUM_CLASSES = len(LABEL_COLUMNS)

# --- Step 1: Define Multi-Label Dataset Class ---
class RetinalMultiLabelDataset(Dataset):
    def __init__(self, df, image_dir_train, image_dir_test, transform=None):
        self.labels_df = df
        self.image_dir_train = image_dir_train # Path to training data
        self.image_dir_test = image_dir_test   # Path to testing data
        self.transform = transform
        self.labels = self.labels_df[LABEL_COLUMNS].values
        self.image_names = self.labels_df['Image_File'].values

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Check both possible paths for the image (assuming data might be split across two folders)
        img_path_train = os.path.join(self.image_dir_train, img_name)
        img_path_test = os.path.join(self.image_dir_test, img_name)
        
        img_path = None
        if os.path.exists(img_path_train):
            img_path = img_path_train
        elif os.path.exists(img_path_test):
            img_path = img_path_test
        else:
            # Check for common case sensitivity issues (.jpg vs .jpeg)
            img_name_jpeg = img_name.replace('.jpg', '.jpeg')
            img_path_train_jpeg = os.path.join(self.image_dir_train, img_name_jpeg)
            img_path_test_jpeg = os.path.join(self.image_dir_test, img_name_jpeg)
            
            if os.path.exists(img_path_train_jpeg):
                img_path = img_path_train_jpeg
            elif os.path.exists(img_path_test_jpeg):
                img_path = img_path_test_jpeg
            else:
                print(f"Warning: Image file not found: {img_name}. Skipping.")
                # Return a blank image and placeholder label
                return torch.zeros((3, 380, 380)), torch.zeros((NUM_CLASSES,), dtype=torch.float32)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading {img_path}. Error: {e}. Skipping.")
            return torch.zeros((3, 380, 380)), torch.zeros((NUM_CLASSES,), dtype=torch.float32)

        labels = torch.tensor(self.labels[idx].astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, labels

# --- Step 2: Define Transforms ---
mean = [0.485, 0.456, 0.406]
std = [0.299, 0.224, 0.225] 

train_transform = transforms.Compose([
    transforms.Resize((380, 380)), # EfficientNet-B4 size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- Step 3: Build the Model ---
# --- CRITICAL EDIT: Loading weights from our local file ---
def get_model(num_classes):
    # 1. Create the model structure WITHOUT downloading weights
    model = timm.create_model('efficientnet_b4', pretrained=False)
    
    # 2. Manually load our downloaded weights
    if os.path.exists(WEIGHTS_FILE):
        print(f"Loading pretrained weights from {WEIGHTS_FILE}...")
        # Load the weights. We use map_location='cpu' as a safety measure
        state_dict = torch.load(WEIGHTS_FILE, map_location='cpu')
        
        # We use strict=False because our classifier head is new
        # and won't match the weights file, which is expected.
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    else:
        # This will stop the job if the weights file is missing
        print(f"CRITICAL ERROR: Weights file not found at {WEIGHTS_FILE}")
        raise FileNotFoundError(f"Weights file not found at {WEIGHTS_FILE}")

    # 3. Replace the final classifier layer for our 4 classes
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model
# -------------------------------------------------------------------

# --- Step 4: Main Training Function ---
def train_model():
    print("--- Starting PyTorch Multi-Label Training on ModelArts (using timm) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Check if data files exist
    print(f"Checking for data in {DATA_DIR}...")
    try:
        print(f"Contents of {DATA_DIR}: {os.listdir(DATA_DIR)}")
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot list contents of {DATA_DIR}. Error: {e}")
        return
        
    if not os.path.exists(LABEL_FILE):
        print(f"CRITICAL ERROR: Label file not found at {LABEL_FILE}.")
        return
    if not os.path.exists(IMAGE_DIR_TRAIN):
        print(f"CRITICAL ERROR: Training Image folder not found at {IMAGE_DIR_TRAIN}.")
        return
    if not os.path.exists(IMAGE_DIR_TEST):
        print(f"CRITICAL ERROR: Testing Image folder not found at {IMAGE_DIR_TEST}.")
        return
    print("All data paths found successfully.")

    # 2. Load and Split Data
    df_labels = pd.read_csv(LABEL_FILE)
    df_train, df_val = train_test_split(df_labels, test_size=0.20, random_state=42)
    print(f"Total records: {len(df_labels)}, Training: {len(df_train)}, Validation: {len(df_val)}")

    # 3. Instantiate DataLoaders
    train_dataset = RetinalMultiLabelDataset(df_train, IMAGE_DIR_TRAIN, IMAGE_DIR_TEST, transform=train_transform)
    val_dataset = RetinalMultiLabelDataset(df_val, IMAGE_DIR_TRAIN, IMAGE_DIR_TEST, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 4. Build Model
    model = get_model(NUM_CLASSES).to(device)

    # 5. Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss() # Correct for multi-label
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 6. Training Loop
    num_epochs = 25
    best_val_auc = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_retinal_model.pth')

    print(f"Model will be saved to: {best_model_path}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if images.shape[0] == 0:
                print("Skipping empty batch.")
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        if len(train_dataset) == 0:
             print("Training dataset is empty. Cannot calculate epoch loss.")
             continue
        
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                if images.shape[0] == 0:
                    print("Skipping empty val batch.")
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.sigmoid(outputs)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(probabilities.cpu().numpy())
        
        if not all_labels:
            print("Validation set is empty. Cannot calculate AUC.")
            continue

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        val_auc = roc_auc_score(all_labels, all_preds, average='macro')
        
        scheduler.step(val_auc)
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Val AUC (Macro): {val_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f'   ---> Model saved. New best AUC: {best_val_auc:.4f}')

    print(f'Training Complete. Best Validation AUC: {best_val_auc:.4f}. Model saved to {best_model_path}')

if __name__ == '__main__':
    try:
        print("Checking installed packages...")
        os.system("pip list")
    except Exception as e:
        print(f"Could not list packages: {e}")
        
    train_model()
