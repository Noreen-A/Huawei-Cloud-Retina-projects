
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import timm 
# --- NEW IMPORT ---
from sklearn.metrics import classification_report
# ------------------

print("--- RUNNING SCRIPT VERSION V3 (Selfie Model + Full Report) ---")

# --- 1. CONFIGURATION FOR HUAWEI CLOUD MODELARTS ---
DATA_DIR = '/home/ma-user/modelarts/inputs/input_data_0/' 
MODEL_SAVE_DIR = '/home/ma-user/modelarts/outputs/model_output_0/' 

# --- CRITICAL FIX: Look for the weights file inside the DATA_DIR ---
WEIGHTS_FILE = os.path.join(DATA_DIR, 'pytorch_model.bin')
# ---------------------------------------------------

# --- 2. DEFINE FILE PATHS BASED ON THE ABOVE ---
TRAIN_DIR = os.path.join(DATA_DIR, 'training')
VAL_DIR = os.path.join(DATA_DIR, 'validation')

NUM_CLASSES = 5 # cataract, conjunctivitis, eyelid, normal, uveitis
IMAGE_SIZE = 380 # EfficientNet-B4 size

# --- Step 1: Define Transforms ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- Step 2: Build the Model ---
def get_model(num_classes):
    model = timm.create_model('efficientnet_b4', pretrained=False)
    
    if os.path.exists(WEIGHTS_FILE):
        print(f"Loading pretrained weights from {WEIGHTS_FILE}...")
        state_dict = torch.load(WEIGHTS_FILE, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    else:
        print(f"CRITICAL ERROR: Weights file not found at {WEIGHTS_FILE}")
        raise FileNotFoundError(f"Weights file not found at {WEIGHTS_FILE}")

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# --- Step 3: Main Training Function ---
def train_model():
    print("--- Starting PyTorch Multi-Class Training on ModelArts ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Check if data files exist
    print(f"Looking for data in: {DATA_DIR}")
    try:
        print(f"Contents of {DATA_DIR}: {os.listdir(DATA_DIR)}")
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot list contents of {DATA_DIR}. Error: {e}")
        return
        
    if not os.path.exists(TRAIN_DIR):
        print(f"CRITICAL ERROR: Training folder not found at {TRAIN_DIR}.")
        return
    if not os.path.exists(VAL_DIR):
        print(f"CRITICAL ERROR: Validation folder not found at {VAL_DIR}.")
        return
    print("All data paths found successfully.")

    # 2. Instantiate DataLoaders using ImageFolder
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    class_names = train_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # 3. Build Model
    model = get_model(len(class_names)).to(device) # Use found classes

    # 4. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training Loop
    num_epochs = 25
    best_val_accuracy = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_selfie_model_v3.pth') # New save name

    print(f"Model will be saved to: {best_model_path}")
    print("\n--- Starting Training Loop ---")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if images.shape[0] == 0: continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- MODIFIED VALIDATION PHASE ---
        model.eval()
        correct = 0
        total = 0
        all_val_labels = [] # To store all true labels
        all_val_preds = [] # To store all predicted labels
        
        with torch.no_grad():
            for images, labels in val_loader:
                if images.shape[0] == 0: continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Add batch results to our master lists
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
        
        val_accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%')

        # --- NEW: PRINT THE FULL REPORT ---
        print(f"\n--- Classification Report for Epoch {epoch+1} ---")
        if total > 0:
            try:
                report = classification_report(
                    all_val_labels, 
                    all_val_preds, 
                    target_names=class_names, 
                    digits=4
                )
                print(report)
            except Exception as e:
                print(f"Could not generate classification report: {e}")
        else:
            print("No validation samples processed, skipping report.")
        # ------------------------------------

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'   ---> Model saved. New best accuracy: {best_val_accuracy:.2f}%')

    print(f'Training Complete. Best Validation Accuracy: {best_val_accuracy:.2f}%. Model saved to {best_model_path}')

if __name__ == '__main__':
    train_model()
