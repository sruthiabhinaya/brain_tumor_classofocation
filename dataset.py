import zipfile
import os
import shutil
from sklearn.model_selection import train_test_split

# ----------------------------
# Step 1: Define paths
# ----------------------------
# Use raw strings (r"") to avoid Windows escape sequence issues
zip_path = r"D:\brh35\archive (10).zip"        # Path to your uploaded ZIP
dataset_dir = r"D:\brh35\BrainTumor17Class"    # Folder to extract dataset
os.makedirs(dataset_dir, exist_ok=True)

# ----------------------------
# Step 2: Extract ZIP file
# ----------------------------
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)
print("Extraction complete!")

# ----------------------------
# Step 3: Organize train/test folders
# ----------------------------
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate through tumor type folders
for tumor_type in os.listdir(dataset_dir):
    tumor_path = os.path.join(dataset_dir, tumor_type)
    
    # Skip train/test folders themselves
    if os.path.isdir(tumor_path) and tumor_type not in ["train", "test"]:
        # Create subfolders in train/test
        os.makedirs(os.path.join(train_dir, tumor_type), exist_ok=True)
        os.makedirs(os.path.join(test_dir, tumor_type), exist_ok=True)
        
        # List all images in this tumor folder
        images = os.listdir(tumor_path)
        
        # Split into 80% train / 20% test
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        # Copy images to train folder
        for img in train_imgs:
            shutil.copy(os.path.join(tumor_path, img), os.path.join(train_dir, tumor_type, img))
        # Copy images to test folder
        for img in test_imgs:
            shutil.copy(os.path.join(tumor_path, img), os.path.join(test_dir, tumor_type, img))

print("Train/Test split completed!")

# ----------------------------
# Step 4: Optional summary
# ----------------------------
print("\nDataset Summary:")
for folder, dir_path in zip(["Train", "Test"], [train_dir, test_dir]):
    print(f"\n{folder} folder:")
    for tumor_type in os.listdir(dir_path):
        tumor_path = os.path.join(dir_path, tumor_type)
        num_images = len(os.listdir(tumor_path))
        print(f"  {tumor_type}: {num_images} images")
