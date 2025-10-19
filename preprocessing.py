import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, class_weight

# ----------------------------
# Step 1: Define dataset paths
# ----------------------------
dataset_dir = r"D:\brh35\BrainTumor17Class"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# ----------------------------
# Step 2: Parameters
# ----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

# ----------------------------
# Step 3: Load images and labels
# ----------------------------
def load_dataset(data_dir):
    images = []
    labels = []
    classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]
    
    for cls in classes:
        cls_folder = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            try:
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(cls)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    images = np.array(images, dtype="float32")
    labels = np.array(labels)
    images, labels = shuffle(images, labels, random_state=42)
    return images, labels, classes

print("Loading training data...")
X_train, y_train, class_names = load_dataset(train_dir)
print("Loading testing data...")
X_test, y_test, _ = load_dataset(test_dir)

# ----------------------------
# Step 4: Encode labels
# ----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# ----------------------------
# Step 5: Balance training data via augmentation
# ----------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Count samples per class
unique, counts = np.unique(y_train_enc, return_counts=True)
max_count = counts.max()  # target number of images per class
X_augmented = []
y_augmented = []

for cls_idx, count in zip(unique, counts):
    if count < max_count:
        # Identify images of this class
        cls_indices = np.where(y_train_enc == cls_idx)[0]
        X_cls = X_train[cls_indices]
        y_cls = y_train_enc[cls_indices]
        
        # Augment until reaching max_count
        while len(X_cls) + len(X_augmented) < max_count:
            for img in X_cls:
                img_exp = np.expand_dims(img, 0)
                for batch in datagen.flow(img_exp, batch_size=1):
                    X_augmented.append(batch[0])
                    y_augmented.append(cls_idx)
                    if len(X_cls) + len(X_augmented) >= max_count:
                        break
                if len(X_cls) + len(X_augmented) >= max_count:
                    break

# Combine augmented data with original
X_train_balanced = np.concatenate([X_train, np.array(X_augmented)], axis=0)
y_train_balanced = np.concatenate([y_train_enc, np.array(y_augmented)], axis=0)

# Shuffle balanced dataset
X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=42)

# ----------------------------
# Step 6: Save preprocessed data
# ----------------------------
np.save(os.path.join(dataset_dir, "X_train_balanced.npy"), X_train_balanced)
np.save(os.path.join(dataset_dir, "y_train_balanced.npy"), y_train_balanced)
np.save(os.path.join(dataset_dir, "X_test.npy"), X_test)
np.save(os.path.join(dataset_dir, "y_test.npy"), y_test_enc)

# ----------------------------
# Step 7: Summary
# ----------------------------
print(f"Number of classes: {len(class_names)}")
print(f"Original training samples: {X_train.shape[0]}")
print(f"Balanced training samples: {X_train_balanced.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("Preprocessing and balancing complete!")
