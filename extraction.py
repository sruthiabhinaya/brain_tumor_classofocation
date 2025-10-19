import numpy as np
from tensorflow.keras.applications import VGG16, ResNet101, InceptionV3, DenseNet201
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load preprocessed balanced data
# ----------------------------
dataset_dir = r"D:\brh35\BrainTumor17Class"
X_train = np.load(f"{dataset_dir}/X_train_balanced.npy")
y_train = np.load(f"{dataset_dir}/y_train_balanced.npy")
X_test = np.load(f"{dataset_dir}/X_test.npy")
y_test = np.load(f"{dataset_dir}/y_test.npy")

print("Data loaded successfully!")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ----------------------------
# Step 2: Load pre-trained CNNs for feature extraction
# ----------------------------
def get_feature_extractor(model_class, input_shape=(224,224,3)):
    base_model = model_class(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

# Instantiate models
print("Loading pre-trained models...")
vgg_model = get_feature_extractor(VGG16)
resnet_model = get_feature_extractor(ResNet101)
inception_model = get_feature_extractor(InceptionV3)
densenet_model = get_feature_extractor(DenseNet201)

# ----------------------------
# Step 3: Feature extraction
# ----------------------------
def extract_features(model, X):
    features = model.predict(X, batch_size=32, verbose=1)
    return features

print("Extracting VGG16 features...")
vgg_features_train = extract_features(vgg_model, X_train)
vgg_features_test = extract_features(vgg_model, X_test)

print("Extracting ResNet101 features...")
resnet_features_train = extract_features(resnet_model, X_train)
resnet_features_test = extract_features(resnet_model, X_test)

print("Extracting InceptionV3 features...")
inception_features_train = extract_features(inception_model, X_train)
inception_features_test = extract_features(inception_model, X_test)

print("Extracting DenseNet201 features...")
densenet_features_train = extract_features(densenet_model, X_train)
densenet_features_test = extract_features(densenet_model, X_test)

# ----------------------------
# Step 4: Feature fusion (concatenation)
# ----------------------------
from numpy import hstack

X_train_fused = hstack([vgg_features_train, resnet_features_train, inception_features_train, densenet_features_train])
X_test_fused = hstack([vgg_features_test, resnet_features_test, inception_features_test, densenet_features_test])

print(f"Fused feature shape: {X_train_fused.shape}")

np.save(f"{dataset_dir}/X_train_fused.npy", X_train_fused)
np.save(f"{dataset_dir}/X_test_fused.npy", X_test_fused)
print("Fused features saved successfully!")

