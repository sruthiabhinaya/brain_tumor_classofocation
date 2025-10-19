import gradio as gr
import numpy as np
import joblib
import os
from tensorflow.keras.applications import VGG16, ResNet101, InceptionV3, DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# ----------------------------
# Step 1: Load models
# ----------------------------
dataset_dir = r"D:\brh35\BrainTumor17Class"

xgb_clf = joblib.load(os.path.join(dataset_dir, "xgboost_classifier.pkl"))
le = joblib.load(os.path.join(dataset_dir, "label_encoder.pkl"))

selector_path = os.path.join(dataset_dir, "feature_selector.pkl")
selector = joblib.load(selector_path) if os.path.exists(selector_path) else None

# ----------------------------
# Step 2: Define CNN feature extractors
# ----------------------------
def get_feature_extractor(model_class, input_shape=(224,224,3)):
    base_model = model_class(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

vgg_model = get_feature_extractor(VGG16)
resnet_model = get_feature_extractor(ResNet101)
inception_model = get_feature_extractor(InceptionV3)
densenet_model = get_feature_extractor(DenseNet201)

# ----------------------------
# Step 3: Preprocess uploaded image
# ----------------------------
def preprocess_image(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize
    return img_array

# ----------------------------
# Step 4: Extract and fuse features
# ----------------------------
def extract_features(img_array):
    vgg_feat = vgg_model.predict(img_array)
    resnet_feat = resnet_model.predict(img_array)
    inception_feat = inception_model.predict(img_array)
    densenet_feat = densenet_model.predict(img_array)
    fused = np.hstack([vgg_feat, resnet_feat, inception_feat, densenet_feat])
    
    if selector:
        fused = selector.transform(fused)
    return fused

# ----------------------------
# Step 5: Prediction function
# ----------------------------
def predict_tumor(img):
    try:
        img_array = preprocess_image(img)
        fused_feat = extract_features(img_array)
        pred_class_idx = xgb_clf.predict(fused_feat)[0]
        pred_label = le.inverse_transform([pred_class_idx])[0]
        return f"Tumor type: {pred_label}"
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# Step 6: Launch Gradio
# ----------------------------
iface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs="text",
    title="Brain Tumor Classifier (Upload MRI Image)",
    description="Upload a new MRI scan to predict the tumor type instantly."
)

iface.launch()
