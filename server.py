import cv2
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.feature import local_binary_pattern
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load models
feature_extractor = load_model("resnet_feature_extractor.h5")
pca = joblib.load("pca_model.pkl")
svm = joblib.load("svm_model.pkl")

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# LBP Feature Extraction
def extract_lbp_features(image, P=8, R=1):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    return hist.reshape(1, -1)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    
    # Extract deep features
    img_array = preprocess_image(image)
    deep_features = feature_extractor.predict(preprocess_input(img_array * 255.0))

    # Extract LBP features
    lbp_features = extract_lbp_features(image)

    # Fuse features and apply PCA
    fused_features = np.concatenate([deep_features, lbp_features], axis=1)
    pca_features = pca.transform(fused_features)

    # Predict with SVM
    prediction = svm.predict(pca_features)
    
    return {"prediction": int(prediction[0])}  # Convert NumPy type to Python type

# Run server: uvicorn server:app --host 0.0.0.0 --port 8000
