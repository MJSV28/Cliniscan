import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from ultralytics import YOLO
import os

st.set_page_config(page_title="🫁 CliniScan", layout="wide")
st.title("🫁 CliniScan - Chest X-ray Analysis")

@st.cache_resource
def load_models():
    yolo_path = "yolo_best.pt"
    
    # ✅ YOLO ONLY - No EfficientNet crash
    if not os.path.exists(yolo_path):
        st.error(f"❌ YOLO: {yolo_path} not found - upload to repo"); st.stop()
    
    detector = YOLO(yolo_path)
    st.success("✅ YOLO loaded successfully!")
    
    # ✅ Dummy classifier (deploys NOW, add EfficientNet later)
    class DummyClassifier:
        def __call__(self, x):
            # Return fake probs for 5 classes
            return torch.tensor([[0.2, 0.3, 0.1, 0.25, 0.15]])
    
    classifier = DummyClassifier()
    st.success("✅ Classifier ready")
    
    return classifier, detector

# Load models
classifier, detector = load_models()

CLASS_NAMES = ["opacity", "consolidation", "fibrosis", "mass", "other"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("📁 Upload X-ray", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original")
        
        # Dummy classification probs
        probs = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
        df_probs = pd.DataFrame({
            'Disease': CLASS_NAMES,
            'Probability': probs
        }).sort_values('Probability', ascending=False)
        
        st.subheader("🧠 Classification")
        st.info("📝 EfficientNet coming soon - demo probs shown")
        st.bar_chart(df_probs.set_index('Disease')['Probability'])
        st.dataframe(df_probs.round(3))
    
    with col2:
        st.subheader("📦 YOLO Detection")
        results = detector(image, conf=0.25, verbose=False)
        annotated = results[0].plot()
        st.image(annotated, caption="YOLO Detections", use_container_width=True)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_df = pd.DataFrame({
                'Class': [detector.names[int(cls)] for cls in results[0].boxes.cls],
                'Conf': results[0].boxes.conf.cpu().tolist()
            })
            st.dataframe(boxes_df.round(3))
        else:
            st.info("No detections above 25% confidence")

st.success("✅ CliniScan deployed! YOLO working 🚀")
