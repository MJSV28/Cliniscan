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
st.markdown("---")

try:
    # Load YOLO
    if not os.path.exists("yolo_best.pt"):
        st.error("❌ **yolo_best.pt** missing - upload to repo root")
        st.stop()
    
    detector = YOLO("yolo_best.pt")
    
    # Load EfficientNet  
    if not os.path.exists("efficientnet_best.pt"):
        st.error("❌ **efficientnet_best.pt** missing - upload to repo root")
        st.stop()
    
    classifier = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = classifier.classifier[1].in_features
    classifier.classifier[1] = nn.Linear(in_features, 5)
    
    checkpoint = torch.load("efficientnet_best.pt", map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    model_dict = checkpoint
    state_dict = classifier.state_dict()
    for k, v in model_dict.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k] = v
    classifier.load_state_dict(state_dict)
    classifier.eval()
    
    st.success("✅ **Both models loaded successfully!**")
    
except Exception as e:
    st.error(f"❌ Load error: {str(e)}")
    st.stop()

CLASS_NAMES = ["Opacity", "Consolidation", "Fibrosis", "Mass", "Other"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("📁 Upload Chest X-ray", type=['png', 'jpg', 'jpeg'], key="uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    # Classification Column
    with col1:
        st.subheader("🧠 **Classification** (EfficientNet)")
        st.image(image, caption="Input X-ray", use_container_width=True)
        
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = classifier(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        df_probs = pd.DataFrame({
            'Disease': CLASS_NAMES,
            'Probability': probs
        }).sort_values('Probability', ascending=False)
        
        st.markdown("**Top Predictions:**")
        st.dataframe(df_probs.round(3), use_container_width=True)
        st.bar_chart(df_probs.set_index('Disease')['Probability'])
    
    # Detection Column  
    with col2:
        st.subheader("📦 **Detection** (YOLO)")
        results = detector(image, conf=0.25, verbose=False)
        
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="YOLO Bounding Boxes", use_container_width=True)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_df = pd.DataFrame({
                'Class': [detector.names[int(cls)] for cls in results[0].boxes.cls.cpu()],
                'Confidence': results[0].boxes.conf.cpu().tolist()
            })
            st.markdown("**Detections:**")
            st.dataframe(boxes_df.round(3), use_container_width=True)
        else:
            st.info("✅ No abnormalities detected (conf > 25%)")

st.markdown("---")
st.success("🚀 **CliniScan Ready** - Upload X-rays for dual analysis!")
st.caption("YOLO Detection + EfficientNet Classification")
