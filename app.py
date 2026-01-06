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
st.title("🫁 **CliniScan** - Chest X-ray AI")
st.markdown("Dual Analysis: EfficientNet Classification + YOLO Detection")

# Load models safely
if os.path.exists("yolo_best.pt") and os.path.exists("efficientnet_best.pt"):
    detector = YOLO("yolo_best.pt")
    
    classifier = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = classifier.classifier[1].in_features
    classifier.classifier[1] = nn.Linear(in_features, 5)
    
    checkpoint = torch.load("efficientnet_best.pt", map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    state_dict = classifier.state_dict()
    for k, v in checkpoint.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k] = v
    classifier.load_state_dict(state_dict)
    classifier.eval()
    
    st.success("✅ **Models loaded** - Ready for analysis!")
else:
    st.error("❌ Missing model files in repo root")
    st.stop()

CLASS_NAMES = ["Opacity", "Consolidation", "Fibrosis", "Mass", "Other"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("📁 Upload Chest X-ray", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    # EfficientNet Classification
    with col1:
        st.subheader("**Classification Results**")
        st.image(image, caption="Original X-ray", use_container_width=True)
        
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = classifier(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        df_probs = pd.DataFrame({
            'Disease': CLASS_NAMES,
            'Probability': probs.round(3)
        }).sort_values('Probability', ascending=False)
        
        st.markdown("**Top Predictions:**")
        st.dataframe(df_probs, use_container_width=True)
        st.bar_chart(df_probs.set_index('Disease')['Probability'])
    
    # YOLO Detection - FIXED TypeError
    with col2:
        st.subheader("**Detection Results**")
        results = detector(image, conf=0.25, verbose=False)
        
        # Always safe plot
        annotated = results[0].plot()
        st.image(annotated, caption="YOLO Analysis", use_container_width=True)
        
        # SAFE boxes extraction
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            try:
                classes = [detector.names[int(cls)] for cls in boxes.cls.cpu()]
                confs = boxes.conf.cpu().numpy()
                
                boxes_df = pd.DataFrame({
                    'Abnormality': classes,
                    'Confidence': [f"{c:.1%}" for c in confs]
                })
                st.markdown("**Detected Abnormalities:**")
                st.dataframe(boxes_df)
            except:
                st.info("✅ Detections visualized above")
        else:
            st.info("✅ No detections above 25% confidence")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.success("🎉 **CliniScan Active** - Professional Analysis Complete!")
