
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
    yolo_path = "yolo_best.pt"      # ✅ HARDCODED
    effnet_path = "efficientnet.pth"  # ✅ HARDCODED
    
    if not os.path.exists(yolo_path):
        st.error(f"❌ YOLO: {yolo_path}"); st.stop()
    if not os.path.exists(effnet_path):
        st.error(f"❌ EfficientNet: {effnet_path}"); st.stop()
    
    detector = YOLO(yolo_path)
    st.success("✅ YOLO loaded")
    
    classifier = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = classifier.classifier[1].in_features
    classifier.classifier[1] = nn.Linear(in_features, 5)
    
    checkpoint = torch.load(effnet_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    state_dict = classifier.state_dict()
    model_dict = checkpoint
    for k, v in model_dict.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k] = v
    classifier.load_state_dict(state_dict)
    st.success("✅ EfficientNet loaded")
    
    return classifier.eval(), detector

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
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = classifier(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        df_probs = pd.DataFrame({
            'Disease': CLASS_NAMES,
            'Probability': probs
        }).sort_values('Probability', ascending=False)
        
        st.subheader("🧠 Classification")
        st.bar_chart(df_probs.set_index('Disease')['Probability'])
        st.dataframe(df_probs.round(3))
    
    with col2:
        st.subheader("📦 Detection")
        results = detector(image, conf=0.25, verbose=False)
        annotated = results[0].plot()
        st.image(annotated, caption="YOLO Detections", use_container_width=True)
        
        if hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_df = pd.DataFrame({
                'Class': [detector.names[int(cls)] for cls in results[0].boxes.cls],
                'Conf': results[0].boxes.conf.cpu().tolist()
            })
            st.dataframe(boxes_df.round(3))

st.success("✅ Ready - Upload chest X-ray!")
