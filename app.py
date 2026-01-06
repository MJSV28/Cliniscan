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
st.markdown("YOLO Detection + EfficientNet Classification")

# NO CACHE - NO SCOPE ISSUES
if "detector" not in st.session_state:
    if os.path.exists("yolo_best.pt"):
        st.session_state.detector = YOLO("yolo_best.pt")
        st.success("✅ YOLO loaded")
    else:
        st.error("❌ **yolo_best.pt** missing in repo root")
        st.stop()

if "classifier" not in st.session_state:
    if os.path.exists("efficientnet_best.pt"):
        classifier = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = classifier.classifier[1].in_features
        classifier.classifier[1] = nn.Linear(in_features, 5)
        
        checkpoint = torch.load("efficientnet_best.pt", map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        state_dict = classifier.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
        classifier.load_state_dict(state_dict)
        classifier.eval()
        st.session_state.classifier = classifier
        st.success("✅ EfficientNet loaded")
    else:
        st.error("❌ **efficientnet_best.pt** missing in repo root")
        st.stop()

detector = st.session_state.detector
classifier = st.session_state.classifier

CLASS_NAMES = ["Opacity", "Consolidation", "Fibrosis", "Mass", "Other"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("📁 **Upload Chest X-ray**", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("**🧠 Classification**")
        st.image(image, caption="Original", use_container_width=True)
        
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = classifier(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        df_probs = pd.DataFrame({'Disease': CLASS_NAMES, 'Probability (%)': (probs*100).round(1)})
        df_probs = df_probs.sort_values('Probability (%)', ascending=False)
        
        st.markdown("**Top 5 Predictions:**")
        st.dataframe(df_probs, use_container_width=True)
        st.bar_chart(df_probs.set_index('Disease')['Probability (%)'])
    
    with col2:
        st.subheader("**📦 Detection**")
        results = detector(image, conf=0.25, verbose=False)
        annotated = results[0].plot()
        st.image(annotated, caption="YOLO Bounding Boxes", use_container_width=True)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_df = pd.DataFrame({
                'Abnormality': [detector.names[int(cls)] for cls in results[0].boxes.cls.cpu()],
                'Confidence (%)': (results[0].boxes.conf.cpu() * 100).round(1).tolist()
            })
            st.markdown("**Detected Abnormalities:**")
            st.dataframe(boxes_df, use_container_width=True)

st.markdown("---")
st.success("✅ **CliniScan Ready** - Dual AI Analysis Complete!")
st.caption("Upload X-rays for instant abnormality detection + classification")
