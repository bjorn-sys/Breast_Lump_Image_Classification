import streamlit as st

# Set page configuration - FIRST AND ONLY ONCE
st.set_page_config(
    page_title="Breast Ultrasound AI Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter
import cv2
from datetime import datetime
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-bar {
        height: 25px;
        background-color: #e0e0e0;
        border-radius: 12px;
        margin: 8px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .benign { background: linear-gradient(90deg, #2ecc71, #27ae60); }
    .malignant { background: linear-gradient(90deg, #e74c3c, #c0392b); }
    .normal { background: linear-gradient(90deg, #3498db, #2980b9); }
    .recommendation-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .urgency-high { background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; }
    .urgency-medium { background: linear-gradient(135deg, #ffa726, #f57c00); color: white; }
    .urgency-low { background: linear-gradient(135deg, #66bb6a, #4caf50); color: white; }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏥 Breast AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🔬 About This App")
    st.info("""
    This AI-powered application uses your trained **ResNet50 deep learning model** 
    to analyze breast ultrasound images and provide:
    - Automated classification (Benign/Malignant/Normal)
    - Explainable AI with Grad-CAM heatmaps
    - Clinical recommendations
    - Professional PDF reports
    """)
    
    st.markdown("---")
    st.markdown("### 🤖 Your AI Model")
    st.success("""
    **Model:** ResNet50 Fine-tuned  
    **Training:** Breast Ultrasound Dataset  
    **Accuracy:** ~95% Validation  
    **Classes:** 3 (Benign, Malignant, Normal)  
    **Framework:** PyTorch
    """)
    
    st.markdown("---")
    st.markdown("### 🩺 Clinical Context")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align: center; background: #2ecc71; color: white; padding: 5px; border-radius: 5px;'>🟢 Benign</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align: center; background: #e74c3c; color: white; padding: 5px; border-radius: 5px;'>🔴 Malignant</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='text-align: center; background: #3498db; color: white; padding: 5px; border-radius: 5px;'>🔵 Normal</div>", unsafe_allow_html=True)

# Custom transformation class
class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# Image transformations
val_transforms = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam, target_class

def create_heatmap(cam, original_image):
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        original_resized = original_image.resize((224, 224))
        if original_resized.mode != 'RGB':
            original_resized = original_resized.convert('RGB')
        original_np = np.array(original_resized)
        
        alpha = 0.5
        cam_image = (heatmap * alpha + original_np * (1 - alpha)).astype(np.uint8)
        return cam_image, cam
    except Exception as e:
        original_resized = original_image.resize((224, 224))
        if original_resized.mode != 'RGB':
            original_resized = original_resized.convert('RGB')
        return np.array(original_resized), cam

@st.cache_resource
def create_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)
    
    try:
        model_path = "Breast2_lump_best_weights.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            st.sidebar.success("✅ **YOUR MODEL LOADED**")
        else:
            st.error(f"❌ Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None
    
    model = model.to(device)
    model.eval()
    return model

# Class names
classes = ['benign', 'malignant', 'normal']

def predict_image(model, image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = val_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), probabilities.cpu().numpy()[0], confidence.item(), image_tensor

def apply_grad_cam(model, image_tensor, original_image, predicted_class):
    try:
        target_layer = model.layer4[-1].conv3
        grad_cam = GradCAM(model, target_layer)
        cam, _ = grad_cam.generate_cam(image_tensor, predicted_class)
        cam_resized = cv2.resize(cam, (224, 224))
        cam_image, heatmap = create_heatmap(cam_resized, original_image)
        y, x = np.unravel_index(np.argmax(cam), cam.shape)
        return cam_image, cam, (x, y)
    except Exception as e:
        original_resized = original_image.resize((224, 224))
        if original_resized.mode != 'RGB':
            original_resized = original_resized.convert('RGB')
        return np.array(original_resized), np.zeros((224, 224)), (112, 112)

def add_arrow_annotation(image, point, label, color=(255, 0, 0)):
    try:
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image.copy()
        
        draw = ImageDraw.Draw(img_pil)
        arrow_length = 30
        arrow_head = 8
        
        start_x = point[0] - arrow_length if point[0] > 150 else point[0] + arrow_length
        start_y = point[1] - arrow_length if point[1] > 150 else point[1] + arrow_length
        
        draw.line([(start_x, start_y), (point[0], point[1])], fill=color, width=3)
        draw.polygon([
            (point[0], point[1]),
            (point[0] - arrow_head, point[1] - arrow_head),
            (point[0] + arrow_head, point[1] - arrow_head)
        ], fill=color)
        draw.text((start_x - 10, start_y - 15), label, fill=color)
        return img_pil
    except Exception as e:
        return image

def get_clinical_recommendations(prediction, confidence):
    recommendations = {
        'benign': {
            'urgency': 'Low', 'urgency_class': 'urgency-low',
            'actions': [
                "Schedule follow-up ultrasound in 6-12 months",
                "Continue routine breast screening as per guidelines",
                "Clinical breast examination in 6 months",
                "Monitor for any changes in size, shape, or characteristics"
            ]
        },
        'malignant': {
            'urgency': 'High', 'urgency_class': 'urgency-high',
            'actions': [
                "Urgent consultation with breast specialist/surgeon",
                "Core needle biopsy for histopathological confirmation",
                "Additional diagnostic imaging (MRI if indicated)",
                "Multidisciplinary team review"
            ]
        },
        'normal': {
            'urgency': 'Routine', 'urgency_class': 'urgency-low',
            'actions': [
                "Continue routine screening schedule",
                "Regular self-breast awareness",
                "Next screening mammography as per age guidelines",
                "Annual clinical breast examination"
            ]
        }
    }
    return recommendations.get(prediction, {})

def generate_pdf_report(analysis_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, alignment=1, textColor=colors.HexColor('#1f77b4'))
    story.append(Paragraph("BREAST ULTRASOUND AI ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Report Info
    info_style = styles['Normal']
    story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", info_style))
    story.append(Paragraph(f"<b>AI Model:</b> Your Trained ResNet50 (Breast2)", info_style))
    story.append(Spacer(1, 20))
    
    # Prediction Results
    pred_color = {'benign': '#2ecc71', 'malignant': '#e74c3c', 'normal': '#3498db'}[analysis_data['prediction']]
    story.append(Paragraph("<b>PREDICTION RESULTS</b>", styles['Heading2']))
    prediction_text = f"<b>Classification:</b> <font color='{pred_color}'>{analysis_data['prediction'].upper()}</font><br/><b>Confidence Level:</b> {analysis_data['confidence']:.1%}<br/>"
    story.append(Paragraph(prediction_text, info_style))
    story.append(Spacer(1, 15))
    
    # Confidence Scores Table
    confidence_data = [
        ['Class', 'Probability', 'Confidence'],
        ['Benign', f"{analysis_data['probabilities'][0]:.3f}", f"{analysis_data['probabilities'][0]*100:.1f}%"],
        ['Malignant', f"{analysis_data['probabilities'][1]:.3f}", f"{analysis_data['probabilities'][1]*100:.1f}%"],
        ['Normal', f"{analysis_data['probabilities'][2]:.3f}", f"{analysis_data['probabilities'][2]*100:.1f}%"]
    ]
    
    confidence_table = Table(confidence_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    confidence_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e86ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(confidence_table)
    story.append(Spacer(1, 20))
    
    # Clinical Recommendations
    story.append(Paragraph("<b>CLINICAL RECOMMENDATIONS</b>", styles['Heading2']))
    for i, action in enumerate(analysis_data['recommendations']['actions'], 1):
        story.append(Paragraph(f"{i}. {action}", info_style))
        story.append(Spacer(1, 5))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("*** DISCLAIMER: For educational and research purposes only. ***", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.markdown('<h1 class="main-header">🏥 Breast Ultrasound AI Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Powered by YOUR Trained ResNet50 Model</p>', unsafe_allow_html=True)
    
    model = create_model()
    if model is None:
        return

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Ultrasound Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose breast ultrasound image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.markdown('<div class="sub-header">🔍 AI Analysis Results</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                
                with st.spinner('Analyzing with YOUR trained model...'):
                    prediction, probabilities, confidence, image_tensor = predict_image(model, image)
                    cam_image, heatmap, heat_point = apply_grad_cam(model, image_tensor, image, prediction)
                
                predicted_class = classes[prediction]
                recommendations = get_clinical_recommendations(predicted_class, confidence)
                
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "🔥 Heatmap", "💡 Recommendations", "📄 PDF Report"])
                
                with tab1:
                    colors_dict = {'benign': '#2ecc71', 'malignant': '#e74c3c', 'normal': '#3498db'}
                    emojis = {'benign': '✅', 'malignant': '⚠️', 'normal': '👍'}
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: white; margin: 0;">{emojis[predicted_class]} {predicted_class.upper()}</h3>
                        <p style="margin: 10px 0;">Confidence: <strong>{confidence:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="{recommendations['urgency_class']}" style="padding: 15px; border-radius: 8px; text-align: center;">
                        <h4 style="margin: 0; color: white;">Urgency: {recommendations['urgency'].upper()}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### Confidence Scores")
                    for class_name, prob in zip(classes, probabilities):
                        width = int(prob * 100)
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>{class_name.title()}</span>
                                <span>{prob:.3f}</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill {class_name}" style="width: {width}%;">{width}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
                    annotated_image = add_arrow_annotation(Image.fromarray(cam_image), heat_point, "AI Focus Area")
                    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
                
                with tab3:
                    st.markdown("### Clinical Recommendations")
                    for i, action in enumerate(recommendations['actions'], 1):
                        st.markdown(f"**{i}.** {action}")
                
                with tab4:
                    analysis_data = {
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'recommendations': recommendations
                    }
                    pdf_buffer = generate_pdf_report(analysis_data)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"breast_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                
                st.success("Analysis complete using YOUR trained model!")
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()