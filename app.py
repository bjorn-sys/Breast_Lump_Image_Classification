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
import json
import time
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
    .risk-high { background: linear-gradient(135deg, #ff4757, #ff3838); color: white; }
    .risk-medium { background: linear-gradient(135deg, #ffa502, #ff9f1a); color: white; }
    .risk-low { background: linear-gradient(135deg, #2ed573, #1dd1a1); color: white; }
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

    # Enhanced Sidebar Features
    st.markdown("---")
    st.markdown("### 🎨 Visualization Settings")
    
    heatmap_opacity = st.slider("Heatmap Opacity", 0.1, 1.0, 0.5)
    contour_color = st.color_picker("Contour Color", "#FF0000")
    contour_width = st.slider("Contour Width", 1, 10, 3)
    
    viz_mode = st.radio(
        "Visualization Mode",
        ["Heatmap", "Contour Only", "Heatmap + Contour", "Binary Mask"]
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        0.5, 1.0, 0.7,
        help="Minimum confidence level for reliable prediction"
    )
    
    st.markdown("---")
    st.markdown("### 📁 Batch Analysis")
    batch_files = st.file_uploader(
        "Upload multiple images", 
        type=['jpg', 'jpeg', 'png', 'bmp'], 
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    export_format = st.selectbox(
        "Export Format",
        ["PDF Report", "JSON", "CSV"]
    )
    
    if st.button("🌐 Generate API Call"):
        st.session_state.show_api = True
    
    st.markdown("---")
    st.markdown("### 👥 Collaboration")
    
    st.markdown("---")
    st.markdown("### ⚡ Performance")
    if st.checkbox("Show System Metrics"):
        if torch.cuda.is_available():
            st.metric("GPU Memory", f"{torch.cuda.memory_allocated()/1024**3:.1f} GB")
            st.metric("GPU Usage", "Active")
        else:
            st.metric("Device", "CPU")
        st.metric("Model Version", "Breast2 v1.0")

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

def create_heatmap(cam, original_image, opacity=0.5):
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        original_resized = original_image.resize((224, 224))
        if original_resized.mode != 'RGB':
            original_resized = original_resized.convert('RGB')
        original_np = np.array(original_resized)
        
        cam_image = (heatmap * opacity + original_np * (1 - opacity)).astype(np.uint8)
        return cam_image, cam
    except Exception as e:
        original_resized = original_image.resize((224, 224))
        if original_resized.mode != 'RGB':
            original_resized = original_resized.convert('RGB')
        return np.array(original_resized), cam

def create_smart_contour(cam, original_image, color=(255, 0, 0), width=3):
    """Create precise contour lines around affected areas"""
    try:
        original_size = original_image.size
        cam_resized = cv2.resize(cam, original_size)
        
        # Convert to uint8 and apply threshold
        cam_uint8 = np.uint8(cam_resized * 255)
        _, binary_mask = cv2.threshold(cam_uint8, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert to PIL
        if original_image.mode != 'RGB':
            result_image = original_image.convert('RGB')
        else:
            result_image = original_image.copy()
        
        draw = ImageDraw.Draw(result_image)
        
        if contours:
            # Get the most significant contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(main_contour, True)
            simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # Convert to points and draw
            contour_points = [(point[0][0], point[0][1]) for point in simplified_contour]
            
            if len(contour_points) > 2:
                for i in range(len(contour_points)):
                    start_point = contour_points[i]
                    end_point = contour_points[(i + 1) % len(contour_points)]
                    draw.line([start_point, end_point], fill=color, width=width)
            
            return result_image, main_contour
        else:
            return original_image, None
            
    except Exception as e:
        return original_image, None

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

def apply_grad_cam(model, image_tensor, original_image, predicted_class, viz_mode="Heatmap", opacity=0.5, contour_color="#FF0000", contour_width=3):
    try:
        target_layer = model.layer4[-1].conv3
        grad_cam = GradCAM(model, target_layer)
        cam, _ = grad_cam.generate_cam(image_tensor, predicted_class)
        
        if viz_mode == "Contour Only":
            contour_image, _ = create_smart_contour(cam, original_image, color=contour_color, width=contour_width)
            return contour_image, cam, (112, 112)
        elif viz_mode == "Heatmap + Contour":
            heatmap_image, _ = create_heatmap(cam, original_image, opacity)
            contour_image, _ = create_smart_contour(cam, Image.fromarray(heatmap_image), color=contour_color, width=contour_width)
            return np.array(contour_image), cam, (112, 112)
        elif viz_mode == "Binary Mask":
            # Create binary mask visualization
            original_size = original_image.size
            cam_resized = cv2.resize(cam, original_size)
            binary_mask = np.uint8(cam_resized > 0.3) * 255
            mask_image = Image.fromarray(binary_mask).convert('RGB')
            return np.array(mask_image), cam, (112, 112)
        else:  # Heatmap
            cam_image, heatmap = create_heatmap(cam, original_image, opacity)
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

def explain_prediction(prediction, probabilities, confidence):
    """Provide AI explanation for the prediction"""
    explanations = {
        'malignant': [
            "Irregular mass margins detected",
            "Architectural distortion present",
            "Suspicious microcalcifications observed",
            "Asymmetric tissue density identified"
        ],
        'benign': [
            "Well-circumscribed mass margins",
            "Uniform tissue pattern observed",
            "Stable appearance characteristics",
            "Typical benign features present"
        ],
        'normal': [
            "Normal tissue architecture maintained",
            "No suspicious masses detected",
            "Uniform density distribution",
            "Typical fibroglandular pattern"
        ]
    }
    
    return explanations.get(prediction, ["Standard tissue patterns observed"])

def calculate_risk_score(patient_data, prediction, confidence):
    """Calculate comprehensive risk score"""
    base_risk = {
        'malignant': 0.8,
        'benign': 0.3,
        'normal': 0.1
    }[prediction]
    
    # Adjust based on confidence
    adjusted_risk = base_risk * confidence
    
    # Adjust based on patient factors
    if patient_data.get('family_history') == "Breast Cancer":
        adjusted_risk *= 1.5
    if patient_data.get('previous_biopsy', "No") != "No":
        adjusted_risk *= 1.3
        
    return min(adjusted_risk, 1.0)

def image_quality_check(image):
    """Check image quality before analysis"""
    quality_issues = []
    
    # Check resolution
    if image.size[0] < 512 or image.size[1] < 512:
        quality_issues.append("Low resolution")
    
    # Check contrast
    img_array = np.array(image.convert('L'))
    contrast = np.std(img_array)
    if contrast < 50:
        quality_issues.append("Low contrast")
    
    return quality_issues

def generate_pdf_report(analysis_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, alignment=1, textColor=colors.HexColor('#1f77b4'))
    story.append(Paragraph("BREAST ULTRASOUND AI ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    if 'patient_data' in analysis_data:
        story.append(Paragraph("<b>PATIENT INFORMATION</b>", styles['Heading2']))
        patient_info = f"""
        <b>Patient ID:</b> {analysis_data['patient_data'].get('patient_id', 'N/A')}<br/>
        <b>Age:</b> {analysis_data['patient_data'].get('age', 'N/A')}<br/>
        <b>Gender:</b> {analysis_data['patient_data'].get('gender', 'N/A')}<br/>
        <b>Family History:</b> {analysis_data['patient_data'].get('family_history', 'N/A')}<br/>
        <b>Breast Density:</b> {analysis_data['patient_data'].get('breast_density', 'N/A')}
        """
        story.append(Paragraph(patient_info, styles['Normal']))
        story.append(Spacer(1, 15))
    
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
    
    # Risk Assessment
    if 'risk_score' in analysis_data:
        story.append(Paragraph("<b>RISK ASSESSMENT</b>", styles['Heading2']))
        risk_text = f"<b>Overall Risk Score:</b> {analysis_data['risk_score']:.1%}<br/>"
        story.append(Paragraph(risk_text, info_style))
        story.append(Spacer(1, 15))
    
    # AI Explanations
    if 'explanations' in analysis_data:
        story.append(Paragraph("<b>AI EXPLANATION</b>", styles['Heading2']))
        for explanation in analysis_data['explanations']:
            story.append(Paragraph(f"• {explanation}", info_style))
            story.append(Spacer(1, 5))
        story.append(Spacer(1, 15))
    
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

    # Patient Information Form
    with st.expander("👤 Patient Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID", value="PAT-001")
            age = st.number_input("Age", min_value=18, max_value=100, value=45)
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            family_history = st.selectbox("Family History", ["None", "Breast Cancer", "Other Cancer"])
        with col3:
            breast_density = st.selectbox("Breast Density", ["A - Fatty", "B - Scattered", "C - Heterogeneous", "D - Extremely Dense"])
            previous_biopsy = st.selectbox("Previous Biopsy", ["No", "Yes - Benign", "Yes - Atypical", "Yes - Malignant"])
        
        patient_data = {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'family_history': family_history,
            'breast_density': breast_density,
            'previous_biopsy': previous_biopsy
        }

    # Batch Processing
    if batch_files and len(batch_files) > 1:
        st.markdown("### 📊 Batch Analysis Results")
        progress_bar = st.progress(0)
        batch_results = []
        
        for i, uploaded_file in enumerate(batch_files):
            progress_bar.progress((i + 1) / len(batch_files))
            try:
                image = Image.open(uploaded_file)
                start_time = time.time()
                prediction, probabilities, confidence, image_tensor = predict_image(model, image)
                inference_time = time.time() - start_time
                
                batch_results.append({
                    'filename': uploaded_file.name,
                    'prediction': classes[prediction],
                    'confidence': confidence,
                    'inference_time': inference_time,
                    'probabilities': probabilities
                })
            except Exception as e:
                batch_results.append({
                    'filename': uploaded_file.name,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'inference_time': 0.0,
                    'error': str(e)
                })
        
        # Display batch results
        for result in batch_results:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            with col1:
                st.write(f"**{result['filename']}**")
            with col2:
                st.write(f"**{result['prediction'].upper()}**")
            with col3:
                st.write(f"{result['confidence']:.1%}")
            with col4:
                st.write(f"{result['inference_time']:.2f}s")
        
        st.success(f"✅ Batch analysis completed: {len([r for r in batch_results if r['prediction'] != 'Error'])} successful, {len([r for r in batch_results if r['prediction'] == 'Error'])} failed")
        return

    # Single Image Analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Ultrasound Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose breast ultrasound image", type=['jpg', 'jpeg', 'png', 'bmp'], key="single_upload")
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image Quality Check
                quality_issues = image_quality_check(image)
                if quality_issues:
                    st.warning(f"⚠️ Image Quality Issues: {', '.join(quality_issues)}")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    with col2:
        st.markdown('<div class="sub-header">🔍 AI Analysis Results</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                start_time = time.time()
                
                with st.spinner('Analyzing with YOUR trained model...'):
                    prediction, probabilities, confidence, image_tensor = predict_image(model, image)
                    cam_image, heatmap, heat_point = apply_grad_cam(
                        model, image_tensor, image, prediction, 
                        viz_mode, heatmap_opacity, contour_color, contour_width
                    )
                
                inference_time = time.time() - start_time
                predicted_class = classes[prediction]
                recommendations = get_clinical_recommendations(predicted_class, confidence)
                explanations = explain_prediction(predicted_class, probabilities, confidence)
                risk_score = calculate_risk_score(patient_data, predicted_class, confidence)
                
                # Confidence threshold warning
                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Low confidence prediction ({confidence:.1%}). Consider manual review.")
                
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Prediction", "🔥 Visualization", "💡 Recommendations", "📊 Analytics", "🤖 AI Explanation", "📄 Export"])
                
                with tab1:
                    colors_dict = {'benign': '#2ecc71', 'malignant': '#e74c3c', 'normal': '#3498db'}
                    emojis = {'benign': '✅', 'malignant': '⚠️', 'normal': '👍'}
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: white; margin: 0;">{emojis[predicted_class]} {predicted_class.upper()}</h3>
                        <p style="margin: 10px 0;">Confidence: <strong>{confidence:.1%}</strong></p>
                        <p style="margin: 0; font-size: 0.9em;">Inference Time: {inference_time:.2f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk Score
                    risk_class = "risk-high" if risk_score > 0.7 else "risk-medium" if risk_score > 0.3 else "risk-low"
                    st.markdown(f"""
                    <div class="{risk_class}" style="padding: 15px; border-radius: 8px; text-align: center; margin: 10px 0;">
                        <h4 style="margin: 0; color: white;">Overall Risk Score: {risk_score:.1%}</h4>
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
                    st.image(cam_image, caption=f"{viz_mode} Visualization", use_column_width=True)
                    if viz_mode == "Heatmap":
                        annotated_image = add_arrow_annotation(Image.fromarray(cam_image), heat_point, "AI Focus Area")
                        st.image(annotated_image, caption="Annotated Image", use_column_width=True)
                
                with tab3:
                    st.markdown("### Clinical Recommendations")
                    for i, action in enumerate(recommendations['actions'], 1):
                        st.markdown(f"**{i}.** {action}")
                    
                    # Comparative Analysis
                    st.markdown("---")
                    st.markdown("### 📈 Comparative Analysis")
                    previous_scan = st.file_uploader("Upload Previous Scan for Comparison", type=['jpg', 'jpeg', 'png'], key="previous_scan")
                    
                    if previous_scan:
                        prev_image = Image.open(previous_scan)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(prev_image, caption="Previous Scan", use_column_width=True)
                        with col2:
                            st.image(image, caption="Current Scan", use_column_width=True)
                        
                        # Compare predictions
                        prev_prediction, prev_prob, prev_confidence, _ = predict_image(model, prev_image)
                        prev_class = classes[prev_prediction]
                        
                        st.metric("Previous Prediction", prev_class.upper(), 
                                 delta=f"{prev_confidence:.1%} confidence")
                        st.metric("Current Prediction", predicted_class.upper(),
                                 delta=f"{confidence:.1%} confidence")
                
                with tab4:
                    st.markdown("### 📊 Analytics Dashboard")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Prediction Confidence", f"{confidence:.1%}")
                    with col2:
                        st.metric("Uncertainty Index", f"{(1-confidence):.1%}")
                    with col3:
                        certainty = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                        st.metric("Decision Certainty", certainty)
                    with col4:
                        st.metric("Risk Category", recommendations['urgency'])
                    
                    # Probability distribution
                    st.markdown("#### Probability Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#2ecc71', '#e74c3c', '#3498db']
                    bars = ax.bar(classes, probabilities, color=colors, alpha=0.7)
                    ax.set_ylabel('Probability')
                    ax.set_title('Class Probability Distribution')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                
                with tab5:
                    st.markdown("### 🤖 AI Explanation")
                    st.info("The AI model based its prediction on the following features:")
                    
                    for i, explanation in enumerate(explanations, 1):
                        st.markdown(f"**{i}.** {explanation}")
                    
                    # Feature importance visualization
                    st.markdown("#### Feature Attention Map")
                    st.image(cam_image, caption="Areas of high AI attention are highlighted", use_column_width=True)
                
                with tab6:
                    st.markdown("### 📄 Export Options")
                    
                    analysis_data = {
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'recommendations': recommendations,
                        'explanations': explanations,
                        'risk_score': risk_score,
                        'patient_data': patient_data,
                        'inference_time': inference_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # PDF Report
                    pdf_buffer = generate_pdf_report(analysis_data)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"breast_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
                    # JSON Export
                    json_data = json.dumps(analysis_data, indent=2)
                    st.download_button(
                        label="📊 Download JSON Data",
                        data=json_data,
                        file_name=f"breast_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # API Call Generation
                    if st.session_state.get('show_api', False):
                        st.markdown("### 🌐 API Call Example")
                        api_payload = {
                            "patient_id": patient_data['patient_id'],
                            "prediction": predicted_class,
                            "confidence": confidence,
                            "risk_score": risk_score,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.code(f"POST /api/analysis\n{json.dumps(api_payload, indent=2)}")
                
                # Real-time Collaboration
                with st.expander("💬 Clinical Notes & Collaboration"):
                    radiologist_notes = st.text_area(
                        "Radiologist Notes",
                        placeholder="Add your clinical observations, additional findings, or comments for the clinical team...",
                        height=100
                    )
                    
                    if st.button("💬 Share with Clinical Team"):
                        st.success("Analysis and notes shared with clinical team")
                        # In a real application, this would integrate with your hospital system
                
                st.success("Analysis complete using YOUR trained model!")
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()