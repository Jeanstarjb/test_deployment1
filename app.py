import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
import base64
import io
import time
import pandas as pd
from datetime import datetime
import random
import logging
import os
from dotenv import load_dotenv
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="OCULUS | AI Diagnostics",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'workflow_step' not in st.session_state:
    st.session_state.workflow_step = 1

# Initialize Gemini API key from environment
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv('GEMINI_API_KEY', '')

# ================= GRAD-CAM FUNCTIONS =================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image
    
    Args:
        img_array: Preprocessed image array (1, 224, 224, 3)
        model: Trained Keras model
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Class index to visualize (if None, uses top prediction)
    
    Returns:
        heatmap: Normalized heatmap array
    """
    try:
        # Create a model that maps input to activations of last conv layer and predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradient of top predicted class w.r.t. output feature map
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            # Use tf.gather for tensor-safe indexing within GradientTape
            class_channel = tf.gather(preds[0], pred_index)
        
        # Gradient of output w.r.t. output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of mean intensity of gradient over specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by "how important this channel is"
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        # Return empty heatmap on error
        return np.zeros((7, 7))

def create_gradcam_overlay(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Create Grad-CAM overlay on original image with enhanced visibility
    
    Args:
        img: Original PIL Image
        heatmap: Grad-CAM heatmap
        alpha: Transparency of overlay (higher = more visible heatmap)
        colormap: OpenCV colormap to use
    
    Returns:
        superimposed_img: PIL Image with heatmap overlay
    """
    # Resize heatmap to match image size
    img_array = np.array(img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # ENHANCEMENT 1: Boost contrast - apply power transform
    heatmap_resized = np.power(heatmap_resized, 0.7)  # Makes highlights more prominent
    
    # ENHANCEMENT 2: Increase minimum visibility
    heatmap_resized = np.clip(heatmap_resized * 1.5, 0, 1)  # Amplify values
    
    # Convert heatmap to RGB
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # ENHANCEMENT 3: Better blending - use weighted combination
    # Reduce original image brightness to make heatmap stand out
    img_dimmed = img_array * 0.5  # Dim the original image
    superimposed_img = heatmap_colored * alpha + img_dimmed * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

def generate_multi_class_gradcam(img, model, last_conv_layer_name, disease_names, top_n=3, alpha=0.7, colormap=cv2.COLORMAP_JET):
    """
    Generate Grad-CAM for multiple disease classes
    
    Returns:
        dict: {disease_name: heatmap_overlay_image}
    """
    img_array = preprocess_image(img)
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top N predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    
    gradcam_results = {}
    for idx in top_indices:
        disease = disease_names[idx]
        prob = predictions[idx]
        
        # Generate heatmap for this class
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=idx)
        overlay = create_gradcam_overlay(img, heatmap, alpha=alpha, colormap=colormap)
        
        gradcam_results[disease] = {
            'overlay': overlay,
            'probability': prob,
            'heatmap': heatmap
        }
    
    return gradcam_results

def get_last_conv_layer_name(model):
    """
    Automatically find the last convolutional layer in the model
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    
    # If no Conv2D found, try common DenseNet layer names
    try:
        model.get_layer('conv5_block16_concat')
        return 'conv5_block16_concat'
    except:
        pass
    
    # Fallback
    return None

# ================= LOADING SCREEN =================
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False
    st.session_state.loading_start_time = time.time()

if not st.session_state.app_loaded:
    BASE64_VIDEO = ""
    
    loading_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ overflow: hidden; background: #000; font-family: 'Courier New', monospace; }}
            #fallback-bg {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
                animation: gradientPulse 4s ease-in-out infinite;
            }}
            @keyframes gradientPulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.9; }} }}
            #loading-content {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                display: flex; flex-direction: column; justify-content: center; align-items: center;
                z-index: 10;
            }}
            #loading-bar-container {{ width: 400px; text-align: center; }}
            #loading-bar-bg {{
                width: 100%; height: 4px; background: rgba(0, 242, 254, 0.2);
                border-radius: 2px; overflow: hidden; margin-bottom: 20px;
            }}
            #loading-bar {{
                height: 100%; width: 0%;
                background: linear-gradient(90deg, #00f2fe, #ff00ff);
                animation: loadProgress 2s ease-out forwards;
            }}
            @keyframes loadProgress {{ 0% {{ width: 0%; }} 100% {{ width: 100%; }} }}
            #loading-text {{
                color: #00f2fe; font-size: 16px; letter-spacing: 8px;
                font-weight: 900; text-transform: uppercase;
            }}
        </style>
    </head>
    <body>
        <div id="fallback-bg"></div>
        <div id="loading-content">
            <div id="loading-bar-container">
                <div id="loading-bar-bg"><div id="loading-bar"></div></div>
                <div id="loading-text">INITIALIZING GRAD-CAM...</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    st.components.v1.html(loading_html, height=900, scrolling=False)
    time.sleep(2)
    st.session_state.app_loaded = True
    st.rerun()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("my_ocular_model_densenet121.keras", compile=False)
        logger.info("✅ Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ================= CONSTANTS =================
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

OPTIMAL_THRESHOLDS = {
    'Normal': 0.514,
    'Diabetes': 0.300,
    'Glaucoma': 0.531,
    'Cataract': 0.682,
    'AMD': 0.517,
    'Hypertension': 0.533,
    'Myopia': 0.529,
    'Other': 0.256
}

# ================= MAIN APP CSS =================
st.markdown("""
<style>
    :root {
        --primary: #00f2fe;
        --neon-pink: #ff00ff;
        --matrix-green: #00ff41;
        --bg-deep: #0a0a12;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
    }
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--neon-pink)) !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 800 !important;
        padding: 1rem 2.5rem !important;
        border-radius: 50px !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================
def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_diseases(image):
    if model is None:
        st.error("❌ Model not loaded")
        return None
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    return predictions

def format_predictions(predictions, use_optimal_thresholds=True):
    results = []
    detected = []
    
    for i, prob in enumerate(predictions):
        disease_name = DISEASE_NAMES[i]
        threshold = OPTIMAL_THRESHOLDS[disease_name] if use_optimal_thresholds else 0.5
        is_detected = bool(prob >= threshold)
        
        results.append({
            'disease': disease_name,
            'probability': float(prob),
            'threshold': threshold,
            'detected': is_detected
        })
        if is_detected:
            detected.append(disease_name)
    
    return results, detected

# ================= MAIN APP =================
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 class="hero-title">OCULUS PRIME + GRAD-CAM</h1>
    <div style="font-size: 1.5rem; color: #94a3b8;">Visual AI Explainability</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎮 CONTROL DECK")
    
    if model is not None:
        st.success("✓ CNN Model: Loaded")
        st.success("✓ Grad-CAM: Active")
    else:
        st.error("❌ CNN Model: Not Loaded")
    
    st.markdown("---")
    st.markdown("### 🔬 Grad-CAM Settings")
    
    show_gradcam = st.checkbox("Enable Grad-CAM Visualization", value=True)
    gradcam_alpha = st.slider("Heatmap Intensity", 0.3, 0.9, 0.7, 0.05, 
                              help="Higher = More visible heatmap")
    gradcam_top_n = st.slider("Show Top N Classes", 1, 5, 3)
    
    colormap_options = {
        "JET (Red-Blue)": cv2.COLORMAP_JET,
        "Hot (Red-Yellow)": cv2.COLORMAP_HOT,
        "Rainbow": cv2.COLORMAP_RAINBOW,
        "Viridis": cv2.COLORMAP_VIRIDIS
    }
    selected_colormap = st.selectbox("Color Scheme", list(colormap_options.keys()))
    colormap = colormap_options[selected_colormap]
    
    # Store in session state
    st.session_state.gradcam_colormap = colormap

# ================= STEP 1: UPLOAD =================
st.markdown("## 📸 RETINAL IMAGE CAPTURE")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card"><h3>👁 LEFT EYE (OS)</h3>', unsafe_allow_html=True)
    l_file = st.file_uploader("📁 Upload Left Eye", type=['png', 'jpg', 'jpeg'], key='l_up')
    if l_file:
        st.session_state.l_img = Image.open(l_file)
        st.image(st.session_state.l_img, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card"><h3>👁 RIGHT EYE (OD)</h3>', unsafe_allow_html=True)
    r_file = st.file_uploader("📁 Upload Right Eye", type=['png', 'jpg', 'jpeg'], key='r_up')
    if r_file:
        st.session_state.r_img = Image.open(r_file)
        st.image(st.session_state.r_img, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 RUN ANALYSIS WITH GRAD-CAM", use_container_width=True, type="primary"):
    if 'l_img' in st.session_state and 'r_img' in st.session_state:
        if model is None:
            st.error("❌ Model not loaded")
        else:
            with st.spinner("🔬 Running CNN + Grad-CAM Analysis..."):
                # Get predictions
                l_pred = predict_diseases(st.session_state.l_img)
                r_pred = predict_diseases(st.session_state.r_img)
                
                st.session_state.l_pred = l_pred
                st.session_state.r_pred = r_pred
                
                # Generate Grad-CAM if enabled
                if show_gradcam:
                    # Find last conv layer
                    last_conv_layer = get_last_conv_layer_name(model)
                    
                    if last_conv_layer:
                        st.info(f"📍 Using layer: {last_conv_layer}")
                        
                        # Get colormap from session state
                        colormap = st.session_state.get('gradcam_colormap', cv2.COLORMAP_JET)
                        
                        # Generate Grad-CAM for both eyes
                        st.session_state.l_gradcam = generate_multi_class_gradcam(
                            st.session_state.l_img, 
                            model, 
                            last_conv_layer, 
                            DISEASE_NAMES,
                            top_n=gradcam_top_n,
                            alpha=gradcam_alpha,
                            colormap=colormap
                        )
                        
                        st.session_state.r_gradcam = generate_multi_class_gradcam(
                            st.session_state.r_img, 
                            model, 
                            last_conv_layer, 
                            DISEASE_NAMES,
                            top_n=gradcam_top_n,
                            alpha=gradcam_alpha,
                            colormap=colormap
                        )
                    else:
                        st.warning("⚠️ Could not find convolutional layer for Grad-CAM")
                
                st.session_state.results_ready = True
                st.success("✅ Analysis Complete!")
    else:
        st.error("❌ Please upload both images")

# ================= STEP 2: RESULTS WITH GRAD-CAM =================
if st.session_state.get('results_ready'):
    st.markdown("---")
    st.markdown("## 🔬 ANALYSIS RESULTS")
    
    # Left Eye Results
    st.markdown("### 👁 LEFT EYE (OS)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        st.image(st.session_state.l_img, use_column_width=True)
    
    with col2:
        st.markdown("#### CNN Predictions")
        l_results, l_detected = format_predictions(st.session_state.l_pred)
        
        for r in sorted(l_results, key=lambda x: x['probability'], reverse=True)[:5]:
            status = "🔴 DETECTED" if r['detected'] else ""
            st.markdown(f"**{r['disease']}**: {r['probability']:.1%} {status}")
    
    # Grad-CAM Visualization for Left Eye
    if show_gradcam and 'l_gradcam' in st.session_state:
        st.markdown("#### 🔥 Grad-CAM Heatmaps (Model Focus Areas)")
        
        gradcam_cols = st.columns(len(st.session_state.l_gradcam))
        
        for idx, (disease, data) in enumerate(st.session_state.l_gradcam.items()):
            with gradcam_cols[idx]:
                st.markdown(f"**{disease}**")
                st.markdown(f"*{data['probability']:.1%}*")
                st.image(data['overlay'], use_column_width=True)
    
    st.markdown("---")
    
    # Right Eye Results
    st.markdown("### 👁 RIGHT EYE (OD)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        st.image(st.session_state.r_img, use_column_width=True)
    
    with col2:
        st.markdown("#### CNN Predictions")
        r_results, r_detected = format_predictions(st.session_state.r_pred)
        
        for r in sorted(r_results, key=lambda x: x['probability'], reverse=True)[:5]:
            status = "🔴 DETECTED" if r['detected'] else ""
            st.markdown(f"**{r['disease']}**: {r['probability']:.1%} {status}")
    
    # Grad-CAM Visualization for Right Eye
    if show_gradcam and 'r_gradcam' in st.session_state:
        st.markdown("#### 🔥 Grad-CAM Heatmaps (Model Focus Areas)")
        
        gradcam_cols = st.columns(len(st.session_state.r_gradcam))
        
        for idx, (disease, data) in enumerate(st.session_state.r_gradcam.items()):
            with gradcam_cols[idx]:
                st.markdown(f"**{disease}**")
                st.markdown(f"*{data['probability']:.1%}*")
                st.image(data['overlay'], use_column_width=True)
    
    # Interpretation Guide
    st.markdown("---")
    st.markdown("""
    ### 🧠 HOW TO INTERPRET GRAD-CAM HEATMAPS
    
    **Grad-CAM** (Gradient-weighted Class Activation Mapping) shows which regions of the retinal image 
    the neural network focuses on when making predictions:
    
    - 🔴 **Red/Hot areas**: Regions that strongly influence the prediction
    - 🟡 **Yellow/Warm areas**: Moderately important regions  
    - 🔵 **Blue/Cool areas**: Less important for this specific diagnosis
    
    **Clinical Value:**
    - Validates that the model looks at clinically relevant features
    - Helps identify false positives (model focusing on wrong areas)
    - Provides visual explanation for AI decisions
    - Assists in training and model improvement
    
    ⚠️ **Note**: Grad-CAM is a visualization tool. Always combine with clinical expertise.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>OCULUS PRIME + GRAD-CAM</strong></p>
    <p>Explainable AI for Retinal Disease Detection</p>
    <p style='font-size: 0.9rem;'>⚠️ For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)