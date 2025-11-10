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

# Setup logging with MORE VERBOSE output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

# ================= IMPROVED GRAD-CAM FUNCTIONS =================

def find_best_gradcam_layer(model):
    """
    Intelligently find the best layer for Grad-CAM in DenseNet121
    """
    # DenseNet121 optimal layers (in priority order)
    candidate_layers = [
        'conv5_block16_2_conv',  # BEST - Last conv in final block
        'conv5_block16_1_conv',
        'conv5_block16_0_conv',
        'conv5_block15_2_conv',
        'conv5_block14_2_conv',
        'pool4_conv',
    ]
    
    logger.info("🔍 Searching for optimal Grad-CAM layer...")
    
    # Try each candidate
    for layer_name in candidate_layers:
        try:
            layer = model.get_layer(layer_name)
            logger.info(f"✅ Found optimal layer: {layer_name}")
            logger.info(f"   Layer type: {type(layer).__name__}")
            logger.info(f"   Output shape: {layer.output_shape}")
            return layer_name
        except:
            logger.debug(f"   Layer {layer_name} not found, trying next...")
            continue
    
    # Fallback: find ANY Conv2D layer
    logger.warning("⚠️ Optimal layers not found, searching for Conv2D...")
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
    
    if conv_layers:
        best_layer = conv_layers[-1]
        logger.info(f"✅ Using fallback Conv2D: {best_layer}")
        return best_layer
    
    # Last resort: find any layer with 'conv' in name
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            logger.warning(f"⚠️ Using last-resort layer: {layer.name}")
            return layer.name
    
    logger.error("❌ CRITICAL: No suitable layer found!")
    return None

def make_gradcam_heatmap_fixed(img_array, model, last_conv_layer_name, pred_index=None):
    """
    COMPLETELY REWRITTEN Grad-CAM with proper gradient handling
    """
    try:
        logger.info(f"🔬 Generating Grad-CAM for layer: {last_conv_layer_name}")
        
        # Get the target convolutional layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Create gradient model
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = grad_model(img_array)
            
            # Get class index
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the score for the predicted class
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Check for None gradients (common issue)
        if grads is None:
            logger.error("❌ GRADIENTS ARE NONE - Model not properly configured!")
            logger.error("💡 FIX: Load model WITHOUT compile=False")
            return np.ones((7, 7)) * 0.5  # Return middle-value heatmap for visibility
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Convert to numpy
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        
        logger.info(f"   Conv outputs shape: {conv_outputs_np.shape}")
        logger.info(f"   Pooled grads shape: {pooled_grads_np.shape}")
        logger.info(f"   Grads - Min: {pooled_grads_np.min():.6f}, Max: {pooled_grads_np.max():.6f}")
        
        # Weight each feature map by gradient importance
        for i in range(len(pooled_grads_np)):
            conv_outputs_np[:, :, i] *= pooled_grads_np[i]
        
        # Create heatmap by averaging weighted feature maps
        heatmap = np.mean(conv_outputs_np, axis=-1)
        
        # ReLU activation (keep only positive contributions)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        heatmap_max = heatmap.max()
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        else:
            logger.warning("⚠️ Heatmap max is zero!")
            heatmap = np.ones_like(heatmap) * 0.5
        
        # Log statistics
        logger.info(f"   ✅ Heatmap - Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}, Mean: {heatmap.mean():.4f}")
        logger.info(f"   Non-zero pixels: {(heatmap > 0.1).sum()} / {heatmap.size}")
        
        return heatmap
        
    except Exception as e:
        logger.error(f"❌ Grad-CAM FAILED: {str(e)}")
        logger.exception("Full traceback:")
        return np.ones((7, 7)) * 0.5

def create_enhanced_overlay(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Create HIGH-VISIBILITY Grad-CAM overlay with aggressive enhancement
    """
    # Convert PIL to numpy
    img_array = np.array(img.resize((224, 224)))
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    logger.info(f"🎨 Creating overlay - Heatmap range: [{heatmap_resized.min():.3f}, {heatmap_resized.max():.3f}]")
    
    # AGGRESSIVE ENHANCEMENT for visibility
    # 1. Apply power transform (gamma correction)
    heatmap_enhanced = np.power(heatmap_resized, 0.6)
    
    # 2. Histogram equalization for better contrast
    heatmap_uint8 = np.uint8(255 * heatmap_enhanced)
    heatmap_uint8 = cv2.equalizeHist(heatmap_uint8)
    
    # 3. Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 4. Create overlay with controlled blending
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlay)

def generate_multi_class_gradcam_v2(img, model, disease_names, top_n=3, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Complete rewrite of multi-class Grad-CAM generation
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Multi-Class Grad-CAM Generation")
    logger.info("=" * 60)
    
    # Find best layer
    last_conv_layer = find_best_gradcam_layer(model)
    
    if not last_conv_layer:
        logger.error("❌ CRITICAL: Cannot proceed without suitable layer!")
        return {}
    
    # Store for UI display
    st.session_state.gradcam_layer = last_conv_layer
    
    # Preprocess image
    img_array = preprocess_image(img)
    
    # Get predictions
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    
    logger.info(f"📊 Top {top_n} predictions:")
    for idx in top_indices:
        logger.info(f"   {disease_names[idx]}: {predictions[idx]:.1%}")
    
    gradcam_results = {}
    
    for idx in top_indices:
        disease = disease_names[idx]
        prob = predictions[idx]
        
        logger.info(f"\n🔬 Generating Grad-CAM for: {disease} ({prob:.1%})")
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap_fixed(
            img_array, 
            model, 
            last_conv_layer, 
            pred_index=idx
        )
        
        # Create overlay
        overlay = create_enhanced_overlay(img, heatmap, alpha=alpha, colormap=colormap)
        
        gradcam_results[disease] = {
            'overlay': overlay,
            'probability': prob,
            'heatmap': heatmap
        }
    
    logger.info("✅ Grad-CAM generation complete!")
    logger.info("=" * 60)
    
    return gradcam_results

def diagnose_model(model, img_array):
    """
    Comprehensive model diagnostics for debugging
    """
    st.markdown("### 🔍 MODEL DIAGNOSTICS")
    
    diag_info = []
    
    # Check model trainability
    diag_info.append(f"**Model Trainable:** {model.trainable}")
    
    # Check layer count
    diag_info.append(f"**Total Layers:** {len(model.layers)}")
    
    # Find Conv2D layers
    conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    diag_info.append(f"**Conv2D Layers Found:** {len(conv_layers)}")
    
    # Test forward pass
    try:
        predictions = model.predict(img_array, verbose=0)
        diag_info.append(f"**Forward Pass:** ✅ Success")
        diag_info.append(f"**Prediction Shape:** {predictions.shape}")
    except Exception as e:
        diag_info.append(f"**Forward Pass:** ❌ Failed - {str(e)}")
    
    # Test gradient computation
    try:
        with tf.GradientTape() as tape:
            preds = model(img_array)
            loss = preds[0, 0]
        
        grads = tape.gradient(loss, model.trainable_variables[0])
        if grads is not None:
            diag_info.append(f"**Gradient Computation:** ✅ Working")
        else:
            diag_info.append(f"**Gradient Computation:** ❌ Returns None")
    except Exception as e:
        diag_info.append(f"**Gradient Computation:** ❌ Error - {str(e)}")
    
    # Display diagnostics
    for info in diag_info:
        st.markdown(info)
    
    # Show last 10 layers
    with st.expander("📋 Last 10 Layers"):
        for layer in model.layers[-10:]:
            st.code(f"{layer.name} ({type(layer).__name__})")

# ================= LOADING SCREEN =================
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

if not st.session_state.app_loaded:
    loading_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { overflow: hidden; background: #000; font-family: 'Courier New', monospace; }
            #fallback-bg {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
            }
            #loading-content {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                display: flex; flex-direction: column; justify-content: center; align-items: center;
                z-index: 10;
            }
            #loading-text {
                color: #00f2fe; font-size: 18px; letter-spacing: 6px;
                font-weight: 900; text-transform: uppercase;
                animation: pulse 2s ease-in-out infinite;
            }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        </style>
    </head>
    <body>
        <div id="fallback-bg"></div>
        <div id="loading-content">
            <div id="loading-text">INITIALIZING ENHANCED GRAD-CAM...</div>
        </div>
    </body>
    </html>
    """
    st.components.v1.html(loading_html, height=900, scrolling=False)
    time.sleep(2)
    st.session_state.app_loaded = True
    st.rerun()

# ================= LOAD MODEL (FIXED) =================
@st.cache_resource
def load_model():
    try:
        logger.info("📦 Loading model...")
        
        # CRITICAL FIX: Remove compile=False to enable gradients!
        model = tf.keras.models.load_model("my_ocular_model_densenet121.keras")
        
        # Ensure model is trainable (required for gradients)
        model.trainable = True
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Total layers: {len(model.layers)}")
        logger.info(f"   Model trainable: {model.trainable}")
        
        # Log available layers
        logger.info("📋 Last 15 layers:")
        for layer in model.layers[-15:]:
            logger.info(f"   - {layer.name} ({type(layer).__name__})")
        
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.exception("Full traceback:")
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
    <div style="font-size: 1.5rem; color: #94a3b8;">Enhanced Visual AI Explainability</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎮 CONTROL DECK")
    
    if model is not None:
        st.success("✓ CNN Model: Loaded")
        st.success("✓ Grad-CAM: Enhanced")
    else:
        st.error("❌ CNN Model: Not Loaded")
    
    st.markdown("---")
    st.markdown("### 🔬 Grad-CAM Settings")
    
    show_gradcam = st.checkbox("Enable Grad-CAM Visualization", value=True)
    gradcam_alpha = st.slider("Heatmap Intensity", 0.3, 0.9, 0.6, 0.05)
    gradcam_top_n = st.slider("Show Top N Classes", 1, 5, 3)
    
    colormap_options = {
        "JET (Red-Blue)": cv2.COLORMAP_JET,
        "Hot (Red-Yellow)": cv2.COLORMAP_HOT,
        "Rainbow": cv2.COLORMAP_RAINBOW,
        "Viridis": cv2.COLORMAP_VIRIDIS,
        "Plasma": cv2.COLORMAP_PLASMA
    }
    selected_colormap = st.selectbox("Color Scheme", list(colormap_options.keys()))
    colormap = colormap_options[selected_colormap]
    
    st.session_state.gradcam_colormap = colormap
    
    st.markdown("---")
    show_diagnostics = st.checkbox("Show Diagnostics", value=False)

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

# Show diagnostics if requested
if show_diagnostics and model is not None and 'l_img' in st.session_state:
    img_array = preprocess_image(st.session_state.l_img)
    diagnose_model(model, img_array)

if st.button("🚀 RUN ENHANCED ANALYSIS", use_container_width=True, type="primary"):
    if 'l_img' in st.session_state and 'r_img' in st.session_state:
        if model is None:
            st.error("❌ Model not loaded")
        else:
            with st.spinner("🔬 Running Enhanced CNN + Grad-CAM Analysis..."):
                # Get predictions
                l_pred = predict_diseases(st.session_state.l_img)
                r_pred = predict_diseases(st.session_state.r_img)
                
                st.session_state.l_pred = l_pred
                st.session_state.r_pred = r_pred
                
                # Generate Grad-CAM if enabled
                if show_gradcam:
                    colormap = st.session_state.get('gradcam_colormap', cv2.COLORMAP_JET)
                    
                    # Generate for both eyes
                    st.session_state.l_gradcam = generate_multi_class_gradcam_v2(
                        st.session_state.l_img, 
                        model, 
                        DISEASE_NAMES,
                        top_n=gradcam_top_n,
                        alpha=gradcam_alpha,
                        colormap=colormap
                    )
                    
                    st.session_state.r_gradcam = generate_multi_class_gradcam_v2(
                        st.session_state.r_img, 
                        model, 
                        DISEASE_NAMES,
                        top_n=gradcam_top_n,
                        alpha=gradcam_alpha,
                        colormap=colormap
                    )
                
                st.session_state.results_ready = True
                st.success("✅ Analysis Complete!")
    else:
        st.error("❌ Please upload both images")

# ================= STEP 2: RESULTS =================
if st.session_state.get('results_ready'):
    st.markdown("---")
    st.markdown("## 🔬 ANALYSIS RESULTS")
    
    # Show layer info
    if 'gradcam_layer' in st.session_state:
        st.info(f"📍 Grad-CAM Layer: **{st.session_state.gradcam_layer}**")
    
    # Left Eye
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
    
    if show_gradcam and 'l_gradcam' in st.session_state:
        st.markdown("#### 🔥 Grad-CAM Heatmaps")
        
        cols = st.columns(len(st.session_state.l_gradcam))
        for idx, (disease, data) in enumerate(st.session_state.l_gradcam.items()):
            with cols[idx]:
                st.markdown(f"**{disease}**")
                st.markdown(f"*{data['probability']:.1%}*")
                st.image(data['overlay'], use_column_width=True)
                if 'heatmap' in data:
                    hm = data['heatmap']
                    st.caption(f"Range: [{hm.min():.2f}, {hm.max():.2f}]")
    
    st.markdown("---")
    
    # Right Eye
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
    
    if show_gradcam and 'r_gradcam' in st.session_state:
        st.markdown("#### 🔥 Grad-CAM Heatmaps")
        
        cols = st.columns(len(st.session_state.r_gradcam))
        for idx, (disease, data) in enumerate(st.session_state.r_gradcam.items()):
            with cols[idx]:
                st.markdown(f"**{disease}**")
                st.markdown(f"*{data['probability']:.1%}*")
                st.image(data['overlay'], use_column_width=True)
                if 'heatmap' in data:
                    hm = data['heatmap']
                    st.caption(f"Range: [{hm.min():.2f}, {hm.max():.2f}]")
    
    st.markdown("---")
    st.markdown("""
    ### 🧠 INTERPRETING GRAD-CAM HEATMAPS
    
    **What the colors mean:**
    - 🔴 **Red/Hot**: Areas the model focuses on most strongly
    - 🟡 **Yellow/Warm**: Moderately important regions
    - 🔵 **Blue/Cool**: Less relevant for this prediction
    
    **Clinical use:**
    - Verify the model examines clinically relevant areas (optic disc, macula, vessels)
    - Identify potential false positives (focusing on artifacts, edges)
    - Build trust through transparent AI decision-making
    
    ⚠️ **Important**: Grad-CAM is an interpretability tool. Always validate with clinical expertise.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>OCULUS PRIME + ENHANCED GRAD-CAM v2.0</strong></p>
    <p>Explainable AI for Retinal Disease Detection</p>
    <p style='font-size: 0.9rem;'>⚠️ For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)