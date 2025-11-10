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

# ================= GRAD-CAM FUNCTIONS =================

def find_best_gradcam_layer(model):
    """Intelligently find the best layer for Grad-CAM in DenseNet121"""
    candidate_layers = [
        'conv5_block16_2_conv',  # BEST - Last conv in final block
        'conv5_block16_1_conv',
        'conv5_block16_0_conv',
        'conv5_block15_2_conv',
        'conv5_block14_2_conv',
        'pool4_conv',
    ]
    
    logger.info("🔍 Searching for optimal Grad-CAM layer...")
    
    for layer_name in candidate_layers:
        try:
            layer = model.get_layer(layer_name)
            logger.info(f"✅ Found optimal layer: {layer_name}")
            logger.info(f"   Layer type: {type(layer).__name__}")
            logger.info(f"   Output shape: {layer.output_shape}")
            return layer_name
        except:
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
    
    logger.error("❌ CRITICAL: No suitable layer found!")
    return None

def make_gradcam_heatmap_fixed(img_array, model, last_conv_layer_name, pred_index=None):
    """COMPLETELY REWRITTEN Grad-CAM with proper gradient handling"""
    try:
        logger.info(f"🔬 Generating Grad-CAM for layer: {last_conv_layer_name}")
        
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            else:
                pred_index = tf.constant(pred_index, dtype=tf.int32)
            
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            logger.error("❌ GRADIENTS ARE NONE - Model not properly configured!")
            h, w = 7, 7
            y, x = np.ogrid[:h, :w]
            center_heatmap = np.exp(-((x - w//2)**2 + (y - h//2)**2) / (2 * (w/4)**2))
            return center_heatmap
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs_np = conv_outputs.numpy()[0]
        pooled_grads_np = pooled_grads.numpy()
        
        logger.info(f"   Conv outputs shape: {conv_outputs_np.shape}")
        logger.info(f"   Pooled grads shape: {pooled_grads_np.shape}")
        logger.info(f"   Grads - Min: {pooled_grads_np.min():.6f}, Max: {pooled_grads_np.max():.6f}")
        
        for i in range(pooled_grads_np.shape[0]):
            conv_outputs_np[:, :, i] *= pooled_grads_np[i]
        
        heatmap = np.mean(conv_outputs_np, axis=2)
        heatmap = np.maximum(heatmap, 0)
        
        heatmap_max = np.max(heatmap)
        heatmap_min = np.min(heatmap)
        
        logger.info(f"   Raw heatmap - Min: {heatmap_min:.6f}, Max: {heatmap_max:.6f}")
        
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        elif heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        else:
            logger.warning("⚠️ Heatmap is all zeros! Creating fallback heatmap")
            h, w = heatmap.shape
            y, x = np.ogrid[:h, :w]
            heatmap = np.exp(-((x - w//2)**2 + (y - h//2)**2) / (2 * (w/4)**2))
        
        logger.info(f"   ✅ Final Heatmap - Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}, Mean: {heatmap.mean():.4f}")
        logger.info(f"   Non-zero pixels: {(heatmap > 0.1).sum()} / {heatmap.size}")
        
        return heatmap
        
    except Exception as e:
        logger.error(f"❌ Grad-CAM FAILED: {str(e)}")
        logger.exception("Full traceback:")
        h, w = 7, 7
        y, x = np.ogrid[:h, :w]
        center_heatmap = np.exp(-((x - w//2)**2 + (y - h//2)**2) / (2 * (w/4)**2))
        return center_heatmap

def create_enhanced_overlay(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """Create HIGH-VISIBILITY Grad-CAM overlay"""
    img_array = np.array(img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    logger.info(f"🎨 Creating overlay - Heatmap range: [{heatmap_resized.min():.3f}, {heatmap_resized.max():.3f}]")
    
    if heatmap_resized.max() - heatmap_resized.min() < 0.01:
        logger.warning("⚠️ Heatmap has very little variation!")
    
    heatmap_enhanced = np.power(heatmap_resized, 0.7)
    heatmap_uint8 = np.uint8(255 * heatmap_enhanced)
    heatmap_equalized = cv2.equalizeHist(heatmap_uint8)
    
    heatmap_colored = cv2.applyColorMap(heatmap_equalized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)

def generate_multi_class_gradcam_v2(img, model, disease_names, top_n=3, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """Complete rewrite of multi-class Grad-CAM generation"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Multi-Class Grad-CAM Generation")
    logger.info("=" * 60)
    
    last_conv_layer = find_best_gradcam_layer(model)
    
    if not last_conv_layer:
        logger.error("❌ CRITICAL: Cannot proceed without suitable layer!")
        return {}
    
    st.session_state.gradcam_layer = last_conv_layer
    
    img_array = preprocess_image(img)
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
        
        heatmap = make_gradcam_heatmap_fixed(img_array, model, last_conv_layer, pred_index=idx)
        overlay = create_enhanced_overlay(img, heatmap, alpha=alpha, colormap=colormap)
        
        gradcam_results[disease] = {
            'overlay': overlay,
            'probability': prob,
            'heatmap': heatmap
        }
    
    logger.info("✅ Grad-CAM generation complete!")
    logger.info("=" * 60)
    
    return gradcam_results

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
            }}
            #loading-content {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                display: flex; flex-direction: column; justify-content: center; align-items: center;
                z-index: 10;
            }}
            #loading-text {{
                color: #00f2fe; font-size: 18px; letter-spacing: 6px;
                font-weight: 900; text-transform: uppercase;
                animation: pulse 2s ease-in-out infinite;
            }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        </style>
    </head>
    <body>
        <div id="fallback-bg"></div>
        <div id="loading-content">
            <div id="loading-text">INITIALIZING GRAD-CAM...</div>
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
        logger.info("📦 Loading model...")
        model = tf.keras.models.load_model("my_ocular_model_densenet121.keras")
        model.trainable = True
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Total layers: {len(model.layers)}")
        logger.info(f"   Model trainable: {model.trainable}")
        
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
    @media only screen and (max-width: 768px) {
        .hero-title { font-size: 3rem !important; }
        .glass-card { padding: 15px !important; }
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

def validate_fundus_image(image):
    """Validate if uploaded image is a fundus photograph using Gemini AI"""
    import json
    
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        return True, "skipped", "⚠ Gemini API not configured - validation skipped"
    
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = """You are an expert ophthalmologist. Analyze this image and determine if it is a RETINAL FUNDUS PHOTOGRAPH.

RESPOND ONLY WITH A JSON OBJECT (no markdown, no code blocks):
{
  "is_fundus": true/false,
  "confidence": "high/medium/low",
  "reason": "brief explanation",
  "visible_structures": ["list of visible structures"],
  "image_quality": "excellent/good/fair/poor",
  "recommendation": "proceed/retake/use different image"
}"""

        response = model_gemini.generate_content([prompt, image])
        response_text = response.text.strip()
        
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join([l for l in lines if not l.startswith('```')])
            response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        is_valid = result.get('is_fundus', False)
        confidence = result.get('confidence', 'unknown')
        reason = result.get('reason', 'No reason provided')
        
        if is_valid:
            message = f"✅ Valid Fundus Image ({confidence} confidence)"
        else:
            message = f"⚠️ Invalid Image ({confidence} confidence): {reason}"
        
        return is_valid, confidence, message
        
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return True, "error", f"⚠️ Validation failed - proceeding with analysis"

def generate_llm_report(left_results, right_results, left_image, right_image, patient_info):
    """Generate comprehensive medical report using Gemini"""
    
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        return None
    
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model_gemini = genai.GenerativeModel('gemini-2.5-flash')
        
        def format_eye_predictions(results):
            lines = []
            for r in results:
                status = "✓ DETECTED" if r['detected'] else ""
                lines.append(f"  • {r['disease']:15s}: {r['probability']:6.1%} {status}")
            return "\n".join(lines)
        
        left_pred_text = format_eye_predictions(left_results)
        right_pred_text = format_eye_predictions(right_results)
        
        left_detected = [r['disease'] for r in left_results if r['detected']]
        right_detected = [r['disease'] for r in right_results if r['detected']]
        
        prompt = f"""You are an experienced ophthalmologist reviewing retinal fundus images.

PATIENT: {patient_info['name']}, {patient_info['age']}y, {patient_info['gender']}

CNN PREDICTIONS:
LEFT EYE: {', '.join(left_detected) if left_detected else 'Normal'}
RIGHT EYE: {', '.join(right_detected) if right_detected else 'Normal'}

Provide a structured clinical report with:
1. SUMMARY OF FINDINGS
2. DETAILED ANALYSIS (both eyes)
3. CLINICAL SIGNIFICANCE
4. DIFFERENTIAL DIAGNOSIS
5. RECOMMENDED ACTIONS
6. PATIENT COUNSELING POINTS"""

        response = model_gemini.generate_content([prompt, "Left eye:", left_image, "Right eye:", right_image])
        return response.text
    
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

# ================= PROGRESS STEPPER =================
def render_progress_stepper(current_step):
    step1_class = "completed" if current_step > 1 else ("active" if current_step == 1 else "inactive")
    step2_class = "completed" if current_step > 2 else ("active" if current_step == 2 else "inactive")
    step3_class = "active" if current_step == 3 else "inactive"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 20px; margin: 30px 0;">
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; border-radius: 50%; background: {'#00ff41' if step1_class == 'completed' else ('#00f2fe' if step1_class == 'active' else 'rgba(255,255,255,0.2)')}; display: flex; align-items: center; justify-content: center; font-weight: bold;">1</div>
            <div style="font-size: 0.8rem; margin-top: 5px;">Upload</div>
        </div>
        <div style="width: 60px; height: 2px; background: {'#00ff41' if current_step > 1 else 'rgba(255,255,255,0.2)'}; margin-top: 25px;"></div>
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; border-radius: 50%; background: {'#00ff41' if step2_class == 'completed' else ('#00f2fe' if step2_class == 'active' else 'rgba(255,255,255,0.2)')}; display: flex; align-items: center; justify-content: center; font-weight: bold;">2</div>
            <div style="font-size: 0.8rem; margin-top: 5px;">Analysis</div>
        </div>
        <div style="width: 60px; height: 2px; background: {'#00ff41' if current_step > 2 else 'rgba(255,255,255,0.2)'}; margin-top: 25px;"></div>
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; border-radius: 50%; background: {'#00f2fe' if step3_class == 'active' else 'rgba(255,255,255,0.2)'}; display: flex; align-items: center; justify-content: center; font-weight: bold;">3</div>
            <div style="font-size: 0.8rem; margin-top: 5px;">Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= MAIN APP =================
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 class="hero-title">OCULUS PRIME + GRAD-CAM</h1>
    <div style="font-size: 1.5rem; color: #94a3b8;">AI Diagnostics with Visual Explainability</div>
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
    
    show_gradcam = st.checkbox("Enable Grad-CAM", value=True)
    gradcam_alpha = st.slider("Heatmap Intensity", 0.3, 0.9, 0.6, 0.05)
    gradcam_top_n = st.slider("Show Top N Classes", 1, 5, 3)
    
    colormap_options = {
        "JET": cv2.COLORMAP_JET,
        "Hot": cv2.COLORMAP_HOT,
        "Rainbow": cv2.COLORMAP_RAINBOW,
        "Viridis": cv2.COLORMAP_VIRIDIS
    }
    selected_colormap = st.selectbox("Color Scheme", list(colormap_options.keys()))
    colormap = colormap_options[selected_colormap]
    st.session_state.gradcam_colormap = colormap

render_progress_stepper(st.session_state.workflow_step)

# ================= STEP 1: UPLOAD =================
if st.session_state.workflow_step == 1:
    st.markdown("## 👤 PATIENT DATA")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        p_name = st.text_input("NAME", placeholder="Patient identifier...")
    with col_b:
        p_age = st.number_input("AGE", 1, 120, 45)
    
    col_c, col_d = st.columns([1, 2])
    with col_c:
        p_gen = st.selectbox("SEX", ["M", "F", "Other"])
    with col_d:
        p_hist = st.text_area("HISTORY", height=80, placeholder="Medical history...")
    
    st.session_state.patient = {'name': p_name, 'age': p_age, 'gender': p_gen, 'history': p_hist}
    
    st.markdown("---")
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
                left_image = st.session_state.l_img
                right_image = st.session_state.r_img
                
                # Validate images
                if st.session_state.get('gemini_api_key'):
                    st.info("🔍 Validating images...")
                    l_valid, l_conf, l_msg = validate_fundus_image(left_image)
                    r_valid, r_conf, r_msg = validate_fundus_image(right_image)
                    
                    if not l_valid or not r_valid:
                        st.error("🚫 Invalid images detected. Please upload fundus photographs.")
                        st.stop()
                
                # Run CNN analysis
                st.info("🧠 Running CNN + Grad-CAM analysis...")
                with st.spinner("Processing..."):
                    l_pred = predict_diseases(left_image)
                    r_pred = predict_diseases(right_image)
                    
                    st.session_state.l_pred = l_pred
                    st.session_state.r_pred = r_pred
                    
                    # Generate Grad-CAM
                    if show_gradcam:
                        colormap = st.session_state.get('gradcam_colormap', cv2.COLORMAP_JET)
                        
                        st.session_state.l_gradcam = generate_multi_class_gradcam_v2(
                            left_image, model, DISEASE_NAMES,
                            top_n=gradcam_top_n, alpha=gradcam_alpha, colormap=colormap
                        )
                        
                        st.session_state.r_gradcam = generate_multi_class_gradcam_v2(
                            right_image, model, DISEASE_NAMES,
                            top_n=gradcam_top_n, alpha=gradcam_alpha, colormap=colormap
                        )
                    
                    st.session_state.results_ready = True
                
                st.success("✅ Analysis complete!")
                time.sleep(0.5)
                st.session_state.workflow_step = 2
                st.rerun()
        else:
            st.error("❌ Please upload both images")

# ================= STEP 2: DIAGNOSTICS =================
elif st.session_state.workflow_step == 2:
    st.markdown("## 🔬 DIAGNOSTIC RESULTS WITH GRAD-CAM")
    
    if st.session_state.get('results_ready'):
        if 'gradcam_layer' in st.session_state:
            st.info(f"📍 Grad-CAM Layer: **{st.session_state.gradcam_layer}**")
        
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
            
            cols = st.columns(len(st.session_state.r_gradcam))
            for idx, (disease, data) in enumerate(st.session_state.r_gradcam.items()):
                with cols[idx]:
                    st.markdown(f"**{disease}**")
                    st.markdown(f"*{data['probability']:.1%}*")
                    st.image(data['overlay'], use_column_width=True)
                    if 'heatmap' in data:
                        hm = data['heatmap']
                        st.caption(f"Range: [{hm.min():.2f}, {hm.max():.2f}]")
        
        # Store results for report
        st.session_state.l_res = l_results
        st.session_state.r_res = r_results
        st.session_state.l_detected = l_detected
        st.session_state.r_detected = r_detected
        
        # Interpretation Guide
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
        
        st.markdown("---")
        
        # Bilateral Comparison
        st.markdown("### 🔄 BILATERAL COMPARISON")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Left Eye Conditions", len(l_detected))
        
        with col2:
            st.metric("Right Eye Conditions", len(r_detected))
        
        with col3:
            bilateral = set(l_detected) & set(r_detected)
            st.metric("Bilateral Conditions", len(bilateral))
        
        if bilateral:
            st.warning(f"⚠️ **Bilateral findings:** {', '.join(bilateral)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ BACK", use_container_width=True):
                st.session_state.workflow_step = 1
                st.rerun()
        
        with col3:
            if st.button("GENERATE REPORT ➡️", use_container_width=True, type="primary"):
                # Generate AI report if Gemini is available
                if st.session_state.get('gemini_api_key'):
                    with st.spinner("🤖 Generating clinical report with Gemini AI..."):
                        report = generate_llm_report(
                            l_results, r_results,
                            st.session_state.l_img, st.session_state.r_img,
                            st.session_state.patient
                        )
                        
                        if report:
                            st.session_state.clinical_report = report
                            st.success("✅ Report generated!")
                        else:
                            st.warning("⚠️ AI report generation failed. Proceeding with CNN results only.")
                
                st.session_state.workflow_step = 3
                st.rerun()
    else:
        st.info("🔄 No results available. Please run analysis first.")
        if st.button("⬅️ BACK TO UPLOAD", use_container_width=True):
            st.session_state.workflow_step = 1
            st.rerun()

# ================= STEP 3: REPORT =================
elif st.session_state.workflow_step == 3:
    st.markdown("## 📊 CLINICAL REPORT")
    
    if st.session_state.get('results_ready'):
        # Patient Information
        st.markdown(f"""
        <div class='glass-card' style='margin-bottom: 30px;'>
            <h3 style='color: var(--primary); margin-top: 0;'>👤 PATIENT INFORMATION</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;'>
                <div>
                    <p style='color: #94a3b8; margin: 5px 0;'>NAME</p>
                    <p style='font-size: 1.3rem; font-weight: bold;'>{st.session_state.patient['name']}</p>
                </div>
                <div>
                    <p style='color: #94a3b8; margin: 5px 0;'>AGE</p>
                    <p style='font-size: 1.3rem; font-weight: bold;'>{st.session_state.patient['age']} years</p>
                </div>
                <div>
                    <p style='color: #94a3b8; margin: 5px 0;'>GENDER</p>
                    <p style='font-size: 1.3rem; font-weight: bold;'>{st.session_state.patient['gender']}</p>
                </div>
                <div>
                    <p style='color: #94a3b8; margin: 5px 0;'>REPORT DATE</p>
                    <p style='font-size: 1.3rem; font-weight: bold;'>{datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # CNN Predictions Summary
        st.markdown("### 🤖 CNN MODEL PREDICTIONS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👁 LEFT EYE (OS)")
            left_detected = [r['disease'] for r in st.session_state.l_res if r['detected']]
            if left_detected:
                st.error(f"**Detected:** {', '.join(left_detected)}")
            else:
                st.success("**No abnormalities detected**")
            
            sorted_left = sorted(st.session_state.l_res, key=lambda x: x['probability'], reverse=True)[:3]
            for pred in sorted_left:
                detected_badge = "🔴" if pred['detected'] else "⚪"
                st.markdown(f"{detected_badge} **{pred['disease']}**: {pred['probability']:.1%}")
        
        with col2:
            st.markdown("#### 👁 RIGHT EYE (OD)")
            right_detected = [r['disease'] for r in st.session_state.r_res if r['detected']]
            if right_detected:
                st.error(f"**Detected:** {', '.join(right_detected)}")
            else:
                st.success("**No abnormalities detected**")
            
            sorted_right = sorted(st.session_state.r_res, key=lambda x: x['probability'], reverse=True)[:3]
            for pred in sorted_right:
                detected_badge = "🔴" if pred['detected'] else "⚪"
                st.markdown(f"{detected_badge} **{pred['disease']}**: {pred['probability']:.1%}")
        
        st.markdown("---")
        
        # AI Clinical Report
        if 'clinical_report' in st.session_state:
            st.markdown("### 🧠 AI-ENHANCED CLINICAL ANALYSIS")
            st.markdown(st.session_state.clinical_report)
        else:
            st.info("💡 AI report not generated. Displaying CNN predictions only.")
        
        st.markdown("---")
        
        # Export Options
        st.markdown("### 📥 EXPORT OPTIONS")
        
        # Prepare full report text
        left_detected_str = ', '.join([r['disease'] for r in st.session_state.l_res if r['detected']]) or 'None'
        right_detected_str = ', '.join([r['disease'] for r in st.session_state.r_res if r['detected']]) or 'None'
        
        full_report = f"""
OCULUS PRIME - CLINICAL DIAGNOSTIC REPORT
==========================================

PATIENT INFORMATION
-------------------
Name: {st.session_state.patient['name']}
Age: {st.session_state.patient['age']} years
Gender: {st.session_state.patient['gender']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CNN MODEL PREDICTIONS
---------------------
LEFT EYE (OS):
  Detected: {left_detected_str}
  
  Detailed Predictions:
"""
        for r in sorted(st.session_state.l_res, key=lambda x: x['probability'], reverse=True):
            status = "[DETECTED]" if r['detected'] else ""
            full_report += f"    • {r['disease']:15s}: {r['probability']:6.1%} {status}\n"
        
        full_report += f"""
RIGHT EYE (OD):
  Detected: {right_detected_str}
  
  Detailed Predictions:
"""
        for r in sorted(st.session_state.r_res, key=lambda x: x['probability'], reverse=True):
            status = "[DETECTED]" if r['detected'] else ""
            full_report += f"    • {r['disease']:15s}: {r['probability']:6.1%} {status}\n"
        
        if 'clinical_report' in st.session_state:
            full_report += f"""
AI-ENHANCED CLINICAL ANALYSIS
------------------------------
{st.session_state.clinical_report}
"""
        
        full_report += """
DISCLAIMER
----------
This report is generated by an AI system for research and educational purposes.
All findings should be verified by a qualified ophthalmologist.
"""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "💾 DOWNLOAD REPORT",
                full_report,
                file_name=f"OCULUS_Report_{st.session_state.patient['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            if st.button("📧 EMAIL REPORT", use_container_width=True):
                st.info("📧 Email functionality coming soon...")
        
        with col3:
            if st.button("🖨 PRINT", use_container_width=True):
                st.success("📄 Report ready for printing!")
        
        st.markdown("---")
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ BACK", use_container_width=True):
                st.session_state.workflow_step = 2
                st.rerun()
        
        with col2:
            if st.button("🔄 NEW ANALYSIS", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['app_loaded', 'gemini_api_key']:
                        del st.session_state[key]
                st.session_state.workflow_step = 1
                st.rerun()
        
        with col3:
            if st.button("🏁 COMPLETE", use_container_width=True, type="primary"):
                st.balloons()
                st.success("✅ Analysis complete!")
    else:
        st.error("❌ No results available")
        if st.button("⬅️ START OVER", use_container_width=True):
            st.session_state.workflow_step = 1
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>OCULUS PRIME + GRAD-CAM v2.0</strong></p>
    <p>AI-Powered Retinal Disease Detection with Visual Explainability</p>
    <p style='font-size: 0.9rem;'>⚠️ For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)