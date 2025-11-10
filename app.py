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

# ================= LOADING SCREEN =================
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

if not st.session_state.app_loaded:
    # PASTE YOUR BASE64 VIDEO STRING HERE IF DESIRED
    BASE64_VIDEO = "" 
    loading_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                overflow: hidden; 
                background: #000; 
                font-family: 'Courier New', monospace;
                width: 100vw; height: 100vh;
            }}
            #fallback-bg {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: radial-gradient(circle at 20% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%),
                            radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
                            linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
                z-index: 1;
                animation: gradientPulse 4s ease-in-out infinite;
            }}
            @keyframes gradientPulse {{
                0%, 100% {{ opacity: 1; filter: hue-rotate(0deg) brightness(1); }}
                50% {{ opacity: 0.9; filter: hue-rotate(180deg) brightness(1.05); }}
            }}
            #loading-content {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                display: flex; flex-direction: column; justify-content: center; align-items: center;
                z-index: 10;
            }}
            #loading-bar-container {{ width: 400px; text-align: center; z-index: 20; }}
            #loading-bar-bg {{
                width: 100%; height: 4px; background: rgba(0, 242, 254, 0.2);
                border-radius: 2px; overflow: hidden; margin-bottom: 20px;
                box-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
            }}
            #loading-bar {{
                height: 100%; width: 0%;
                background: linear-gradient(90deg, #00f2fe, #ff00ff);
                box-shadow: 0 0 30px #00f2fe;
                animation: loadProgress 3s ease-out forwards;
            }}
            @keyframes loadProgress {{ 0% {{ width: 0%; }} 100% {{ width: 100%; }} }}
            #loading-text {{
                color: #00f2fe; font-size: 16px; letter-spacing: 8px;
                font-weight: 900; text-shadow: 0 0 20px #00f2fe;
                text-transform: uppercase;
            }}
        </style>
    </head>
    <body>
        <div id="fallback-bg"></div>
        <div id="loading-content">
            <div id="loading-bar-container">
                <div id="loading-bar-bg"><div id="loading-bar"></div></div>
                <div id="loading-text">INITIALIZING NEURAL CORE...</div>
            </div>
        </div>
        <script>
            setTimeout(() => {{
                document.getElementById('loading-text').textContent = 'ACCESS GRANTED';
                document.getElementById('loading-text').style.color = '#00ff41';
            }}, 2800);
        </script>
    </body>
    </html>
    """
    st.components.v1.html(loading_html, height=900, scrolling=False)
    time.sleep(3.5)
    st.session_state.app_loaded = True
    st.rerun()

# ================= GRAD-CAM FUNCTIONS =================
def find_best_gradcam_layer(model):
    """Intelligently find the best layer for Grad-CAM"""
    candidate_layers = [
        'conv5_block16_2_conv', 'conv5_block16_1_conv', 'conv5_block16_0_conv',
        'conv5_block15_2_conv', 'pool4_conv',
    ]
    for layer_name in candidate_layers:
        try:
            model.get_layer(layer_name)
            return layer_name
        except:
            continue
    # Fallback
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap_fixed(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap with proper gradient handling"""
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None: return np.zeros((7,7)) # Fallback if gradients fail
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_np = conv_outputs.numpy()[0]
        pooled_grads_np = pooled_grads.numpy()
        
        for i in range(pooled_grads_np.shape[0]):
            conv_outputs_np[:, :, i] *= pooled_grads_np[i]
            
        heatmap = np.mean(conv_outputs_np, axis=2)
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return np.zeros((7,7))

def create_enhanced_overlay(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """Create high-visibility Grad-CAM overlay"""
    img_array = np.array(img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlay)

def generate_multi_class_gradcam_v2(img, model, disease_names, top_n=3, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """Generate Grad-CAM for top N predicted classes"""
    last_conv_layer = find_best_gradcam_layer(model)
    if not last_conv_layer: return {}
    
    img_array = preprocess_image(img)
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    
    results = {}
    for idx in top_indices:
        heatmap = make_gradcam_heatmap_fixed(img_array, model, last_conv_layer, pred_index=idx)
        overlay = create_enhanced_overlay(img, heatmap, alpha=alpha, colormap=colormap)
        results[disease_names[idx]] = {
            'overlay': overlay,
            'probability': predictions[idx],
            'heatmap': heatmap
        }
    return results

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        # IMPORTANT: compile=False removed to ensure gradients work for Grad-CAM
        model = tf.keras.models.load_model("my_ocular_model_densenet121.keras")
        model.trainable = True # Ensure trainable for gradient computation
        logger.info("✅ Model loaded successfully with gradients enabled")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ================= CONSTANTS =================
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
OPTIMAL_THRESHOLDS = {
    'Normal': 0.514, 'Diabetes': 0.300, 'Glaucoma': 0.531, 'Cataract': 0.682,
    'AMD': 0.517, 'Hypertension': 0.533, 'Myopia': 0.529, 'Other': 0.256
}

# ================= MAIN APP CSS =================
st.markdown("""
<style>
    :root {
        --primary: #00f2fe; --primary-glow: #00f2feaa; --primary-dark: #4facfe;
        --neon-pink: #ff00ff; --matrix-green: #00ff41; --bg-deep: #0a0a12;
    }
    .stApp {
        background: radial-gradient(circle at 20% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
                    linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif;
    }
    .hero-title {
        font-size: 5rem; font-weight: 900; letter-spacing: -3px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 242, 254, 0.5);
    }
    .hero-subtitle { font-size: 1.8rem; color: #94a3b8; letter-spacing: 6px; text-transform: uppercase; }
    .glass-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px) saturate(180%); -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 24px; padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    }
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%) !important;
        color: #0f172a !important; border: none !important; font-weight: 800 !important;
        padding: 1rem 2.5rem !important; letter-spacing: 2px !important; border-radius: 50px !important;
    }
    .progress-stepper { display: flex; justify-content: center; gap: 20px; margin: 30px 0; padding: 20px; }
    .step-circle { width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; transition: all 0.3s ease; }
    .step-circle.active { background: linear-gradient(135deg, var(--primary), var(--neon-pink)); color: #0f172a; box-shadow: 0 0 20px var(--primary); }
    .step-circle.completed { background: var(--matrix-green); color: #0f172a; }
    .step-circle.inactive { background: rgba(255,255,255,0.1); border: 2px solid rgba(255,255,255,0.3); color: #94a3b8; }
    [data-testid="stSidebar"] { background: rgba(10, 10, 18, 0.9) !important; backdrop-filter: blur(20px) !important; border-right: 1px solid rgba(0, 242, 254, 0.2) !important; }
    @keyframes slideIn { 0% { opacity: 0; transform: translateX(-30px); } 100% { opacity: 1; transform: translateX(0); } }
    .slide-in { animation: slideIn 0.6s ease-out; }
</style>
""", unsafe_allow_html=True)

# ================= BACKEND FUNCTIONS =================
def preprocess_image(image):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_diseases(image):
    if model is None: return None
    return model.predict(preprocess_image(image), verbose=0)[0]

def format_predictions(predictions, use_optimal_thresholds=True):
    results = []
    detected = []
    for i, prob in enumerate(predictions):
        disease_name = DISEASE_NAMES[i]
        threshold = OPTIMAL_THRESHOLDS[disease_name] if use_optimal_thresholds else 0.5
        is_detected = bool(prob >= threshold)
        results.append({'disease': disease_name, 'probability': float(prob), 'threshold': threshold, 'detected': is_detected})
        if is_detected: detected.append(disease_name)
    return results, detected

def validate_fundus_image(image):
    """Simplified validation placeholder or Gemini-based if configured"""
    if not st.session_state.gemini_api_key: return True, "skipped", "Validation skipped (No API key)"
    # ... (Gemini validation logic from original code would go here if needed)
    return True, "skipped", "Validation skipped (Simplified mode)"

import streamlit.components.v1 as components
def results_card_enhanced(eye_side, predictions, use_optimal_thresholds=True):
    results, detected = format_predictions(predictions, use_optimal_thresholds)
    sorted_preds = sorted(results, key=lambda x: x['probability'], reverse=True)
    top_finding = sorted_preds[0]
    is_normal = top_finding['disease'] == 'Normal'
    status_color = "#10b981" if is_normal else ("#f59e0b" if top_finding['probability'] < 0.7 else "#ef4444")
    status_icon = "✅" if is_normal else "⚠️" if top_finding['probability'] < 0.7 else "🚨"
    
    html_content = f"""
    <style>
        body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: transparent; color: #e2e8f0; }}
        .glass-card-inner {{
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 20px;
            border-top: 4px solid {status_color};
        }}
        .tech-bar-bg {{ background: rgba(0,0,0,0.3); height: 8px; border-radius: 10px; overflow: hidden; }}
        .tech-bar-fill {{ height: 100%; transition: width 1s ease-out; }}
    </style>
    <div class="glass-card-inner">
        <h3 style="margin-top: 0; display: flex; justify-content: space-between;">
            <span>{eye_side}</span><span style="font-size: 1.5rem;">{status_icon}</span>
        </h3>
        <div style="font-size: 1.2rem; color: {status_color}; font-weight: 600; margin-bottom: 20px;">
            {top_finding['disease']} ({top_finding['probability']:.1%})
        </div>
    """
    for pred in sorted_preds[:4]:
        prob = pred['probability'] * 100
        color = "#10b981" if pred['disease'] == 'Normal' else ("#f59e0b" if prob < 50 else "#ef4444")
        html_content += f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.9rem;">
                <span>{pred['disease']}</span><span style="color: {color};">{prob:.1f}%</span>
            </div>
            <div class="tech-bar-bg"><div class="tech-bar-fill" style="width: {prob}%; background: {color};"></div></div>
        </div>"""
    html_content += "</div>"
    components.html(html_content, height=300, scrolling=False)
    return results, detected

# ================= MAIN APP LAYOUT =================
with st.sidebar:
    st.markdown("## 🎮 CONTROL DECK")
    if st.session_state.get('gemini_api_key'):
        st.success("✓ AI Report Generation: Active")
    else:
        st.warning("⚠️ AI Reports: Inactive (No Key)")
    if model is not None:
        st.success("✓ CNN Model: Loaded")
    
    st.markdown("---")
    st.markdown("### 🧠 Explainability (Grad-CAM)")
    show_gradcam = st.toggle("Enable Grad-CAM", value=True)
    if show_gradcam:
        gradcam_alpha = st.slider("Heatmap Opacity", 0.1, 1.0, 0.6, 0.1)
        gradcam_top_n = st.slider("Top N Classes", 1, 5, 3)
        colormap_options = {"JET (Red-Blue)": cv2.COLORMAP_JET, "TURBO (Vibrant)": cv2.COLORMAP_TURBO, "HOT (Red-Yellow)": cv2.COLORMAP_HOT}
        selected_cmap = st.selectbox("Color Scheme", list(colormap_options.keys()))
        st.session_state.gradcam_colormap = colormap_options[selected_cmap]
    
    st.markdown("---")
    st.markdown(f"**Workflow Step:** {st.session_state.workflow_step}/3")

# HEADER
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 class="hero-title">OCULUS PRIME</h1>
    <div class="hero-subtitle">NEURAL RETINAL INTERFACE</div>
</div>
""", unsafe_allow_html=True)

# PROGRESS STEPPER
s1 = "active" if st.session_state.workflow_step == 1 else "completed"
s2 = "inactive" if st.session_state.workflow_step < 2 else ("active" if st.session_state.workflow_step == 2 else "completed")
s3 = "inactive" if st.session_state.workflow_step < 3 else "active"
st.markdown(f"""
<div class="progress-stepper">
    <div class="step-circle {s1}">1</div>
    <div class="step-circle {s2}">2</div>
    <div class="step-circle {s3}">3</div>
</div>
""", unsafe_allow_html=True)

# ================= STEP 1: UPLOAD =================
if st.session_state.workflow_step == 1:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card"><h3 style="margin-top:0;">👤 PATIENT DATA</h3>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        p_name = c1.text_input("NAME", placeholder="Patient identifier...")
        p_age = c2.number_input("AGE", 1, 120, 45)
        c3, c4 = st.columns([1, 2])
        p_gen = c3.selectbox("SEX", ["M", "F", "Other"])
        p_hist = c4.text_area("HISTORY", height=70, placeholder="Relevant history...")
        st.session_state.patient = {'name': p_name, 'age': p_age, 'gender': p_gen, 'history': p_hist}
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📸 RETINAL IMAGE CAPTURE")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><h3>👁 LEFT EYE (OS)</h3>', unsafe_allow_html=True)
        lf = st.file_uploader("Upload Left Eye", type=['png','jpg','jpeg'], key='l_up')
        if lf: st.session_state.l_img = Image.open(lf)
        if 'l_img' in st.session_state: st.image(st.session_state.l_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card"><h3>👁 RIGHT EYE (OD)</h3>', unsafe_allow_html=True)
        rf = st.file_uploader("Upload Right Eye", type=['png','jpg','jpeg'], key='r_up')
        if rf: st.session_state.r_img = Image.open(rf)
        if 'r_img' in st.session_state: st.image(st.session_state.r_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 INITIATE NEURAL ANALYSIS", use_container_width=True, type="primary"):
        if 'l_img' in st.session_state and 'r_img' in st.session_state and model:
            with st.spinner("🧠 Processing Neural Analysis & Generating Explanations..."):
                # 1. Predictions
                st.session_state.l_pred = predict_diseases(st.session_state.l_img)
                st.session_state.r_pred = predict_diseases(st.session_state.r_img)
                
                # 2. Grad-CAM (if enabled)
                if show_gradcam:
                    cmap = st.session_state.get('gradcam_colormap', cv2.COLORMAP_JET)
                    st.session_state.l_gradcam = generate_multi_class_gradcam_v2(
                        st.session_state.l_img, model, DISEASE_NAMES, gradcam_top_n, gradcam_alpha, cmap)
                    st.session_state.r_gradcam = generate_multi_class_gradcam_v2(
                        st.session_state.r_img, model, DISEASE_NAMES, gradcam_top_n, gradcam_alpha, cmap)
                
                st.session_state.results_ready = True
                time.sleep(0.5)
                st.session_state.workflow_step = 2
                st.rerun()
        else:
            st.error("❌ Please upload both images before proceeding.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= STEP 2: DIAGNOSTICS =================
elif st.session_state.workflow_step == 2:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    st.markdown("## 🔬 DIAGNOSTIC RESULTS")
    
    if st.session_state.get('results_ready'):
        c1, c2 = st.columns(2)
        
        # LEFT EYE RESULTS
        with c1:
            st.image(st.session_state.l_img, caption="Left Eye (OS)", use_column_width=True)
            l_res, l_det = results_card_enhanced("LEFT EYE", st.session_state.l_pred)
            st.session_state.l_res, st.session_state.l_detected = l_res, l_det
            
            if show_gradcam and 'l_gradcam' in st.session_state:
                with st.expander("🔥 Explainability (Grad-CAM) - Left Eye", expanded=False):
                    tabs = st.tabs(list(st.session_state.l_gradcam.keys()))
                    for i, (disease, data) in enumerate(st.session_state.l_gradcam.items()):
                        with tabs[i]:
                            st.image(data['overlay'], caption=f"{disease}: {data['probability']:.1%}", use_column_width=True)

        # RIGHT EYE RESULTS
        with c2:
            st.image(st.session_state.r_img, caption="Right Eye (OD)", use_column_width=True)
            r_res, r_det = results_card_enhanced("RIGHT EYE", st.session_state.r_pred)
            st.session_state.r_res, st.session_state.r_detected = r_res, r_det
            
            if show_gradcam and 'r_gradcam' in st.session_state:
                with st.expander("🔥 Explainability (Grad-CAM) - Right Eye", expanded=False):
                    tabs = st.tabs(list(st.session_state.r_gradcam.keys()))
                    for i, (disease, data) in enumerate(st.session_state.r_gradcam.items()):
                        with tabs[i]:
                            st.image(data['overlay'], caption=f"{disease}: {data['probability']:.1%}", use_column_width=True)

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        if c1.button("⬅️ BACK"):
             st.session_state.workflow_step = 1
             st.rerun()
        if c3.button("GENERATE REPORT ➡️", type="primary"):
             st.session_state.workflow_step = 3
             st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= STEP 3: REPORT =================
elif st.session_state.workflow_step == 3:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: var(--primary);'>📊 CLINICAL SUMMARY</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='glass-card'>
        <h3>👤 PATIENT: {st.session_state.patient['name']} ({st.session_state.patient['age']}{st.session_state.patient['gender']})</h3>
        <hr style='border-color: rgba(255,255,255,0.1);'>
        <h4>👁 LEFT EYE FINDINGS:</h4>
        {', '.join([d.upper() for d in st.session_state.l_detected]) if st.session_state.l_detected else 'No significant abnormalities detected.'}
        <br><br>
        <h4>👁 RIGHT EYE FINDINGS:</h4>
        {', '.join([d.upper() for d in st.session_state.r_detected]) if st.session_state.r_detected else 'No significant abnormalities detected.'}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c2.button("🔄 NEW ANALYSIS", use_container_width=True):
        st.session_state.workflow_step = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)