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
import streamlit.components.v1 as components

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
    st.session_state.loading_start_time = time.time()
    st.session_state.loading_metrics = {
        'start_time': time.time(),
        'video_loaded': False,
        'timeout_triggered': False
    }

if not st.session_state.app_loaded:
    BASE64_VIDEO = ""  # Paste base64 video if desired
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
                <div id="loading-text">INITIALIZING...</div>
            </div>
        </div>
        <script>
            setTimeout(() => {{
                document.getElementById('loading-text').textContent = 'SYSTEM READY';
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

# ================= CONSTANTS =================
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

OPTIMAL_THRESHOLDS = {
    'Normal': 0.514, 'Diabetes': 0.300, 'Glaucoma': 0.531, 'Cataract': 0.682,
    'AMD': 0.517, 'Hypertension': 0.533, 'Myopia': 0.529, 'Other': 0.256
}

# ================= OPTIMIZED GRAD-CAM IMPLEMENTATION =================

def get_densenet121_gradcam_layer(model):
    """Get optimal Grad-CAM layer for DenseNet121 models."""
    # Priority 1: Known DenseNet121 layers
    DENSENET121_LAYERS = [
        'conv5_block16_2_conv',  # Best - last Conv2D in final dense block
        'conv5_block16_1_conv',  # Backup - second to last Conv2D
        'relu',                    # Final ReLU activation
        'conv5_block16_concat',   # Concatenation layer
    ]
    
    layer_names = [layer.name for layer in model.layers]
    for preferred_layer in DENSENET121_LAYERS:
        if preferred_layer in layer_names:
            try:
                layer = model.get_layer(preferred_layer)
                if len(layer.output_shape) == 4:
                    return preferred_layer
            except:
                continue
    
    # Priority 2: Find last Conv2D layer
    for layer in reversed(model.layers):
        if 'Conv2D' in type(layer).__name__:
            if len(layer.output_shape) == 4:
                return layer.name
                
    # Priority 3: Any 4D layer
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
            return layer.name
            
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap using a specific layer."""
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        return None

def create_gradcam_overlay(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image."""
    try:
        img_array = np.array(image.resize((224, 224)))
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlayed_img = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
        return Image.fromarray(overlayed_img)
    except Exception:
        return image

def predict_diseases_with_gradcam(image, model, disease_names):
    """Get predictions and generate Grad-CAMs for top classes."""
    if model is None: return None, None

    # Preprocess
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    # Grad-CAM
    gradcams = {}
    try:
        # Use cached layer name if available
        if 'gradcam_layer' not in st.session_state:
             st.session_state.gradcam_layer = get_densenet121_gradcam_layer(model)
        
        layer_name = st.session_state.gradcam_layer

        if layer_name:
            # Generate for top 3 predictions with >10% confidence
            top_indices = np.argsort(predictions)[-3:][::-1]
            for idx in top_indices:
                conf = predictions[idx]
                if conf > 0.1:
                    heatmap = make_gradcam_heatmap(img_array, model, layer_name, idx)
                    if heatmap is not None:
                        overlay = create_gradcam_overlay(image, heatmap, alpha=0.5)
                        gradcams[disease_names[idx]] = {'image': overlay, 'confidence': conf}
    except Exception as e:
        logger.error(f"Grad-CAM generation error: {e}")

    return predictions, gradcams if gradcams else None

# ================= MODEL LOADING =================
@st.cache_resource
def load_model():
    try:
        logger.info("🏗️ Loading model...")
        full_model = tf.keras.models.load_model('my_ocular_model_densenet121.keras', compile=False)
        full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("✅ Model loaded successfully!")
        return full_model
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Initialize Grad-CAM layer once after model load
if model is not None and 'gradcam_layer' not in st.session_state:
    st.session_state.gradcam_layer = get_densenet121_gradcam_layer(model)
    if st.session_state.gradcam_layer:
        logger.info(f"🎯 Grad-CAM initialized with layer: {st.session_state.gradcam_layer}")
    else:
        logger.warning("⚠️ No suitable Grad-CAM layer found.")

# ================= MAIN APP CSS =================
st.markdown("""
<style>
    :root { --primary: #00f2fe; --neon-pink: #ff00ff; --bg-deep: #0a0a12; }
    .stApp { background: radial-gradient(circle at 20% 80%, rgba(79,172,254,0.1) 0%, transparent 50%), linear-gradient(135deg, #0a0a12 0%, #16213e 100%); color: #e2e8f0; }
    .glass-card { background: rgba(255,255,255,0.05); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.1); border-radius: 24px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .stButton>button { background: linear-gradient(135deg, var(--primary), var(--neon-pink)) !important; color: #0f172a !important; border: none !important; border-radius: 50px !important; font-weight: 800 !important; }
    .hero-title { font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, var(--primary), var(--neon-pink)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .progress-stepper { display: flex; justify-content: center; gap: 10px; margin: 20px 0; }
    .step-circle { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; }
    .step-circle.active { background: var(--primary); color: #000; box-shadow: 0 0 15px var(--primary); }
    .step-circle.completed { background: #00ff41; color: #000; }
    .step-circle.inactive { background: rgba(255,255,255,0.1); color: #666; }
</style>
""", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================

def format_predictions(predictions, use_optimal_thresholds=True):
    results, detected = [], []
    for i, prob in enumerate(predictions):
        disease = DISEASE_NAMES[i]
        threshold = OPTIMAL_THRESHOLDS[disease] if use_optimal_thresholds else 0.5
        is_detected = prob >= threshold
        results.append({'disease': disease, 'probability': float(prob), 'threshold': threshold, 'detected': is_detected})
        if is_detected and disease != 'Normal': detected.append(disease)
    return results, detected

def results_card_enhanced(eye_side, predictions):
    results, detected = format_predictions(predictions)
    sorted_preds = sorted(results, key=lambda x: x['probability'], reverse=True)
    top = sorted_preds[0]
    color = "#10b981" if top['disease'] == 'Normal' else ("#f59e0b" if top['probability'] < 0.7 else "#ef4444")
    
    html = f"""<div class="glass-card" style="border-top: 4px solid {color}">
    <h3>{eye_side}</h3><div style="font-size: 1.2rem; color: {color}; font-weight: bold; margin-bottom: 15px;">
    {top['disease']} ({top['probability']:.1%})</div>"""
    
    for p in sorted_preds[:4]:
        c = "#10b981" if p['disease'] == 'Normal' else ("#f59e0b" if p['probability'] < 0.5 else "#ef4444")
        html += f"""<div style="margin-bottom: 8px;"><div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
        <span>{p['disease']}</span><span style="color:{c}">{p['probability']:.1%}</span></div>
        <div style="height: 6px; background: rgba(0,0,0,0.3); border-radius: 3px;"><div style="width:{p['probability']*100}%; height:100%; background:{c}; border-radius:3px;"></div></div></div>"""
    html += "</div>"
    components.html(html, height=350, scrolling=False)
    return results, detected

def generate_llm_report(l_res, r_res, patient):
    if not st.session_state.gemini_api_key: return None
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        l_txt = "\n".join([f"- {r['disease']}: {r['probability']:.1%} ({'DETECTED' if r['detected'] else 'Normal'})" for r in l_res])
        r_txt = "\n".join([f"- {r['disease']}: {r['probability']:.1%} ({'DETECTED' if r['detected'] else 'Normal'})" for r in r_res])
        
        prompt = f"""Act as an ophthalmologist. Generate a brief clinical report for patient {patient['name']} ({patient['age']}y/{patient['gender']}).
        History: {patient['history']}
        CNN Results Left Eye:\n{l_txt}
        CNN Results Right Eye:\n{r_txt}
        Provide: 1. Summary of Findings, 2. Assessment, 3. Recommendations. Keep it professional and concise."""
        
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"

# ================= APP STRUCTURE =================

st.markdown('<h1 class="hero-title">OCULUS PRIME</h1>', unsafe_allow_html=True)

# Stepper
s1 = "active" if st.session_state.workflow_step == 1 else "completed"
s2 = "inactive" if st.session_state.workflow_step == 1 else ("active" if st.session_state.workflow_step == 2 else "completed")
s3 = "inactive" if st.session_state.workflow_step < 3 else "active"
st.markdown(f"""<div class="progress-stepper">
<div class="step-circle {s1}">1</div><div class="step-circle {s2}">2</div><div class="step-circle {s3}">3</div>
</div>""", unsafe_allow_html=True)

# --- STEP 1: UPLOAD ---
if st.session_state.workflow_step == 1:
    col1, col2 = st.columns([2, 1])
    with col1: st.session_state.p_name = st.text_input("Patient Name")
    with col2: st.session_state.p_age = st.number_input("Age", 1, 120, 45)
    st.session_state.p_hist = st.text_area("Medical History", height=70)
    st.session_state.patient = {'name': st.session_state.p_name, 'age': st.session_state.p_age, 'gender': 'N/A', 'history': st.session_state.p_hist}

    c1, c2 = st.columns(2)
    with c1:
        lf = st.file_uploader("Upload Left Eye (OS)", type=['jpg','png','jpeg'], key='l')
        if lf: st.session_state.l_img = Image.open(lf)
        if 'l_img' in st.session_state: st.image(st.session_state.l_img, use_column_width=True)
    with c2:
        rf = st.file_uploader("Upload Right Eye (OD)", type=['jpg','png','jpeg'], key='r')
        if rf: st.session_state.r_img = Image.open(rf)
        if 'r_img' in st.session_state: st.image(st.session_state.r_img, use_column_width=True)

    if st.button("INITIATE ANALYSIS 🚀", type="primary", use_container_width=True):
        if 'l_img' in st.session_state and 'r_img' in st.session_state and model:
            with st.spinner("Running neural analysis..."):
                # Use new optimized function
                st.session_state.l_pred, st.session_state.l_gradcams = predict_diseases_with_gradcam(st.session_state.l_img, model, DISEASE_NAMES)
                st.session_state.r_pred, st.session_state.r_gradcams = predict_diseases_with_gradcam(st.session_state.r_img, model, DISEASE_NAMES)
                st.session_state.results_ready = True
                st.session_state.workflow_step = 2
                st.rerun()
        else:
            st.error("Please upload both images.")

# --- STEP 2: RESULTS ---
elif st.session_state.workflow_step == 2:
    if st.session_state.get('results_ready'):
        c1, c2 = st.columns(2)
        with c1:
            st.image(st.session_state.l_img, caption="Left Eye (OS)", use_column_width=True)
            st.session_state.l_res, st.session_state.l_det = results_card_enhanced("LEFT EYE", st.session_state.l_pred)
        with c2:
            st.image(st.session_state.r_img, caption="Right Eye (OD)", use_column_width=True)
            st.session_state.r_res, st.session_state.r_det = results_card_enhanced("RIGHT EYE", st.session_state.r_pred)

        st.markdown("### 🔬 Grad-CAM Analysis")
        t1, t2 = st.tabs(["Left Eye Heatmaps", "Right Eye Heatmaps"])
        with t1:
            if st.session_state.l_gradcams:
                cols = st.columns(len(st.session_state.l_gradcams))
                for i, (d, data) in enumerate(st.session_state.l_gradcams.items()):
                    with cols[i]:
                        st.image(data['image'], caption=f"{d} ({data['confidence']:.1%})", use_column_width=True)
            else:
                st.info("No significant findings to visualize for Left Eye.")
        with t2:
            if st.session_state.r_gradcams:
                cols = st.columns(len(st.session_state.r_gradcams))
                for i, (d, data) in enumerate(st.session_state.r_gradcams.items()):
                    with cols[i]:
                        st.image(data['image'], caption=f"{d} ({data['confidence']:.1%})", use_column_width=True)
            else:
                st.info("No significant findings to visualize for Right Eye.")

        c1, c2 = st.columns([1, 2])
        with c1: 
            if st.button("⬅ BACK", use_container_width=True):
                st.session_state.workflow_step = 1
                st.rerun()
        with c2:
            if st.button("GENERATE REPORT 📄", type="primary", use_container_width=True):
                with st.spinner("Generating AI Report..."):
                    st.session_state.report = generate_llm_report(st.session_state.l_res, st.session_state.r_res, st.session_state.patient)
                    st.session_state.workflow_step = 3
                    st.rerun()

# --- STEP 3: REPORT ---
elif st.session_state.workflow_step == 3:
    st.markdown("## 📋 CLINICAL REPORT")
    if 'report' in st.session_state and st.session_state.report:
        st.markdown(f"<div class='glass-card'>{st.session_state.report}</div>", unsafe_allow_html=True)
    else:
        st.warning("AI Report generation failed or API key missing.")
    
    if st.button("🔄 START OVER", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k not in ['gemini_api_key', 'app_loaded', 'gradcam_layer']: del st.session_state[k]
        st.session_state.workflow_step = 1
        st.rerun()