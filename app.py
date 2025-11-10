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
    st.session_state.loading_start_time = time.time()
    st.session_state.loading_metrics = {
        'start_time': time.time(),
        'video_loaded': False,
        'timeout_triggered': False
    }

if not st.session_state.app_loaded:
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
            @media only screen and (max-width: 768px) {{
                #bg-video, .particle {{ display: none !important; }}
                #fallback-bg {{
                    background: linear-gradient(to bottom, #0a0a12, #121212) !important;
                }}
                #fallback-bg::after {{
                    content: ''; position: absolute;
                    top: 40%; left: 50%;
                    width: 300px; height: 300px;
                    background: radial-gradient(circle, rgba(0, 242, 254, 0.2) 0%, transparent 70%);
                    transform: translate(-50%, -50%);
                    animation: pulse 3s ease-in-out infinite;
                }}
                #loading-bar-container {{ width: 70% !important; }}
                #loading-text {{ font-size: 11px !important; letter-spacing: 3px !important; }}
            }}
            @keyframes gradientPulse {{
                0%, 100% {{ opacity: 1; filter: hue-rotate(0deg) brightness(1); }}
                50% {{ opacity: 0.9; filter: hue-rotate(180deg) brightness(1.05); }}
            }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 0.5; transform: translate(-50%, -50%) scale(1); }} 50% {{ opacity: 1; transform: translate(-50%, -50%) scale(1.2); }} }}
            {"" if not BASE64_VIDEO else f'''
            #bg-video {{
                position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                object-fit: cover; z-index: 2; opacity: 0;
                filter: hue-rotate(180deg) contrast(1.05);
                transition: opacity 0.8s ease-in;
            }}
            #bg-video.loaded {{ opacity: 0.4; }}
            '''}
            #loading-content {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                display: flex; flex-direction: column; justify-content: center; align-items: center;
                z-index: 10;
            }}
            .particle {{
                position: absolute; width: 2px; height: 2px;
                background: #00f2fe; border-radius: 50%;
                box-shadow: 0 0 8px #00f2fe;
                animation: float-up linear infinite;
            }}
            @keyframes float-up {{
                0% {{ transform: translateY(0) scale(0); opacity: 0; }}
                20% {{ opacity: 1; }}
                100% {{ transform: translateY(-100vh) scale(1); opacity: 0; }}
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
                animation: loadProgress 5s ease-out forwards;
            }}
            @keyframes loadProgress {{ 0% {{ width: 0%; }} 100% {{ width: 100%; }} }}
            #loading-text {{
                color: #00f2fe; font-size: 16px; letter-spacing: 8px;
                font-weight: 900; text-shadow: 0 0 20px #00f2fe;
                text-transform: uppercase;
            }}
            #metrics {{
                position: fixed; bottom: 20px; right: 20px;
                background: rgba(0, 242, 254, 0.1);
                border: 1px solid rgba(0, 242, 254, 0.3);
                border-radius: 8px; padding: 10px 15px;
                color: #00f2fe; font-size: 11px; z-index: 30;
            }}
            .metric {{ margin: 3px 0; }}
            .good {{ color: #00ff41; }}
        </style>
    </head>
    <body>
        <div id="fallback-bg"></div>
        {"" if not BASE64_VIDEO else f'''<video autoplay loop muted playsinline id="bg-video"><source src="data:video/mp4;base64,{BASE64_VIDEO}" type="video/mp4"></video>'''}
        <div id="loading-content">
            <div id="loading-bar-container">
                <div id="loading-bar-bg"><div id="loading-bar"></div></div>
                <div id="loading-text">INITIALIZING...</div>
            </div>
        </div>
        <div id="metrics">
            <div class="metric"><span style="opacity:0.7">Status:</span> <span class="good" id="status">Loading...</span></div>
            <div class="metric"><span style="opacity:0.7">Time:</span> <span class="good" id="timer">0.0s</span></div>
        </div>
        <script>
            const startTime = performance.now();
            const statusEl = document.getElementById('status');
            const timerEl = document.getElementById('timer');
            const textEl = document.getElementById('loading-text');
            if (window.innerWidth > 768) {{
                for (let i = 0; i < 20; i++) {{
                    const p = document.createElement('div');
                    p.className = 'particle';
                    p.style.left = Math.random() * 100 + 'vw';
                    p.style.bottom = '-10px';
                    p.style.animationDuration = (Math.random() * 3 + 2) + 's';
                    p.style.animationDelay = (Math.random() * 2) + 's';
                    document.body.appendChild(p);
                }}
            }}
            setInterval(() => {{
                timerEl.textContent = ((performance.now() - startTime) / 1000).toFixed(1) + 's';
            }}, 100);
            const msgs = ['INITIALIZING NEURAL INTERFACE...', 'NEURAL SYNC...', 'CALIBRATING SENSORS...', 'LOADING AI MODELS...', 'SYSTEM READY'];
            let i = 0;
            setInterval(() => {{ if (i < msgs.length) textEl.textContent = msgs[i++]; }}, 1200);
            setTimeout(() => {{
                textEl.textContent = 'ACCESS GRANTED';
                textEl.style.color = '#00ff41';
                statusEl.textContent = 'Complete ✓';
            }}, 5500);
            {"" if not BASE64_VIDEO else "const v = document.getElementById('bg-video'); v.addEventListener('loadeddata', () => v.classList.add('loaded'));"}
        </script>
    </body>
    </html>
    """
    st.components.v1.html(loading_html, height=900, scrolling=False)
    TOTAL_LOAD_TIME = 1
    CHUNK_SIZE = 0.5
    chunks = int(TOTAL_LOAD_TIME / CHUNK_SIZE)
    for i in range(chunks):
        time.sleep(CHUNK_SIZE)
    total_load_time = time.time() - st.session_state.loading_start_time
    st.session_state.loading_metrics['total_time'] = total_load_time
    st.session_state.loading_metrics['method'] = 'VIDEO' if BASE64_VIDEO else 'CSS_OPTIMIZED'
    st.session_state.loading_metrics['video_loaded'] = bool(BASE64_VIDEO)
    st.session_state.app_loaded = True
    st.rerun()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        # CRITICAL FIX FOR GRAD-CAM: Removed compile=False so gradients work
        model = tf.keras.models.load_model("my_ocular_model_densenet121.keras")
        # Ensure model is trainable (required for gradients)
        model.trainable = True
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
        --neon-pink: #ff00ff; --neon-purple: #8a2be2; --matrix-green: #00ff41;
        --bg-deep: #0a0a12; --bg-card: rgba(20, 20, 40, 0.8);
        --text-glow: 0 0 10px rgba(0, 242, 254, 0.7);
    }
    .stApp {
        background: radial-gradient(circle at 20% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(0, 242, 254, 0.05) 0%, transparent 50%),
                    linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    @keyframes slideIn { 0% { opacity: 0; transform: translateX(-30px); } 100% { opacity: 1; transform: translateX(0); } }
    .slide-in { animation: slideIn 0.6s ease-out; }
    .hero-title { font-size: 5rem; font-weight: 900; letter-spacing: -3px; background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(0, 242, 254, 0.5), 0 0 60px rgba(0, 242, 254, 0.3); margin-bottom: 0; }
    .hero-subtitle { font-size: 1.8rem; color: #94a3b8; letter-spacing: 6px; text-transform: uppercase; margin-bottom: 3rem; }
    .glass-card { background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(20px) saturate(180%); -webkit-backdrop-filter: blur(20px) saturate(180%); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 24px; padding: 30px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.2); transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1); }
    .glass-card:hover { transform: translateY(-8px) scale(1.02); border-color: var(--primary); box-shadow: 0 15px 40px 0 rgba(0, 242, 254, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.3); }
    .stButton>button { background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%) !important; color: #0f172a !important; border: none !important; font-weight: 800 !important; padding: 1rem 2.5rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; border-radius: 50px !important; transition: all 0.3s ease !important; box-shadow: 0 5px 15px rgba(0, 242, 254, 0.3) !important; }
    .stButton>button:hover { transform: translateY(-3px) scale(1.05) !important; box-shadow: 0 10px 25px rgba(0, 242, 254, 0.5), 0 0 30px rgba(0, 242, 254, 0.3) !important; letter-spacing: 3px !important; }
    [data-testid="stSidebar"] { background: rgba(10, 10, 18, 0.9) !important; backdrop-filter: blur(20px) !important; border-right: 1px solid rgba(0, 242, 254, 0.2) !important; }
    .progress-stepper { display: flex; justify-content: center; align-items: center; gap: 20px; margin: 30px 0; padding: 20px; }
    .step { display: flex; align-items: center; gap: 10px; }
    .step-circle { width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem; transition: all 0.3s ease; }
    .step-circle.active { background: linear-gradient(135deg, var(--primary), var(--neon-pink)); color: #0f172a; box-shadow: 0 0 20px var(--primary); }
    .step-circle.completed { background: var(--matrix-green); color: #0f172a; }
    .step-circle.inactive { background: rgba(255,255,255,0.1); border: 2px solid rgba(255,255,255,0.3); color: #94a3b8; }
    .step-label { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .step-connector { width: 60px; height: 2px; background: rgba(255,255,255,0.2); }
    .step-connector.completed { background: var(--matrix-green); }
    @media only screen and (max-width: 768px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        .hero-title { font-size: 3rem !important; }
        .hero-subtitle { font-size: 1.2rem !important; letter-spacing: 3px !important; }
        .glass-card { padding: 15px !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; min-width: auto !important; }
        .progress-stepper { align-items: flex-start !important; gap: 5px !important; padding: 20px 5px !important; }
        .step { flex-direction: column !important; align-items: center !important; flex: 1 !important; }
        .step-circle { width: 40px !important; height: 40px !important; font-size: 1rem !important; margin-bottom: 5px !important; }
        .step-label { font-size: 0.6rem !important; text-align: center !important; white-space: normal !important; line-height: 1.1 !important; }
        .step-connector { width: 15px !important; min-width: 10px !important; margin-top: 20px !important; }
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section { background-color: rgba(0, 242, 254, 0.03); border: 2px dashed var(--primary); border-radius: 20px; padding: 30px; transition: all 0.3s ease; text-align: center; }
    .custom-upload-wrapper [data-testid='stFileUploader'] section:hover { background-color: rgba(0, 242, 254, 0.1); box-shadow: 0 0 30px rgba(0, 242, 254, 0.2); border-color: var(--neon-pink); }
</style>
""", unsafe_allow_html=True)

# ================= BACKEND FUNCTIONS =================

def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_diseases(image):
    if model is None: return None
    processed = preprocess_image(image)
    return model.predict(processed, verbose=0)[0]

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

# === GRAD-CAM FUNCTIONS (INSERTED EXACTLY AS REQUESTED) ===

def find_best_gradcam_layer(model):
    candidate_layers = ['conv5_block16_2_conv', 'conv5_block16_1_conv', 'conv5_block16_0_conv', 'conv5_block15_2_conv', 'conv5_block14_2_conv', 'pool4_conv']
    for layer_name in candidate_layers:
        try:
            model.get_layer(layer_name)
            return layer_name
        except: continue
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D): return layer.name
    return None

def make_gradcam_heatmap_fixed(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(inputs=[model.inputs], outputs=[last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None: pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None: return np.zeros((7, 7)) # Fallback if gradients fail
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_np = conv_outputs.numpy()[0]
        pooled_grads_np = pooled_grads.numpy()
        for i in range(pooled_grads_np.shape[0]): conv_outputs_np[:, :, i] *= pooled_grads_np[i]
        heatmap = np.mean(conv_outputs_np, axis=2)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
        return heatmap
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return np.zeros((7, 7))

def create_enhanced_overlay(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    img_array = np.array(img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_enhanced = np.power(heatmap_resized, 0.7)
    heatmap_uint8 = np.uint8(255 * heatmap_enhanced)
    heatmap_equalized = cv2.equalizeHist(heatmap_uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_equalized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

def generate_multi_class_gradcam_v2(img, model, disease_names, top_n=3, alpha=0.6, colormap=cv2.COLORMAP_JET):
    last_conv_layer = find_best_gradcam_layer(model)
    if not last_conv_layer: return {}
    st.session_state.gradcam_layer = last_conv_layer
    img_array = preprocess_image(img)
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    gradcam_results = {}
    for idx in top_indices:
        disease = disease_names[idx]
        heatmap = make_gradcam_heatmap_fixed(img_array, model, last_conv_layer, pred_index=idx)
        overlay = create_enhanced_overlay(img, heatmap, alpha=alpha, colormap=colormap)
        gradcam_results[disease] = {'overlay': overlay, 'probability': predictions[idx], 'heatmap': heatmap}
    return gradcam_results

# === END GRAD-CAM FUNCTIONS ===

def validate_fundus_image(image):
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        return True, "skipped", "⚠ Gemini API not configured - validation skipped"
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = """You are an expert ophthalmologist. Analyze this image and determine if it is a RETINAL FUNDUS PHOTOGRAPH.
RESPOND ONLY WITH A JSON OBJECT: { "is_fundus": true/false, "confidence": "high/medium/low", "reason": "brief explanation", "visible_structures": ["list structures"], "image_quality": "excellent/good/fair/poor", "recommendation": "proceed/retake" }"""
        response = model_gemini.generate_content([prompt, image])
        import json
        result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
        is_valid = result.get('is_fundus', False)
        msg = f"✅ **Valid Fundus Image**\n**Confidence:** {result.get('confidence','').upper()}\n**Assessment:** {result.get('reason','')}" if is_valid else f"⚠️ **Invalid Image**\n**Reason:** {result.get('reason','')}"
        return is_valid, result.get('confidence', 'unknown'), msg
    except Exception as e:
        return True, "error", f"⚠️ Validation failed ({str(e)}) - proceeding"

def generate_llm_report(left_results, right_results, left_image, right_image, patient_info):
    if not st.session_state.get('gemini_api_key'): return None
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        def fmt(res): return "\n".join([f"• {r['disease']}: {r['probability']:.1%} {'[DETECTED]' if r['detected'] else ''}" for r in res])
        prompt = f"""Patient: {patient_info['name']}, {patient_info['age']}y, {patient_info['gender']}\nHistory: {patient_info['history']}\n\nLEFT EYE:\n{fmt(left_results)}\n\nRIGHT EYE:\n{fmt(right_results)}\n\nProvide a professional ophthalmology clinical report with: 1. Summary of Findings, 2. Detailed Analysis (Left/Right), 3. Clinical Significance, 4. Recommendations."""
        response = model_gemini.generate_content([prompt, "Left Fundus:", left_image, "Right Fundus:", right_image])
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None

import streamlit.components.v1 as components
def results_card_enhanced(eye_side, predictions, use_optimal_thresholds=True):
    results, detected = format_predictions(predictions, use_optimal_thresholds)
    sorted_preds = sorted(results, key=lambda x: x['probability'], reverse=True)
    top_finding = sorted_preds[0]
    is_normal = top_finding['disease'] == 'Normal'
    status_color = "#10b981" if is_normal else ("#f59e0b" if top_finding['probability'] < 0.7 else "#ef4444")
    status_icon = "✅" if is_normal else "⚠️" if top_finding['probability'] < 0.7 else "🚨"
    html_content = f"""<style>body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: transparent; color: #e2e8f0; }} .glass-card {{ background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 20px; border-top: 4px solid {status_color}; }} .tech-bar-bg {{ background: rgba(0,0,0,0.3); height: 10px; border-radius: 10px; overflow: hidden; margin: 8px 0; }} .tech-bar-fill {{ height: 100%; box-shadow: 0 0 10px var(--primary); transition: width 1s ease-out; }}</style><div class="glass-card"><h3 style="margin-top: 0; display: flex; justify-content: space-between; align-items: center;"><span>{eye_side}</span><span style="font-size: 2rem;">{status_icon}</span></h3><div style="font-size: 1.2rem; color: {status_color}; font-weight: 600; margin-bottom: 20px;">{top_finding['disease']} ({top_finding['probability']:.1%})</div>"""
    for pred in sorted_preds[:4]:
        prob = pred['probability'] * 100
        color = "#10b981" if pred['disease'] == 'Normal' else ("#f59e0b" if prob < 50 else "#ef4444")
        html_content += f"""<div style="margin-bottom: 15px;"><div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem;"><span>{pred['disease']}</span><span style="color: {color}; font-weight: bold;">{prob:.1f}%</span></div><div class="tech-bar-bg"><div class="tech-bar-fill" style="width: {prob}%; background: linear-gradient(90deg, {color}, {color}dd);"></div></div></div>"""
    html_content += "</div>"
    components.html(html_content, height=350, scrolling=False)
    return results, detected

def render_progress_stepper(current_step):
    s1, s2, s3 = ("active" if current_step==1 else "completed"), ("active" if current_step==2 else ("completed" if current_step>2 else "inactive")), ("active" if current_step==3 else "inactive")
    c1, c2 = ("completed" if current_step>1 else ""), ("completed" if current_step>2 else "")
    st.markdown(f"""<div class="progress-stepper"><div class="step"><div class="step-circle {s1}">1</div><div class="step-label">Data & Upload</div></div><div class="step-connector {c1}"></div><div class="step"><div class="step-circle {s2}">2</div><div class="step-label">Diagnostics</div></div><div class="step-connector {c2}"></div><div class="step"><div class="step-circle {s3}">3</div><div class="step-label">Report</div></div></div>""", unsafe_allow_html=True)

# ================= MAIN APP =================
with st.sidebar:
    st.markdown("## 🎮 CONTROL DECK")
    if st.session_state.get('gemini_api_key'): st.success("✓ AI Report: Active")
    else: st.error("❌ AI Report: Unavailable")
    if model is not None: st.success("✓ CNN Model: Loaded")
    else: st.error("❌ CNN Model: Not Loaded")
    
    st.markdown("---")
    st.markdown("### 🔬 Explainability (Grad-CAM)")
    show_gradcam = st.toggle("Enable Grad-CAM", value=True)
    if show_gradcam:
        gradcam_alpha = st.slider("Heatmap Intensity", 0.3, 0.9, 0.6, 0.1)
        gradcam_top_n = st.slider("Top N Classes", 1, 4, 3)
        colormap_options = {"JET (Red-Blue)": cv2.COLORMAP_JET, "Hot (Red-Yellow)": cv2.COLORMAP_HOT, "Viridis": cv2.COLORMAP_VIRIDIS}
        selected_cmap = st.selectbox("Color Scheme", list(colormap_options.keys()))
        st.session_state.gradcam_colormap = colormap_options[selected_cmap]
    
    st.markdown("---")
    st.markdown(f"**Workflow Step:** {st.session_state.workflow_step}/3")

st.markdown("""<div style="text-align: center; margin-bottom: 30px;"><h1 class="hero-title">OCULUS PRIME</h1><div class="hero-subtitle">NEURAL RETINAL INTERFACE</div></div>""", unsafe_allow_html=True)
render_progress_stepper(st.session_state.workflow_step)

# ================= STEP 1: PATIENT DATA & UPLOAD =================
if st.session_state.workflow_step == 1:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card"><h3 style="margin-top: 0;">👤 PATIENT DATA</h3>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1: p_name = st.text_input("NAME", placeholder="Patient identifier...")
        with c2: p_age = st.number_input("AGE", 1, 120, 45)
        c3, c4 = st.columns([1, 2])
        with c3: p_gen = st.selectbox("SEX", ["M", "F", "Other"])
        with c4: p_hist = st.text_area("HISTORY", height=80, placeholder="Medical history...")
        st.session_state.patient = {'name': p_name, 'age': p_age, 'gender': p_gen, 'history': p_hist}
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📸 RETINAL IMAGE CAPTURE")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card"><h3>👁 LEFT EYE (OS)</h3>', unsafe_allow_html=True)
        l_file = st.file_uploader("Upload Left Eye", type=['png','jpg','jpeg'], key='l_up')
        if l_file: st.session_state.l_img = Image.open(l_file)
        if 'l_img' in st.session_state: st.image(st.session_state.l_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card"><h3>👁 RIGHT EYE (OD)</h3>', unsafe_allow_html=True)
        r_file = st.file_uploader("Upload Right Eye", type=['png','jpg','jpeg'], key='r_up')
        if r_file: st.session_state.r_img = Image.open(r_file)
        if 'r_img' in st.session_state: st.image(st.session_state.r_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 INITIATE NEURAL ANALYSIS", use_container_width=True, type="primary"):
        if 'l_img' in st.session_state and 'r_img' in st.session_state:
            if model is None: st.error("❌ Model not loaded.")
            else:
                # 1. Validation
                val_needed = st.session_state.get('gemini_api_key') is not None
                if val_needed:
                    with st.spinner("🔍 Validating images with AI..."):
                        l_val = validate_fundus_image(st.session_state.l_img)
                        r_val = validate_fundus_image(st.session_state.r_img)
                        st.session_state.l_validation = {'valid': l_val[0], 'confidence': l_val[1], 'message': l_val[2]}
                        st.session_state.r_validation = {'valid': r_val[0], 'confidence': r_val[1], 'message': r_val[2]}
                        if not l_val[0] or not r_val[0]:
                            st.error("🚫 Invalid images detected.")
                            st.stop()

                # 2. CNN Analysis & Grad-CAM
                with st.spinner("🧠 Running CNN & Grad-CAM analysis..."):
                    st.session_state.l_pred = predict_diseases(st.session_state.l_img)
                    st.session_state.r_pred = predict_diseases(st.session_state.r_img)
                    
                    if show_gradcam:
                        cmap = st.session_state.get('gradcam_colormap', cv2.COLORMAP_JET)
                        st.session_state.l_gradcam = generate_multi_class_gradcam_v2(st.session_state.l_img, model, DISEASE_NAMES, top_n=gradcam_top_n, alpha=gradcam_alpha, colormap=cmap)
                        st.session_state.r_gradcam = generate_multi_class_gradcam_v2(st.session_state.r_img, model, DISEASE_NAMES, top_n=gradcam_top_n, alpha=gradcam_alpha, colormap=cmap)

                    st.session_state.results_ready = True
                    st.session_state.workflow_step = 2
                    st.rerun()
        else: st.error("❌ Please upload both images")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= STEP 2: DIAGNOSTICS =================
elif st.session_state.workflow_step == 2:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    st.markdown("## 🔬 DIAGNOSTIC RESULTS")
    if st.session_state.get('results_ready'):
        col1, col2 = st.columns(2)
        
        # --- LEFT EYE RESULTS ---
        with col1:
            left_tabs = st.tabs(["👁 Original", "🔥 AI Explainability"])
            with left_tabs[0]:
                st.image(st.session_state.l_img, caption="Left Eye (OS)", use_column_width=True)
            with left_tabs[1]:
                if 'l_gradcam' in st.session_state and st.session_state.l_gradcam:
                    gc_cols = st.columns(len(st.session_state.l_gradcam))
                    for i, (disease, data) in enumerate(st.session_state.l_gradcam.items()):
                        with gc_cols[i]:
                            st.image(data['overlay'], caption=f"{disease}", use_column_width=True)
                else: st.info("Grad-CAM not enabled for this run.")
            
            l_res, l_det = results_card_enhanced("LEFT EYE (OS)", st.session_state.l_pred)
            st.session_state.l_res, st.session_state.l_detected = l_res, l_det

        # --- RIGHT EYE RESULTS ---
        with col2:
            right_tabs = st.tabs(["👁 Original", "🔥 AI Explainability"])
            with right_tabs[0]:
                st.image(st.session_state.r_img, caption="Right Eye (OD)", use_column_width=True)
            with right_tabs[1]:
                if 'r_gradcam' in st.session_state and st.session_state.r_gradcam:
                    gc_cols = st.columns(len(st.session_state.r_gradcam))
                    for i, (disease, data) in enumerate(st.session_state.r_gradcam.items()):
                        with gc_cols[i]:
                            st.image(data['overlay'], caption=f"{disease}", use_column_width=True)
                else: st.info("Grad-CAM not enabled for this run.")

            r_res, r_det = results_card_enhanced("RIGHT EYE (OD)", st.session_state.r_pred)
            st.session_state.r_res, st.session_state.r_detected = r_res, r_det

        # Navigation
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
             if st.button("⬅️ BACK", use_container_width=True):
                st.session_state.workflow_step = 1
                st.rerun()
        with c3:
            if st.button("GENERATE REPORT ➡️", use_container_width=True, type="primary"):
                if st.session_state.get('gemini_api_key'):
                    with st.spinner("🤖 Generating clinical report..."):
                        st.session_state.clinical_report = generate_llm_report(l_res, r_res, st.session_state.l_img, st.session_state.r_img, st.session_state.patient)
                st.session_state.workflow_step = 3
                st.rerun()
    else:
        st.session_state.workflow_step = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= STEP 3: REPORT =================
elif st.session_state.workflow_step == 3:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    if st.session_state.get('results_ready'):
        st.markdown("""<div style='text-align: center; margin-bottom: 30px;'><h1 style='color: var(--primary); font-size: 3rem; margin-bottom: 0.5rem;'>📊 CLINICAL REPORT</h1><p style='color: #94a3b8; font-size: 1.2rem;'>Comprehensive Diagnostic Analysis</p></div>""", unsafe_allow_html=True)
        
        # Simple Report Display
        if 'clinical_report' in st.session_state and st.session_state.clinical_report:
             st.markdown(f"<div class='glass-card'>{st.session_state.clinical_report}</div>", unsafe_allow_html=True)
        else:
            st.warning("AI report not generated (Check API Key)")
            c1, c2 = st.columns(2)
            with c1: results_card_enhanced("LEFT EYE", st.session_state.l_pred)
            with c2: results_card_enhanced("RIGHT EYE", st.session_state.r_pred)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⬅️ BACK TO DIAGNOSIS", use_container_width=True):
                st.session_state.workflow_step = 2
                st.rerun()
        with c2:
            if st.button("🔄 NEW ANALYSIS", use_container_width=True):
                for k in list(st.session_state.keys()):
                    if k not in ['app_loaded', 'loading_metrics', 'gemini_api_key']: del st.session_state[k]
                st.session_state.workflow_step = 1
                st.rerun()
        with c3:
            if st.button("🏁 COMPLETE", use_container_width=True, type="primary"):
                st.balloons()
                st.success("Analysis completed successfully!")
    st.markdown('</div>', unsafe_allow_html=True)