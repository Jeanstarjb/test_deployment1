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
# Initialize first, before any checks
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False
    st.session_state.loading_start_time = time.time()
    st.session_state.loading_metrics = {
        'start_time': time.time(),
        'video_loaded': False,
        'timeout_triggered': False
    }

if not st.session_state.app_loaded:
    # PASTE YOUR BASE64 VIDEO STRING HERE
    BASE64_VIDEO = ""  # ← PASTE YOUR BASE64 VIDEO STRING HERE (leave empty for no video)
    # Simplified loading HTML (same as before)
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
            
            /* --- DESKTOP BACKGROUND --- */
            #fallback-bg {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: radial-gradient(circle at 20% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%),
                            radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
                            linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
                z-index: 1;
                animation: gradientPulse 4s ease-in-out infinite;
            }}

            /* --- MOBILE SPECIFIC VIEW (The fix for "ridiculously big") --- */
            @media only screen and (max-width: 768px) {{
                /* 1. Hide the big video and particles on phone to save space/processing */
                #bg-video, .particle {{ display: none !important; }}
                
                /* 2. Simpler, cleaner background for phone */
                #fallback-bg {{
                    background: linear-gradient(to bottom, #0a0a12, #121212) !important;
                }}
                /* 3. Add a cool simplified glowing orb for mobile instead of video */
                #fallback-bg::after {{
                    content: ''; position: absolute;
                    top: 40%; left: 50%;
                    width: 300px; height: 300px;
                    background: radial-gradient(circle, rgba(0, 242, 254, 0.2) 0%, transparent 70%);
                    transform: translate(-50%, -50%);
                    animation: pulse 3s ease-in-out infinite;
                }}
                
                /* 4. Shrink text and loading bar to fit phone screen comfortably */
                #loading-bar-container {{ width: 70% !important; }}
                #loading-text {{ font-size: 11px !important; letter-spacing: 3px !important; }}
            }}
            
            @keyframes gradientPulse {{
                0%, 100% {{ opacity: 1; filter: hue-rotate(0deg) brightness(1); }}
                50% {{ opacity: 0.9; filter: hue-rotate(180deg) brightness(1.05); }}
            }}
            
            @keyframes pulse {{ 0%, 100% {{ opacity: 0.5; transform: translate(-50%, -50%) scale(1); }} 50% {{ opacity: 1; transform: translate(-50%, -50%) scale(1.2); }} }}

            /* --- VIDEO BACKGROUND --- */
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
            
            // Only create particles on wider screens to save mobile performance
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
    
    # Display loading screen
    st.components.v1.html(loading_html, height=900, scrolling=False)
    
    # ⭐ SAFE SLEEP METHOD - breaks into smaller chunks
    TOTAL_LOAD_TIME = 1  # ← ADJUST THIS (in seconds)
    CHUNK_SIZE = 0.5  # Sleep in 0.5 second chunks
    
    chunks = int(TOTAL_LOAD_TIME / CHUNK_SIZE)
    for i in range(chunks):
        time.sleep(CHUNK_SIZE)
    
    # Complete loading
    total_load_time = time.time() - st.session_state.loading_start_time
    st.session_state.loading_metrics['total_time'] = total_load_time
    st.session_state.loading_metrics['method'] = 'VIDEO' if BASE64_VIDEO else 'CSS_OPTIMIZED'
    st.session_state.loading_metrics['video_loaded'] = bool(BASE64_VIDEO)
    
    try:
        logger.info(f"✅ Loading complete - Total time: {total_load_time:.2f}s")
    except:
        print(f"✅ Loading complete - Total time: {total_load_time:.2f}s")
    
    st.session_state.app_loaded = True
    st.rerun()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        logger.info("🏗 Loading model from .keras file...")
        
        # STEP 1: Load the complete model from .keras file
        full_model = tf.keras.models.load_model('my_ocular_model_densenet121.keras', compile=False)
        logger.info("✅ Model architecture loaded from .keras file!")
        
        # STEP 3: Compile the model
        full_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✅ Model ready for inference!")
        logger.info(f"📊 Model has {len(full_model.layers)} layers")
        
        return full_model
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        st.error(f"Failed to load model. Error: {e}")
        return None

# Initialize model
model = load_model()

# ================= OPTIMIZED GRAD-CAM LAYER DETECTION =================

def get_densenet121_gradcam_layer(model):
    """
    Get the optimal Grad-CAM layer for DenseNet121 models.
    
    Returns the name of the best convolutional layer for Grad-CAM visualization.
    This function checks in order of preference:
    1. Architecture-specific known layers
    2. Last Conv2D layer with 4D output
    3. Fallback to any 4D layer
    
    Args:
        model: TensorFlow/Keras model
        
    Returns:
        str: Layer name suitable for Grad-CAM, or None if not found
    """
    
    # Priority 1: Known DenseNet121 layers (in order of preference)
    DENSENET121_LAYERS = [
        'conv5_block16_2_conv',  # Best - last Conv2D in final dense block
        'conv5_block16_1_conv',  # Backup - second to last Conv2D
        'relu',                   # Final ReLU activation
        'conv5_block16_concat',   # Concatenation layer (less ideal but works)
    ]
    
    logger.info("🔍 Searching for optimal Grad-CAM layer for DenseNet121...")
    
    # Try known layers first
    layer_names = [layer.name for layer in model.layers]
    
    for preferred_layer in DENSENET121_LAYERS:
        if preferred_layer in layer_names:
            try:
                layer = model.get_layer(preferred_layer)
                if len(layer.output_shape) == 4:  # Verify it's 4D
                    logger.info(f"✅ Found DenseNet121 layer: {preferred_layer}")
                    logger.info(f"   Output shape: {layer.output_shape}")
                    return preferred_layer
            except Exception as e:
                logger.warning(f"⚠ Layer {preferred_layer} found but invalid: {e}")
                continue
    
    # Priority 2: Find last Conv2D layer
    logger.info("📍 Searching for last Conv2D layer...")
    
    for layer in reversed(model.layers):
        layer_type = type(layer).__name__
        
        # Check for Conv2D or similar convolutional layers
        if 'Conv2D' in layer_type or 'SeparableConv2D' in layer_type:
            try:
                if len(layer.output_shape) == 4:
                    logger.info(f"✅ Found Conv2D layer: {layer.name}")
                    logger.info(f"   Type: {layer_type}")
                    logger.info(f"   Output shape: {layer.output_shape}")
                    return layer.name
            except:
                continue
    
    # Priority 3: Find any 4D layer (last resort)
    logger.info("📍 Searching for any 4D layer...")
    
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                logger.info(f"✅ Found 4D layer: {layer.name}")
                logger.info(f"   Type: {type(layer).__name__}")
                logger.info(f"   Output shape: {layer.output_shape}")
                return layer.name
        except:
            continue
    
    # Nothing found
    logger.error("❌ No suitable Grad-CAM layer found!")
    return None


def verify_gradcam_layer(model, layer_name):
    """
    Verify that a layer is suitable for Grad-CAM.
    
    Args:
        model: TensorFlow/Keras model
        layer_name: Name of the layer to verify
        
    Returns:
        bool: True if layer is valid for Grad-CAM
    """
    try:
        layer = model.get_layer(layer_name)
        
        # Check 1: Must have 4D output
        if len(layer.output_shape) != 4:
            logger.error(f"❌ Layer {layer_name} has wrong shape: {layer.output_shape}")
            return False
        
        # Check 2: Must have reasonable spatial dimensions
        _, h, w, c = layer.output_shape
        if h is None or w is None or c is None:
            logger.error(f"❌ Layer {layer_name} has None dimensions")
            return False
        
        if h < 4 or w < 4:
            logger.warning(f"⚠ Layer {layer_name} has very small spatial size: {h}x{w}")
        
        logger.info(f"✅ Layer {layer_name} verified for Grad-CAM")
        logger.info(f"   Spatial: {h}x{w}, Channels: {c}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying layer {layer_name}: {e}")
        return False

# Detect optimal Grad-CAM layer on startup
if model is not None:
    logger.info("=== GRAD-CAM LAYER DETECTION ===")
    optimal_gradcam_layer = get_densenet121_gradcam_layer(model)
    if optimal_gradcam_layer:
        logger.info(f"✅ Will use layer: {optimal_gradcam_layer}")
    else:
        logger.error("❌ No suitable Grad-CAM layer found!")
        optimal_gradcam_layer = None
else:
    optimal_gradcam_layer = None

# ================= CONSTANTS =================
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

# Optimal thresholds for each disease
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
        --primary-glow: #00f2feaa;
        --primary-dark: #4facfe;
        --neon-pink: #ff00ff;
        --neon-purple: #8a2be2;
        --matrix-green: #00ff41;
        --bg-deep: #0a0a12;
        --bg-card: rgba(20, 20, 40, 0.8);
        --text-glow: 0 0 10px rgba(0, 242, 254, 0.7);
    }

    .stApp {
        background: 
            radial-gradient(circle at 20% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 242, 254, 0.05) 0%, transparent 50%),
            linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', system-ui, sans-serif;
        min-height: 100vh;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem;}

    @keyframes float { 0%, 100% { transform: translateY(0px) rotate(0deg); } 50% { transform: translateY(-10px) rotate(1deg); } }
    @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.05); opacity: 0.8; } 100% { transform: scale(1); opacity: 1; } }
    @keyframes slideIn { 0% { opacity: 0; transform: translateX(-30px); } 100% { opacity: 1; transform: translateX(0); } }
    .floating { animation: float 6s ease-in-out infinite; }
    .pulse-glow { animation: pulse 2s ease-in-out infinite; }
    .slide-in { animation: slideIn 0.6s ease-out; }

    .hero-title { font-size: 5rem; font-weight: 900; letter-spacing: -3px; background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(0, 242, 254, 0.5), 0 0 60px rgba(0, 242, 254, 0.3); margin-bottom: 0; }
    .hero-subtitle { font-size: 1.8rem; color: #94a3b8; letter-spacing: 6px; text-transform: uppercase; margin-bottom: 3rem; }
    .glass-card { background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(20px) saturate(180%); -webkit-backdrop-filter: blur(20px) saturate(180%); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 24px; padding: 30px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.2); transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1); }
    .glass-card:hover { transform: translateY(-8px) scale(1.02); border-color: var(--primary); box-shadow: 0 15px 40px 0 rgba(0, 242, 254, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.3); }
    .stButton>button { background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%) !important; color: #0f172a !important; border: none !important; font-weight: 800 !important; padding: 1rem 2.5rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; border-radius: 50px !important; transition: all 0.3s ease !important; box-shadow: 0 5px 15px rgba(0, 242, 254, 0.3) !important; }
    .stButton>button:hover { transform: translateY(-3px) scale(1.05) !important; box-shadow: 0 10px 25px rgba(0, 242, 254, 0.5), 0 0 30px rgba(0, 242, 254, 0.3) !important; letter-spacing: 3px !important; }

    .scanner-container { position: relative; overflow: hidden; border-radius: 20px; border: 2px solid var(--primary); background: linear-gradient(45deg, rgba(0, 242, 254, 0.1) 0%, transparent 50%); }
    .scan-line { position: absolute; width: 100%; height: 6px; background: linear-gradient(90deg, transparent, var(--primary), var(--neon-pink), transparent); box-shadow: 0 0 20px var(--primary), 0 0 40px var(--neon-pink); opacity: 0.8; animation: scanline 1.5s linear infinite; z-index: 10; }
    @keyframes scanline { 0% { top: 0%; } 100% { top: 100%; } }
    .tech-bar-bg { background: rgba(0,0,0,0.3); height: 12px; border-radius: 10px; overflow: hidden; margin: 10px 0; border: 1px solid rgba(255,255,255,0.1); }
    .tech-bar-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--neon-pink)); box-shadow: 0 0 10px var(--primary); transition: width 1.5s cubic-bezier(0.23, 1, 0.32, 1); }
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

    /* --- MOBILE RESPONSIVENESS FIXES (ALL IN ONE) --- */
    @media only screen and (max-width: 768px) {
        /* 1. Fix Squeezed UI */
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        .hero-title { font-size: 3rem !important; }
        .hero-subtitle { font-size: 1.2rem !important; letter-spacing: 3px !important; }
        .glass-card { padding: 15px !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; min-width: auto !important; }
        #loading-bar-container { width: 90% !important; }
        .step-circle { width: 35px !important; height: 35px !important; font-size: 1rem !important; }
        .step-label { font-size: 0.7rem !important; }
        .step-connector { width: 30px !important; }

        /* --- New Mobile Stepper: Horizontal with labels below --- */
        .progress-stepper {
            align-items: flex-start !important;
            gap: 5px !important;
            padding: 20px 5px !important;
        }
        .step {
            flex-direction: column !important;
            align-items: center !important;
            flex: 1 !important;
        }
        .step-circle {
            width: 40px !important;
            height: 40px !important;
            font-size: 1rem !important;
            margin-bottom: 5px !important;
        }
        .step-label {
            font-size: 0.6rem !important;
            text-align: center !important;
            white-space: normal !important;
            line-height: 1.1 !important;
        }
        .step-connector {
             width: 15px !important;
             min-width: 10px !important;
             margin-top: 20px !important;
        }
    }
            
    .custom-upload-wrapper [data-testid='stFileUploader'] {
        width: 100%;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section {
        background-color: rgba(0, 242, 254, 0.03);
        border: 2px dashed var(--primary);
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
        text-align: center;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section:hover {
        background-color: rgba(0, 242, 254, 0.1);
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.2);
        border-color: var(--neon-pink);
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button {
        display: inline-block !important;
        width: auto !important;
        height: auto !important;
        opacity: 1 !important;
        background: transparent !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 50px !important;
        padding: 0.5rem 1.5rem !important;
        margin-top: 10px;
        transition: all 0.3s ease !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button:hover {
        background: var(--primary) !important;
        color: var(--bg-deep) !important;
        box-shadow: 0 0 15px var(--primary) !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section > div > div span {
        color: var(--primary) !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= BACKEND FUNCTIONS =================

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_diseases(image):
    """Get predictions from CNN model"""
    if model is None:
        st.error("❌ Model not loaded. Please check model path.")
        return None
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    return predictions

def format_predictions(predictions, use_optimal_thresholds=True):
    """Format predictions into readable format using optimal thresholds"""
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

# ================= OPTIMIZED GRAD-CAM IMPLEMENTATION =================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap with automatic layer detection.
    
    Args:
        img_array: Preprocessed image array (batch, height, width, channels)
        model: TensorFlow/Keras model
        last_conv_layer_name: Optional layer name, auto-detected if None
        pred_index: Target class index, uses max prediction if None
        
    Returns:
        numpy.ndarray: Heatmap array, or None if failed
    """
    try:
        # Auto-detect layer if not provided
        if last_conv_layer_name is None:
            last_conv_layer_name = get_densenet121_gradcam_layer(model)
            
            if last_conv_layer_name is None:
                logger.error("❌ Could not find suitable layer for Grad-CAM")
                return None
        
        # Verify layer
        if not verify_gradcam_layer(model, last_conv_layer_name):
            return None
        
        # Get the layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            logger.error("❌ Gradients are None - layer may not be trainable")
            return None
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize to [0, 1]
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        logger.info(f"✅ Grad-CAM heatmap generated successfully")
        return heatmap.numpy()
        
    except Exception as e:
        logger.error(f"❌ Grad-CAM generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_gradcam_overlay(image, heatmap, alpha=0.4):
    """Create visualization overlay of Grad-CAM heatmap"""
    try:
        import cv2
        
        img_array = np.array(image.resize((224, 224)))
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlayed_img = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
        return Image.fromarray(overlayed_img)
        
    except ImportError:
        logger.error("OpenCV not installed")
        return image
    except Exception as e:
        logger.error(f"Error creating overlay: {e}")
        return image

def predict_diseases_with_gradcam(image):
    """Get predictions with Grad-CAM using optimized layer detection"""
    if model is None:
        st.error("❌ Model not loaded")
        return None, None
    
    # Get predictions
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    
    # Generate Grad-CAMs
    gradcams = {}
    try:
        # Use the pre-detected optimal layer
        layer_name = optimal_gradcam_layer
        
        if layer_name:
            # Generate for top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            for idx in top_indices:
                disease_name = DISEASE_NAMES[idx]
                confidence = predictions[idx]
                
                if confidence > 0.1:
                    heatmap = make_gradcam_heatmap(
                        processed, 
                        model, 
                        last_conv_layer_name=layer_name,
                        pred_index=idx
                    )
                    
                    if heatmap is not None:
                        overlay = create_gradcam_overlay(image, heatmap, alpha=0.5)
                        gradcams[disease_name] = {
                            'image': overlay,
                            'confidence': confidence
                        }
        else:
            logger.warning("⚠ No suitable layer for Grad-CAM")
            
    except Exception as e:
        logger.error(f"❌ Grad-CAM error: {e}")
    
    return predictions, gradcams if gradcams else None

def validate_fundus_image(image):
    """
    Validate if uploaded image is a fundus photograph using Gemini AI
    Returns: (is_valid: bool, confidence: str, message: str)
    """
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        # If no Gemini API, skip validation
        return True, "skipped", "⚠ Gemini API not configured - validation skipped"
    
    try:
        # Configure Gemini
        genai.configure(api_key=st.session_state.gemini_api_key)
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = """You are an expert ophthalmologist. Analyze this image and determine if it is a RETINAL FUNDUS PHOTOGRAPH.

A valid fundus photograph should show:
- Optic disc (optic nerve head)
- Blood vessels radiating from optic disc
- Macula (darker area temporal to optic disc)
- Retinal background (orange/red color)
- Circular field of view (typical of fundus camera)

RESPOND ONLY WITH A JSON OBJECT (no markdown, no code blocks, just raw JSON):
{
  "is_fundus": true/false,
  "confidence": "high/medium/low",
  "reason": "brief explanation of your assessment",
  "visible_structures": ["list of visible retinal structures"],
  "image_quality": "excellent/good/fair/poor",
  "recommendation": "proceed with analysis / retake image / use different image"
}"""

        # Generate content
        response = model_gemini.generate_content([prompt, image])
        response_text = response.text.strip()
        
        # Clean response (remove markdown if present)
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join([l for l in lines if not l.startswith('```')])
            response_text = response_text.strip()
        
        # Parse JSON response
        import json
        result = json.loads(response_text)
        
        is_valid = result.get('is_fundus', False)
        confidence = result.get('confidence', 'unknown')
        reason = result.get('reason', 'No reason provided')
        quality = result.get('image_quality', 'unknown')
        recommendation = result.get('recommendation', 'Review image')
        structures = result.get('visible_structures', [])
        
        # Format message
        if is_valid:
            message = f"""✅ **Valid Fundus Image Detected**
            
**Confidence:** {confidence.upper()}
**Image Quality:** {quality.capitalize()}
**Visible Structures:** {', '.join(structures) if structures else 'Standard retinal features'}

**Assessment:** {reason}

✓ Proceeding with CNN analysis..."""
        else:
            message = f"""⚠ **Invalid or Non-Fundus Image Detected**

**Confidence:** {confidence.upper()}
**Reason:** {reason}
**Recommendation:** {recommendation}

Please upload a proper retinal fundus photograph for accurate analysis."""
        
        return is_valid, confidence, message
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response: {e}")
        logger.error(f"Raw response: {response_text}")
        return True, "error", "⚠ Image validation failed (JSON parse error) - proceeding with analysis"
        
    except Exception as e:
        logger.error(f"Error validating fundus image: {e}")
        return True, "error", f"⚠ Image validation failed ({str(e)}) - proceeding with analysis"


def generate_llm_report(left_results, right_results, left_image, right_image, patient_info):
    """Generate comprehensive medical report using Gemini"""
    
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        return None
    
    try:
        # Configure Gemini
        genai.configure(api_key=st.session_state.gemini_api_key)
        
        # Use Gemini Pro Vision model
        model_gemini = genai.GenerativeModel('gemini-2.5-flash')
        
        # Format predictions text with detected conditions highlighted
        def format_eye_predictions(results):
            lines = []
            for r in results:
                status = "✓ DETECTED" if r['detected'] else ""
                lines.append(f"  • {r['disease']:15s}: {r['probability']:6.1%} (threshold: {r['threshold']:.1%}) {status}")
            return "\n".join(lines)
        
        left_pred_text = format_eye_predictions(left_results)
        right_pred_text = format_eye_predictions(right_results)
        
        # Get detected conditions
        left_detected = [r['disease'] for r in left_results if r['detected']]
        right_detected = [r['disease'] for r in right_results if r['detected']]
        
        prompt = f"""You are an experienced ophthalmologist reviewing retinal fundus images. Provide a clear, structured clinical report.

PATIENT INFORMATION:
- Name: {patient_info['name']}
- Age: {patient_info['age']} years
- Gender: {patient_info['gender']}
- Medical History: {patient_info['history'] or 'None provided'}

CNN MODEL PREDICTIONS (using optimized thresholds):

LEFT EYE (OS):
{left_pred_text}
Detected Conditions: {', '.join(left_detected) if left_detected else 'None'}

RIGHT EYE (OD):
{right_pred_text}
Detected Conditions: {', '.join(right_detected) if right_detected else 'None'}

Please analyze both retinal images and provide a comprehensive clinical report with the following sections:

1. SUMMARY OF FINDINGS
   - Brief overview of key findings in both eyes
   - Comparison between left and right eyes

2. DETAILED ANALYSIS
   - LEFT EYE: Describe visible pathological features and validate CNN predictions
   - RIGHT EYE: Describe visible pathological features and validate CNN predictions
   - Note any asymmetry or bilateral findings

3. CLINICAL SIGNIFICANCE
   - Severity assessment
   - Progression risk
   - Impact on vision and quality of life

4. DIFFERENTIAL DIAGNOSIS
   - Primary diagnoses
   - Alternative considerations
   - Conditions requiring exclusion

5. RECOMMENDED ACTIONS
   - Immediate steps required
   - Follow-up schedule
   - Specialist referrals needed
   - Additional tests recommended
   - Lifestyle modifications

6. PATIENT COUNSELING POINTS
   - Key information to communicate to the patient
   - Warning signs to watch for
   - Questions patient should ask their doctor

Please be thorough, professional, and provide actionable clinical guidance. Use medical terminology but explain key concepts clearly."""

        # Generate content with both images
        response = model_gemini.generate_content([
            prompt,
            "Left eye fundus image:",
            left_image,
            "Right eye fundus image:",
            right_image
        ])
        
        return response.text
    
    except Exception as e:
        st.error(f"Error generating Gemini report: {e}")
        logger.error(f"Gemini API error: {e}")
        return None

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def scanning_effect_enhanced(image_key):
    if image_key in st.session_state:
        st.markdown(f"""
        <div class="scanner-container">
            <div class="scan-line"></div>
            <img src="data:image/png;base64,{image_to_base64(st.session_state[image_key])}" 
                 width="100%" 
                 style="border-radius: 18px; display: block; opacity: 0.9;">
        </div>
        """, unsafe_allow_html=True)

import streamlit.components.v1 as components

def results_card_enhanced(eye_side, predictions, use_optimal_thresholds=True):
    """Display predictions for one eye with optimized thresholds"""
    results, detected = format_predictions(predictions, use_optimal_thresholds)
    
    sorted_preds = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    top_finding = sorted_preds[0]
    is_normal = top_finding['disease'] == 'Normal'
    status_color = "#10b981" if is_normal else ("#f59e0b" if top_finding['probability'] < 0.7 else "#ef4444")
    status_icon = "✅" if is_normal else "⚠" if top_finding['probability'] < 0.7 else "🚨"
    
    html_content = f"""
    <style>
        body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: transparent; color: #e2e8f0; }}
        .glass-card {{
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
            border-top: 4px solid {status_color};
        }}
        .tech-bar-bg {{
            background: rgba(0,0,0,0.3);
            height: 10px;
            border-radius: 10px;
            overflow: hidden;
            margin: 8px 0;
        }}
        .tech-bar-fill {{
            height: 100%;
            box-shadow: 0 0 10px var(--primary);
            transition: width 1s ease-out;
        }}
    </style>
    <div class="glass-card">
        <h3 style="margin-top: 0; display: flex; justify-content: space-between; align-items: center;">
            <span>{eye_side}</span>
            <span style="font-size: 2rem;">{status_icon}</span>
        </h3>
        <div style="font-size: 1.2rem; color: {status_color}; font-weight: 600; margin-bottom: 20px;">
            {top_finding['disease']} ({top_finding['probability']:.1%})
        </div>
    """
    
    for pred in sorted_preds[:4]:
        prob = pred['probability'] * 100
        color = "#10b981" if pred['disease'] == 'Normal' else ("#f59e0b" if prob < 50 else "#ef4444")
        html_content += f"""
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem;">
                <span>{pred['disease']}</span>
                <span style="color: {color}; font-weight: bold;">{prob:.1f}%</span>
            </div>
            <div class="tech-bar-bg">
                <div class="tech-bar-fill" style="width: {prob}%; background: linear-gradient(90deg, {color}, {color}dd);"></div>
            </div>
        </div>
        """
    html_content += "</div>"

    components.html(html_content, height=350, scrolling=False)
    return results, detected

# ================= PROGRESS STEPPER =================
def render_progress_stepper(current_step):
    step1_class = "completed" if current_step > 1 else ("active" if current_step == 1 else "inactive")
    step2_class = "completed" if current_step > 2 else ("active" if current_step == 2 else "inactive")
    step3_class = "active" if current_step == 3 else "inactive"
    
    connector1_class = "completed" if current_step > 1 else ""
    connector2_class = "completed" if current_step > 2 else ""
    
    st.markdown(f"""
    <div class="progress-stepper">
        <div class="step">
            <div class="step-circle {step1_class}">1</div>
            <div class="step-label">Data & Upload</div>
        </div>
        <div class="step-connector {connector1_class}"></div>
        <div class="step">
            <div class="step-circle {step2_class}">2</div>
            <div class="step-label">Diagnostics</div>
        </div>
        <div class="step-connector {connector2_class}"></div>
        <div class="step">
            <div class="step-circle {step3_class}">3</div>
            <div class="step-label">Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= MAIN APP =================

with st.sidebar:
    st.markdown("## 🎮 CONTROL DECK")
    
    # API Configuration Status
    if st.session_state.get('gemini_api_key'):
        st.success("✓ AI Report Generation: Active")
        st.caption("🤖 Powered by Google Gemini")
    else:
        st.error("❌ AI Report Generation: Unavailable")
        st.caption("⚙ Configure API key in .env file")
    
    # Model Status
    if model is not None:
        st.success("✓ CNN Model: Loaded")
        st.caption("🧠 DenseNet121 Neural Network")
    else:
        st.error("❌ CNN Model: Not Loaded")
    
    st.markdown("---")
    
    st.markdown("### Workflow Progress")
    st.markdown(f"Current Step: {st.session_state.workflow_step}/3")
    
    # Display loading metrics if available
    if 'loading_metrics' in st.session_state and st.session_state.loading_metrics.get('total_time'):
        st.markdown("---")
        st.markdown("### 📊 Performance Metrics")
        metrics = st.session_state.loading_metrics
        
        load_time = metrics.get('total_time', 0)
        load_color = "🟢" if load_time < 2 else "🟡" if load_time < 4 else "🔴"
        
        st.markdown(f"**Load Time:** {load_color} {load_time:.2f}s")
        st.markdown(f"**Method:** 🎨 CSS Optimized")
        st.markdown(f"**Performance:** ⚡ 60 FPS")
    
    st.markdown("---")

# ================= HEADER =================
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 class="hero-title">OCULUS PRIME</h1>
    <div class="hero-subtitle">NEURAL RETINAL INTERFACE</div>
</div>
""", unsafe_allow_html=True)

# ================= STEPPER =================
render_progress_stepper(st.session_state.workflow_step)

# ================= STEP 1: PATIENT DATA & IMAGE UPLOAD =================
if st.session_state.workflow_step == 1:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    
    # --- PATIENT DATA SECTION ---
    with st.container():
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top: 0; margin-bottom: 20px;">👤 PATIENT DATA</h3>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            p_name = st.text_input("NAME", placeholder="Patient identifier...")
        with col_b:
            p_age = st.number_input("AGE", 1, 120, 45)
        
        col_c, col_d = st.columns([1, 2])
        with col_c:
            p_gen = st.selectbox("SEX", ["M", "F", "X"])
        with col_d:
            p_hist = st.text_area("HISTORY", height=80, placeholder="Medical history...")
        
        st.session_state.patient = {'name': p_name, 'age': p_age, 'gender': p_gen, 'history': p_hist}
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- THRESHOLD SETTINGS (HIDDEN - ALWAYS USE OPTIMAL) ---
    # Automatically set to use optimal thresholds in background
    st.session_state.use_optimal_thresholds = True
    
    st.markdown("<br>", unsafe_allow_html=True)
      
    # --- IMAGE UPLOAD SECTION ---
    st.markdown("## 📸 RETINAL IMAGE CAPTURE")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card"><h3>👁 LEFT EYE (OS)</h3>', unsafe_allow_html=True)
        l_file = st.file_uploader("📁 Upload Left Eye", type=['png', 'jpg', 'jpeg'], key='l_up')
        if l_file:
            st.session_state.l_img = Image.open(l_file)
        
        if 'l_img' in st.session_state:
            st.image(st.session_state.l_img, use_column_width=True)
            st.success("✅ Left image captured!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card"><h3>👁 RIGHT EYE (OD)</h3>', unsafe_allow_html=True)
        r_file = st.file_uploader("📁 Upload Right Eye", type=['png', 'jpg', 'jpeg'], key='r_up')
        if r_file:
            st.session_state.r_img = Image.open(r_file)
            
        if 'r_img' in st.session_state:
            st.image(st.session_state.r_img, use_column_width=True)
            st.success("✅ Right image captured!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚀 INITIATE NEURAL ANALYSIS", use_container_width=True, type="primary"):
        if 'l_img' in st.session_state and 'r_img' in st.session_state:
            if model is None:
                st.error("❌ Model not loaded. Cannot perform analysis.")
            else:
                # Store images temporarily to avoid session state issues
                left_image = st.session_state.l_img
                right_image = st.session_state.r_img
                
                # ===== PHASE 1: VALIDATE IMAGES (SIMPLIFIED) =====
                validation_needed = st.session_state.get('gemini_api_key') is not None
                
                if validation_needed:
                    st.info("🔍 **Phase 1/2:** Validating fundus images with AI...")
                    
                    col1, col2 = st.columns(2)
                    
                    # Validate left eye
                    with col1:
                        with st.spinner("Validating left eye..."):
                            l_valid, l_conf, l_msg = validate_fundus_image(left_image)
                        
                        if l_valid:
                            st.success("✅ Left Eye: Valid")
                        else:
                            st.error("❌ Left Eye: Invalid")
                            st.warning(l_msg)
                    
                    # Validate right eye
                    with col2:
                        with st.spinner("Validating right eye..."):
                            r_valid, r_conf, r_msg = validate_fundus_image(right_image)
                        
                        if r_valid:
                            st.success("✅ Right Eye: Valid")
                        else:
                            st.error("❌ Right Eye: Invalid")
                            st.warning(r_msg)
                    
                    # Block if invalid
                    if not l_valid or not r_valid:
                        st.error("🚫 **Analysis Blocked:** Invalid images detected.")
                        st.info("💡 Please upload proper fundus photographs.")
                        st.stop()
                    
                    st.success("✅ Both images validated!")
                else:
                    st.warning("⚠ Validation skipped (No Gemini API key)")
                
                # ===== PHASE 2: CNN ANALYSIS (SIMPLIFIED) =====
                st.info("🧠 **Running CNN analysis...**")
                
                # Simple progress without complex animations
                with st.spinner("Processing images..."):
                    time.sleep(1)  # Brief pause for UX
                    
                    # Run predictions
                    l_pred, l_gradcams = predict_diseases_with_gradcam(st.session_state.l_img)
                    r_pred, r_gradcams = predict_diseases_with_gradcam(st.session_state.r_img)

                    st.session_state.l_pred = l_pred
                    st.session_state.r_pred = r_pred
                    st.session_state.l_gradcams = l_gradcams if l_gradcams else {}
                    st.session_state.r_gradcams = r_gradcams if r_gradcams else {}
                    
                    # Store results
                    st.session_state.results_ready = True
                    
                    # Store validation info if done
                    if validation_needed:
                        st.session_state.l_validation = {
                            'valid': l_valid, 
                            'confidence': l_conf, 
                            'message': l_msg
                        }
                        st.session_state.r_validation = {
                            'valid': r_valid, 
                            'confidence': r_conf, 
                            'message': r_msg
                        }
                
                st.success("🎉 Analysis complete!")
                
                # CRITICAL: Minimal delay before rerun
                time.sleep(0.5)
                
                # Update workflow step
                st.session_state.workflow_step = 2
                
                # Rerun immediately
                st.rerun()
        else:
            st.error("❌ Please upload both images")

    
    st.markdown('</div>', unsafe_allow_html=True)

# ================= STEP 2: DIAGNOSTICS =================
elif st.session_state.workflow_step == 2:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    st.markdown("## 🔬 DIAGNOSTIC RESULTS")
    
    if st.session_state.get('results_ready'):
        st.success("🎉 Neural analysis complete! Review the diagnostic findings below.")
        
        use_optimal = st.session_state.get('use_optimal_thresholds', True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.l_img, caption="Left Eye", use_column_width=True)
            left_results, left_detected = results_card_enhanced("LEFT EYE (OS)", st.session_state.l_pred, use_optimal)
            st.session_state.l_res = left_results
            st.session_state.l_detected = left_detected
        
        with col2:
            st.image(st.session_state.r_img, caption="Right Eye", use_column_width=True)
            right_results, right_detected = results_card_enhanced("RIGHT EYE (OD)", st.session_state.r_pred, use_optimal)
            st.session_state.r_res = right_results
            st.session_state.r_detected = right_detected
        
        if 'l_validation' in st.session_state and 'r_validation' in st.session_state:
            st.markdown("---")
            st.markdown("### 🔍 Image Validation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                l_val = st.session_state.l_validation
                if l_val['confidence'] == 'skipped':
                    st.info("⚠ Left Eye: Validation skipped (No API key)")
                elif l_val['valid']:
                    st.success(f"✅ Left Eye: Validated ({l_val['confidence']} confidence)")
                else:
                    st.warning(f"⚠ Left Eye: Validation concerns ({l_val['confidence']} confidence)")
                
                with st.expander("View Details"):
                    st.markdown(l_val['message'])
            
            with col2:
                r_val = st.session_state.r_validation
                if r_val['confidence'] == 'skipped':
                    st.info("⚠ Right Eye: Validation skipped (No API key)")
                elif r_val['valid']:
                    st.success(f"✅ Right Eye: Validated ({r_val['confidence']} confidence)")
                else:
                    st.warning(f"⚠ Right Eye: Validation concerns ({r_val['confidence']} confidence)")
                
                with st.expander("View Details"):
                    st.markdown(r_val['message'])
            
            st.markdown("---")

        # Comparison summary
        st.markdown("---")
        st.markdown("### 🔄 Bilateral Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Left Eye Conditions", len(left_detected))
        
        with col2:
            st.metric("Right Eye Conditions", len(right_detected))
        
        with col3:
            bilateral = set(left_detected) & set(right_detected)
            st.metric("Bilateral Conditions", len(bilateral))
        
        if bilateral:
            st.warning(f"⚠ **Bilateral findings detected:** {', '.join(bilateral)}")
        
        # Grad-CAM visualizations
        if st.session_state.get('l_gradcams') or st.session_state.get('r_gradcams'):
            st.markdown("---")
            st.markdown("### 🔥 Grad-CAM Heatmap Analysis")
            st.info("These visualizations show which regions of the retina influenced the AI's predictions.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.get('l_gradcams'):
                    st.markdown("**Left Eye Heatmaps:**")
                    for disease, data in st.session_state.l_gradcams.items():
                        with st.expander(f"🔍 {disease} ({data['confidence']:.1%})"):
                            st.image(data['image'], use_column_width=True)
                else:
                    st.info("No heatmaps generated for left eye")
            
            with col2:
                if st.session_state.get('r_gradcams'):
                    st.markdown("**Right Eye Heatmaps:**")
                    for disease, data in st.session_state.r_gradcams.items():
                        with st.expander(f"🔍 {disease} ({data['confidence']:.1%})"):
                            st.image(data['image'], use_column_width=True)
                else:
                    st.info("No heatmaps generated for right eye")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅ Back to Upload", use_container_width=True):
                st.session_state.workflow_step = 1
                st.rerun()
        with col2:
            if st.button("Generate Full Report ➡", use_container_width=True, type="primary"):
                st.session_state.workflow_step = 3
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================= STEP 3: REPORT GENERATION =================
elif st.session_state.workflow_step == 3:
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    st.markdown("## 📋 COMPREHENSIVE MEDICAL REPORT")
    
    if st.session_state.get('gemini_api_key'):
        with st.spinner("🤖 Generating AI-powered medical report..."):
            if 'full_report' not in st.session_state:
                report = generate_llm_report(
                    st.session_state.l_res,
                    st.session_state.r_res,
                    st.session_state.l_img,
                    st.session_state.r_img,
                    st.session_state.patient
                )
                st.session_state.full_report = report
            else:
                report = st.session_state.full_report
        
        if report:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(report)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"oculus_report_{st.session_state.patient['name']}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.error("Failed to generate report. Please try again.")
    else:
        st.error("❌ Gemini API key not configured. Cannot generate AI report.")
        st.info("💡 Configure your GEMINI_API_KEY in the .env file to enable this feature.")
    
    # Navigation
    if st.button("⬅ Back to Results", use_container_width=True):
        st.session_state.workflow_step = 2
        st.rerun()
    
    if st.button("🔄 Start New Analysis", use_container_width=True, type="primary"):
        # Clear session state
        for key in ['l_img', 'r_img', 'l_pred', 'r_pred', 'l_gradcams', 'r_gradcams', 
                    'results_ready', 'l_res', 'r_res', 'l_detected', 'r_detected', 
                    'l_validation', 'r_validation', 'full_report']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.workflow_step = 1
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)