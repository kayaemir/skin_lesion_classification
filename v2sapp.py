"""
EfficientNetV2-S Hybrid Model — Skin Lesion Diagnostic Assistant (v2sapp)
===================================================================
Streamlit interface for the EfficientNetV2-S based hybrid skin lesion
classification model. Shows model load time prominently.

Usage:
    streamlit run v2sapp.py
"""

import os
import time

os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
from PIL import Image
import gdown

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Lesion Diagnostic Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background-color: #1a1a2e;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .main-header h1 { margin: 0; font-weight: 700; font-size: 2rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.8; font-size: 0.95rem; }
    
    .metric-card {
        background-color: #0f3460;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(15,52,96,0.3);
    }
    .metric-card .label { font-size: 0.8rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }
    
    .alert-box {
        background-color: #d32f2f;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(211,47,47,0.35);
    }
    .safe-box {
        background-color: #2e7d32;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46,125,50,0.35);
    }
    
    div[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
IMG_SIZE = 384
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)

DISEASE_INFO = {
    "akiec": ("Actinic keratosis / SCC", "Premalignant lesions on sun-damaged skin that carry a risk of transforming into squamous cell carcinoma (SCC). Early treatment is important.", True),
    "bcc":   ("Basal cell carcinoma", "The most common form of skin cancer. Grows slowly and rarely metastasizes, but can cause local destruction. Surgical excision is the gold standard.", True),
    "bkl":   ("Benign keratosis", "Encompasses benign keratinocytic lesions such as seborrheic keratosis, solar lentigo, or lichenoid keratosis. Treatment is usually for cosmetic purposes.", False),
    "df":    ("Dermatofibroma", "A small, firm, benign skin tumor often found on the lower extremities. Treatment is generally not required.", False),
    "mel":   ("Melanoma", "⚠️ The most dangerous form of skin cancer. Treatable if caught early. Watch for asymmetry, border irregularity, color variations, and increasing diameter.", True),
    "nv":    ("Melanocytic nevus (Mole)", "Common, generally benign proliferations of melanocytes. Moles that change over time should be carefully monitored.", False),
    "vasc":  ("Vascular lesion", "Encompasses vascular lesions like angiomas, angiokeratomas, or pyogenic granulomas. Most are benign.", False),
}

MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}
CLINICAL_THRESHOLD = 0.30

LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
    "unknown", "upper extremity",
]

LOCALIZATION_EN = {
    "abdomen": "Abdomen", "acral": "Acral", "back": "Back", "chest": "Chest",
    "ear": "Ear", "face": "Face", "foot": "Foot", "genital": "Genital",
    "hand": "Hand", "lower extremity": "Lower Extremity", "neck": "Neck",
    "scalp": "Scalp", "trunk": "Trunk", "unknown": "Unknown",
    "upper extremity": "Upper Extremity",
}

TRAIN_AGE_MAX = 85.0
TRAIN_AGE_MEDIAN = 50.0


# ── Model loading ───────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    """Load the .keras model and return (model, load_time_seconds)."""
    import tensorflow as tf
    from keras import layers
    import keras

    # Custom layer used during training augmentation
    class RandomHueSat(layers.Layer):
        def __init__(self, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(**kwargs)

        def call(self, x, training=None):
            if training is True:
                x = x / 255.0
                x = tf.image.random_hue(x, 0.08)
                x = tf.image.random_saturation(x, 0.7, 1.3)
                x = tf.clip_by_value(x, 0.0, 1.0)
                x = x * 255.0
            return x

    # Focal loss used for compilation
    def categorical_focal_loss(alpha, gamma):
        alpha_t = tf.constant(alpha, dtype=tf.float32)

        def focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha_t * tf.math.pow(tf.abs(y_true - y_pred), gamma)
            loss = weight * cross_entropy
            return tf.reduce_sum(loss, axis=-1)

        return focal_loss

    t0 = time.time()
    model = keras.saving.load_model(
        model_path,
        custom_objects={
            "RandomHueSat": RandomHueSat,
            "focal_loss": categorical_focal_loss([1.0] * NUM_CLASSES, 1.0),
        },
        safe_mode=False,
    )
    load_time = time.time() - t0
    return model, load_time


# ── Preprocessing helpers ────────────────────────────────────────
def preprocess_image(uploaded_file, target_size: int):
    """Return a (1, H, W, 3) float32 numpy array."""
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def encode_metadata(age: float, sex: str, localization: str):
    """Return a (1, 18) float32 numpy array matching training encoding."""
    age_norm = np.clip(age / TRAIN_AGE_MAX, 0.0, 1.0)
    sex_male = 1.0 if sex == "male" else 0.0
    sex_unknown = 1.0 if sex == "unknown" else 0.0

    loc_vec = [1.0 if localization == loc else 0.0 for loc in LOCALIZATIONS]

    meta = np.array(
        [[age_norm, sex_male, sex_unknown] + loc_vec], dtype=np.float32
    )
    return meta  # shape (1, 18)


def apply_clinical_thresholds(probs: np.ndarray):
    """Apply 30 % clinical alert threshold for malignant classes."""
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    alert = False
    alert_classes = []
    for cls_name in MALIGNANT_CLASSES:
        idx = CLASS_NAMES.index(cls_name)
        if probs[idx] >= CLINICAL_THRESHOLD:
            alert = True
            alert_classes.append((cls_name, float(probs[idx])))

    if alert:
        # Force prediction to highest-probability malignant class
        alert_classes.sort(key=lambda x: x[1], reverse=True)
        forced_class = alert_classes[0][0]
        forced_conf = alert_classes[0][1]
        if forced_class != pred_class:
            pred_class = forced_class
            confidence = forced_conf

    return pred_class, confidence, alert, alert_classes


# ── Settings ─────────────────────────────────────────────────────
model_path = "best_model_finetune.keras"

# ── Header ───────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔬 Skin Lesion Diagnostic Assistant</h1>
        <p>EfficientNetV2-S + Clinical Metadata Hybrid Model &nbsp;|&nbsp; 7 Classes &nbsp;|&nbsp; Clinical Threshold System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "**⚠️ IMPORTANT WARNING:** This application is NOT for medical or clinical use. "
    "It is for academic research and prototyping purposes only. "
    "It must not be used for any diagnostic or medical decisions."
)

# ── Inputs ───────────────────────────────────────────────────────
st.markdown("### 📋 1. Patient Information")
col_age, col_sex, col_loc = st.columns(3)
with col_age:
    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1, key="age_input")
with col_sex:
    sex = st.selectbox(
        "Sex",
        options=["female", "male", "unknown"],
        format_func=lambda x: {"female": "Female", "male": "Male", "unknown": "Unknown"}[x],
        key="sex_select",
    )
with col_loc:
    localization = st.selectbox(
        "Lesion Localization",
        options=LOCALIZATIONS,
        format_func=lambda x: LOCALIZATION_EN.get(x, x),
        key="loc_select",
    )

st.markdown("### 📸 2. Image Source")
img_file = st.file_uploader(
    "📂 Select an image or take a photo (mobile)",
    type=["jpg", "jpeg", "png"],
    key="img_uploader",
)


# ── Load model ───────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.warning(f"⚠️ `{model_path}` not found. Downloading via Google Drive, please wait...")
    
    # Create target directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    file_id = "BURAYA_FILE_ID"
    url = f"https://drive.google.com/file/d/1J91oqLsYLkP_eDzPWLQyqXNoXAqrhMdB/view?usp=sharing"
    
    try:
        gdown.download(url, model_path, quiet=False)
        if not os.path.exists(model_path):
            st.error("❌ Download failed. Please check the file ID and your internet connection.")
            st.stop()
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Error during download: {e}")
        st.stop()

with st.spinner("🧠 Loading model… (might take a while on first launch)"):
    model, load_time = load_model(model_path)

# Detect actual image size from model input
actual_img_size = model.inputs[0].shape[1] or IMG_SIZE
meta_dim = model.inputs[1].shape[1] or 18

# Model info is now displayed in the top-right menu (About section).

# ── Footer Helper ────────────────────────────────────────────────
def render_footer():
    st.markdown("---")
    
    with st.expander("ℹ️ About the Model"):
        st.markdown(f"""
        - **Architecture:** EfficientNetV2-S + Clinical Metadata
        - **Image Input Size:** {IMG_SIZE}×{IMG_SIZE} pixels
        - **Metadata Features:** 18 vector size
        - **Supported Classes ({NUM_CLASSES}):** Actinic keratosis / SCC, Basal cell carcinoma, Benign keratosis, Dermatofibroma, Melanoma, Melanocytic nevus, Vascular lesion
        - **Model Load Time:** {load_time:.2f} seconds
        """)

# ── Main content ─────────────────────────────────────────────────
if img_file is None:
    st.info("👆 Please provide an image (upload or take a photo) to get a diagnosis.")
    render_footer()
    st.stop()

# Show uploaded image
st.markdown("---")
col_img, col_result = st.columns([1, 1.5])

with col_img:
    st.markdown("### 🖼️ Input Image")
    st.image(img_file, use_container_width=True)

    st.markdown("**Patient Information**")
    st.markdown(f"- **Age:** {age}")
    sex_tr = {'female': 'Female', 'male': 'Male', 'unknown': 'Unknown'}.get(sex, sex)
    st.markdown(f"- **Sex:** {sex_tr}")
    st.markdown(f"- **Localization:** {LOCALIZATION_EN.get(localization, localization)}")

# Run prediction
img_arr = preprocess_image(img_file, actual_img_size)
meta_arr = encode_metadata(float(age), sex, localization)

# Pad/slice metadata if model expects different dim
if meta_arr.shape[1] < meta_dim:
    pad = np.zeros((1, meta_dim - meta_arr.shape[1]), dtype=np.float32)
    meta_arr = np.concatenate([meta_arr, pad], axis=1)
elif meta_arr.shape[1] > meta_dim:
    meta_arr = meta_arr[:, :meta_dim]

t_pred = time.time()
preds = model.predict({"image_input": img_arr, "meta_input": meta_arr}, verbose=0)
pred_time = time.time() - t_pred
probs = preds[0]

pred_class, confidence, alert, alert_classes = apply_clinical_thresholds(probs)
disease_name, description, is_malignant = DISEASE_INFO[pred_class]

with col_result:
    st.markdown("### 📊 Diagnostic Results")

    # Prediction time
    st.caption(f"⏱️ Prediction time: **{pred_time:.3f}s**")

    # Clinical alert or safe result
    if alert:
        alert_text = ", ".join(
            f"{DISEASE_INFO[c][0]} ({p:.1%})" for c, p in alert_classes
        )
        st.markdown(
            f'<div class="alert-box">'
            f"⚠️ <strong>CLINICAL ALERT</strong> — Malignant class(es) exceeded the threshold (30%):<br>"
            f"{alert_text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="safe-box">'
            f"✅ <strong>Below malignant threshold</strong> — All malignant class probabilities are under 30%.</div>",
            unsafe_allow_html=True,
        )

    # Predicted class
    st.markdown(f"**Prediction:** `{pred_class}` — **{disease_name}**")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
    st.markdown(f"**Risk:** {'🔴 Malignant' if is_malignant else '🟢 Benign'}")

    st.divider()
    st.markdown(f"**Description:** {description}")

# ── Probability distribution ────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Probability Distribution")

html_bars = ""
# Sort probabilities descending for better visualization
sorted_indices = np.argsort(probs)[::-1]

for i in sorted_indices:
    cls = CLASS_NAMES[i]
    prob = probs[i]
    d_name = DISEASE_INFO[cls][0]
    
    if cls == pred_class:
        color = "#e94560"  # Red for predicted
    elif cls in MALIGNANT_CLASSES:
        color = "#f39c12"  # Orange for malignant
    else:
        color = "#3498db"  # Blue for benign
        
    html_bars += f'''
    <div style="margin-bottom: 12px; font-family: 'Inter', sans-serif;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.95rem;">
            <span>{d_name} <span style="opacity:0.6; font-size:0.8rem;">({cls})</span></span>
            <span style="font-weight: 600;">{prob:.1%}</span>
        </div>
        <div style="width: 100%; background-color: rgba(128, 128, 128, 0.15); border-radius: 6px; height: 22px; position: relative; overflow: hidden;">
            <div style="width: {prob*100}%; background: {color}; height: 100%; border-radius: 6px; transition: width 0.5s ease-in-out;"></div>
            <div style="position: absolute; left: 30%; top: 0; bottom: 0; width: 2px; background-color: rgba(233, 69, 96, 0.8); z-index: 10; box-shadow: 0 0 4px rgba(233,69,96,0.5);" title="30% Clinical Threshold"></div>
        </div>
    </div>
    '''

st.markdown(html_bars, unsafe_allow_html=True)

# ── Legend ───────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; gap:1.5rem; justify-content:center; margin-top:1.5rem; font-size:0.85rem; color:#888;">
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; border-radius:50%; background:#e94560;"></div> Predicted class</span>
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; border-radius:50%; background:#f39c12;"></div> Malignant class</span>
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; border-radius:50%; background:#3498db;"></div> Benign class</span>
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:2px; height:14px; background:rgba(233, 69, 96, 0.8);"></div> 30% clinical threshold</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Footer ───────────────────────────────────────────────────────
render_footer()
