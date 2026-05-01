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
    page_title="DermAI v2s — Cilt Lezyonu Tanı Asistanı",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .main-header h1 { margin: 0; font-weight: 700; font-size: 2rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.8; font-size: 0.95rem; }
    
    .metric-card {
        background: linear-gradient(135deg, #0f3460, #533483);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(83,52,131,0.3);
    }
    .metric-card .label { font-size: 0.8rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }
    
    .alert-box {
        background: linear-gradient(135deg, #d32f2f, #b71c1c);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(211,47,47,0.35);
    }
    .safe-box {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46,125,50,0.35);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
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
    "akiec": ("Aktinik Keratoz / SCC", "Güneşe maruz kalan bölgelerde oluşan, skuamöz hücreli karsinoma (SCC) dönüşme riski taşıyan prekanseröz lezyonlardır. Erken tedavi önemlidir.", True),
    "bcc":   ("Bazal Hücreli Karsinom", "En sık görülen deri kanseri türüdür. Yavaş büyür, nadiren metastaz yapar ancak lokal destrüksiyon yapabilir. Cerrahi eksizyon altın standarttır.", True),
    "bkl":   ("Benign Keratoz", "Seboreik keratoz, solar lentigo veya likenoid keratoz gibi iyi huylu keratinositik lezyonları kapsar. Tedavi genellikle kozmetik amaçlıdır.", False),
    "df":    ("Dermatofibrom", "Sıklıkla alt ekstremitede görülen, sert, küçük, iyi huylu bir deri tümörüdür. Tedavi genellikle gerekmez.", False),
    "mel":   ("Melanom", "⚠️ En tehlikeli deri kanseri türüdür. Erken evrede tespit edilirse tedavi edilebilir. Asimetri, düzensiz sınır, renk değişikliği ve çap artışına dikkat edilmelidir.", True),
    "nv":    ("Melanositik Nevüs (Ben)", "Yaygın görülen, genellikle iyi huylu melanosit proliferasyonlarıdır. Değişim gösteren benler dikkatle takip edilmelidir.", False),
    "vasc":  ("Vasküler Lezyon", "Anjiyom, anjiyokeratom veya piyojenik granülom gibi damar kaynaklı lezyonları kapsar. Çoğu iyi huyludur.", False),
}

MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}
CLINICAL_THRESHOLD = 0.30

LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
    "unknown", "upper extremity",
]

LOCALIZATION_TR = {
    "abdomen": "Karın", "acral": "Akral", "back": "Sırt", "chest": "Göğüs",
    "ear": "Kulak", "face": "Yüz", "foot": "Ayak", "genital": "Genital",
    "hand": "El", "lower extremity": "Alt Ekstremite", "neck": "Boyun",
    "scalp": "Kafa Derisi", "trunk": "Gövde", "unknown": "Bilinmiyor",
    "upper extremity": "Üst Ekstremite",
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


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 DermAI v2s")
    st.markdown("**EfficientNetV2-S Hybrid Model**")
    st.divider()

    st.markdown("### 📋 Hasta Bilgileri")
    age = st.slider("Yaş", 0, 100, 50, key="age_slider")
    sex = st.selectbox(
        "Cinsiyet",
        options=["female", "male", "unknown"],
        format_func=lambda x: {"female": "Kadın", "male": "Erkek", "unknown": "Bilinmiyor"}[x],
        key="sex_select",
    )
    localization = st.selectbox(
        "Lezyon Bölgesi",
        options=LOCALIZATIONS,
        format_func=lambda x: LOCALIZATION_TR.get(x, x),
        key="loc_select",
    )

    st.divider()
    st.markdown("### 📸 Görüntü Yükle")
    uploaded_file = st.file_uploader(
        "Dermoskopik görüntü seçin",
        type=["jpg", "jpeg", "png"],
        key="img_uploader",
    )

    st.divider()
    model_path = st.text_input(
        "Model Dosya Yolu",
        value="best_model_finetune-2.keras",
        key="model_path_input",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color: rgba(255, 193, 7, 0.15); border-left: 4px solid #ffc107; padding: 10px; border-radius: 4px; font-size: 0.85rem; color: #ffc107;">
            <strong>⚠️ ÖNEMLİ UYARI:</strong> Bu uygulama medikal veya klinik kullanım için DEĞİLDİR. Sadece akademik araştırma ve prototip amaçlıdır. Hiçbir teşhis veya tıbbi karar için kullanılamaz.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Header ───────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔬 DermAI v2s — Cilt Lezyonu Tanı Asistanı</h1>
        <p>EfficientNetV2-S + Klinik Metadata Hibrit Modeli &nbsp;|&nbsp; 7 Sınıf &nbsp;|&nbsp; Klinik Eşik Sistemi</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load model ───────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.warning(f"⚠️ `{model_path}` bulunamadı. Google Drive üzerinden indiriliyor, lütfen bekleyin...")
    
    # Hedef klasör yoksa oluştur (örn. model/ yoluna indirmek istenirse)
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    # Google Drive dosya ID'sini buraya yaz
    file_id = "BURAYA_FILE_ID"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        gdown.download(url, model_path, quiet=False)
        if not os.path.exists(model_path):
            st.error("❌ İndirme başarısız oldu. Lütfen dosya ID'sini ve internet bağlantınızı kontrol edin.")
            st.stop()
        st.success("✅ Model başarıyla indirildi!")
    except Exception as e:
        st.error(f"❌ İndirme sırasında hata oluştu: {e}")
        st.stop()

with st.spinner("🧠 Model yükleniyor… (ilk açılışta biraz zaman alabilir)"):
    model, load_time = load_model(model_path)

# Detect actual image size from model input
actual_img_size = model.inputs[0].shape[1] or IMG_SIZE
meta_dim = model.inputs[1].shape[1] or 18

# ── Model info cards ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card"><div class="label">Model Yükleme Süresi</div>'
        f'<div class="value">{load_time:.2f}s</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card"><div class="label">Görüntü Boyutu</div>'
        f'<div class="value">{actual_img_size}×{actual_img_size}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card"><div class="label">Metadata Boyutu</div>'
        f'<div class="value">{meta_dim}</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="metric-card"><div class="label">Sınıf Sayısı</div>'
        f'<div class="value">{NUM_CLASSES}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Main content ─────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Sol panelden bir dermoskopik görüntü yükleyin ve hasta bilgilerini girin.")
    st.stop()

# Show uploaded image
col_img, col_result = st.columns([1, 1.5])

with col_img:
    st.markdown("### 🖼️ Yüklenen Görüntü")
    st.image(uploaded_file, use_container_width=True)

    st.markdown("**Hasta Bilgileri**")
    st.markdown(f"- **Yaş:** {age}")
    sex_tr = {'female': 'Kadın', 'male': 'Erkek', 'unknown': 'Bilinmiyor'}.get(sex, sex)
    st.markdown(f"- **Cinsiyet:** {sex_tr}")
    st.markdown(f"- **Bölge:** {LOCALIZATION_TR.get(localization, localization)}")

# Run prediction
img_arr = preprocess_image(uploaded_file, actual_img_size)
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
    st.markdown("### 📊 Tanı Sonuçları")

    # Prediction time
    st.caption(f"⏱️ Tahmin süresi: **{pred_time:.3f}s**")

    # Clinical alert or safe result
    if alert:
        alert_text = ", ".join(
            f"{DISEASE_INFO[c][0]} ({p:.1%})" for c, p in alert_classes
        )
        st.markdown(
            f'<div class="alert-box">'
            f"⚠️ <strong>KLİNİK UYARI</strong> — Malign sınıf(lar) eşik değerini aştı (%30):<br>"
            f"{alert_text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="safe-box">'
            f"✅ <strong>Malign eşik aşılmadı</strong> — Tüm malign sınıf olasılıkları %30 altında.</div>",
            unsafe_allow_html=True,
        )

    # Predicted class
    st.markdown(f"**Tahmin:** `{pred_class}` — **{disease_name}**")
    st.markdown(f"**Güven:** `{confidence:.2%}`")
    st.markdown(f"**Risk:** {'🔴 Malign' if is_malignant else '🟢 Benign'}")

    st.divider()
    st.markdown(f"**Açıklama:** {description}")

# ── Probability distribution ────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Olasılık Dağılımı")

html_bars = ""
# Sort probabilities descending for better visualization
sorted_indices = np.argsort(probs)[::-1]

for i in sorted_indices:
    cls = CLASS_NAMES[i]
    prob = probs[i]
    d_name = DISEASE_INFO[cls][0]
    
    if cls == pred_class:
        color = "linear-gradient(90deg, #e94560, #c0392b)"  # Red for predicted
    elif cls in MALIGNANT_CLASSES:
        color = "linear-gradient(90deg, #f39c12, #d35400)"  # Orange for malignant
    else:
        color = "linear-gradient(90deg, #3498db, #2980b9)"  # Blue for benign
        
    html_bars += f'''
    <div style="margin-bottom: 12px; font-family: 'Inter', sans-serif;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.95rem;">
            <span>{d_name} <span style="opacity:0.6; font-size:0.8rem;">({cls})</span></span>
            <span style="font-weight: 600;">{prob:.1%}</span>
        </div>
        <div style="width: 100%; background-color: rgba(128, 128, 128, 0.15); border-radius: 6px; height: 22px; position: relative; overflow: hidden;">
            <div style="width: {prob*100}%; background: {color}; height: 100%; border-radius: 6px; transition: width 0.5s ease-in-out;"></div>
            <div style="position: absolute; left: 30%; top: 0; bottom: 0; width: 2px; background-color: rgba(233, 69, 96, 0.8); z-index: 10; box-shadow: 0 0 4px rgba(233,69,96,0.5);" title="%30 Klinik Eşik"></div>
        </div>
    </div>
    '''

st.markdown(html_bars, unsafe_allow_html=True)

# ── Legend ───────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; gap:1.5rem; justify-content:center; margin-top:1.5rem; font-size:0.85rem; color:#888;">
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; border-radius:50%; background:#e94560;"></div> Tahmin edilen sınıf</span>
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; border-radius:50%; background:#f39c12;"></div> Malign sınıf</span>
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; border-radius:50%; background:#3498db;"></div> Benign sınıf</span>
        <span style="display:flex; align-items:center; gap:5px;"><div style="width:2px; height:14px; background:rgba(233, 69, 96, 0.8);"></div> %30 klinik eşik</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚕️ Bu araç yalnızca araştırma amaçlıdır ve tıbbi tanı yerine geçmez. "
    "Kesin tanı için bir dermatolog veya onkologa başvurunuz."
)
