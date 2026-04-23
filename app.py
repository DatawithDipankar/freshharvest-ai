import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreshHarvest AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,400;1,600&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:             #080f09;
    --surface:        #0d1810;
    --surface2:       #152019;
    --surface3:       #1c2b20;
    --border:         #243429;
    --border-light:   #2f4535;
    --fresh:          #5efc8d;
    --fresh-mid:      #22c55e;
    --fresh-dim:      #14532d;
    --fresh-glow:     rgba(94,252,141,0.07);
    --fresh-glow-md:  rgba(94,252,141,0.13);
    --spoiled:        #ff6b6b;
    --spoiled-mid:    #ef4444;
    --spoiled-dim:    #7f1d1d;
    --spoiled-glow:   rgba(255,107,107,0.07);
    --spoiled-glow-md:rgba(255,107,107,0.13);
    --warn:           #f59e0b;
    --warn-dim:       #78350f;
    --warn-glow:      rgba(245,158,11,0.08);
    --text:           #eef5ee;
    --text-muted:     #6b9470;
    --text-dim:       #3d5c42;
    --accent:         #5efc8d;
    --gold:           #d4a853;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 3rem !important;
    max-width: 100% !important;
}

/* ── Content centering wrapper ── */
.main-wrap > div,
.stColumn > div { width: 100%; }

/* ── Top bar ── */
.topbar {
    background: linear-gradient(180deg, var(--surface) 0%, rgba(13,24,16,0.95) 100%);
    border-bottom: 1px solid var(--border);
    padding: 0.85rem 3rem;    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.topbar-logo .logo-dot {
    width: 7px; height: 7px;
    background: var(--fresh);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--fresh);
    display: inline-block;
    margin-right: 0.2rem;
}
.topbar-logo span { color: var(--fresh); font-style: italic; font-weight: 600; }
.topbar-pills { display: flex; gap: 0.5rem; align-items: center; }
.pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.3rem 0.9rem;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: border-color 0.2s;
}
.pill:hover { border-color: var(--border-light); }
.pill-green {
    border-color: var(--fresh-dim);
    color: var(--fresh);
    background: var(--fresh-glow);
    box-shadow: 0 0 12px rgba(94,252,141,0.06);
}

/* ── Main wrap ── */
.main-wrap {
    max-width: 100%;
    margin: 0 auto;
    padding: 3rem 0 5rem;
    box-sizing: border-box;
}
[data-testid="stImage"] img {
    max-height: 320px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
    border-radius: 10px;
    display: block !important;
    margin: 0 auto !important;
}
[data-testid="stImage"],
[data-testid="stImage"] > div,
[data-testid="stImage"] > div > div {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}


/* ── Hero ── */
.hero { text-align: center; padding: 2rem 0 3rem; position: relative; }
.hero::before {
    content: '';
    position: absolute;
    top: -20px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(94,252,141,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--fresh-mid);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
}
.hero-eyebrow::before,
.hero-eyebrow::after {
    content: '';
    display: block;
    width: 32px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--fresh-dim));
}
.hero-eyebrow::after { background: linear-gradient(90deg, var(--fresh-dim), transparent); }
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 700;
    color: var(--text);
    line-height: 1.05;
    margin: 0 0 0.9rem;
    letter-spacing: -0.025em;
}
.hero-title em { font-style: italic; color: var(--fresh); font-weight: 600; }
.hero-sub {
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 300;
    max-width: 420px;
    margin: 1.2rem auto 0;
    line-height: 1.8;
    text-align: center;
    display: block;
    width: 100%;
}

.hero p, .hero-sub { text-align: center !important; margin-left: auto !important; margin-right: auto !important; }

/* ── Upload zone ── */
.upload-section {
    background: var(--surface);
    border: 1.5px dashed var(--border-light);
    border-radius: 18px;
    padding: 0.3rem 1.5rem 0.3rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.upload-section::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(94,252,141,0.03) 0%, transparent 60%);
    pointer-events: none;
}
.upload-section:hover {
    border-color: var(--fresh-mid);
    background: var(--surface2);
    box-shadow: 0 0 30px rgba(94,252,141,0.05);
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}
[data-testid="stFileUploaderDropzone"] > div > div { color: var(--text-muted) !important; }
[data-testid="stFileUploaderDropzone"] svg { stroke: var(--text-dim) !important; }

.upload-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-dim);
    margin-bottom: 0.2rem;
    padding-top: 0.2rem;
}

/* ── Result card wrapper ── */
.result-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s cubic-bezier(0.16,1,0.3,1);
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-wrap.is-fresh   { border-color: var(--fresh-dim); box-shadow: 0 0 30px rgba(94,252,141,0.04); }
.result-wrap.is-spoiled { border-color: var(--spoiled-dim); box-shadow: 0 0 30px rgba(255,107,107,0.04); }
.result-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.result-wrap.is-fresh::before   { background: linear-gradient(90deg, transparent, var(--fresh-mid), transparent); }
.result-wrap.is-spoiled::before { background: linear-gradient(90deg, transparent, var(--spoiled-mid), transparent); }

/* ── Verdict ── */
.verdict {
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    margin-bottom: 1.3rem;
    animation: popIn 0.4s cubic-bezier(0.34,1.56,0.64,1);
    position: relative;
}
@keyframes popIn {
    from { opacity:0; transform:scale(0.91) translateY(8px); }
    to   { opacity:1; transform:scale(1)    translateY(0);   }
}
.verdict-fresh {
    background: var(--fresh-glow-md);
    border: 1px solid var(--fresh-dim);
    box-shadow: inset 0 1px 0 rgba(94,252,141,0.08);
}
.verdict-spoiled {
    background: var(--spoiled-glow-md);
    border: 1px solid var(--spoiled-dim);
    box-shadow: inset 0 1px 0 rgba(255,107,107,0.08);
}
.verdict-emoji {
    font-size: 3rem;
    flex-shrink: 0;
    line-height: 1;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
}
.verdict-badge {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.badge-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    display: inline-block;
}
.badge-fresh  { color: var(--fresh); }
.badge-fresh .badge-dot  { background: var(--fresh); box-shadow: 0 0 5px var(--fresh); }
.badge-spoiled { color: var(--spoiled); }
.badge-spoiled .badge-dot { background: var(--spoiled); box-shadow: 0 0 5px var(--spoiled); }
.verdict-fruit {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
    line-height: 1.05;
    letter-spacing: -0.02em;
}
.verdict-conf { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.25rem; font-weight: 400; }

/* ── Confidence bar ── */
.cbar-wrap   { margin-bottom: 1.4rem; }
.cbar-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; }
.cbar-lbl    { font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:var(--text-dim); }
.cbar-val    { font-size:0.82rem; font-weight:700; color:var(--text); }
.cbar-track  {
    background: var(--surface2);
    border-radius:99px;
    height: 7px;
    overflow:hidden;
    border: 1px solid var(--border);
    position: relative;
}
.fill-fresh   { height:100%; border-radius:99px; background:linear-gradient(90deg,#14532d,#22c55e,#5efc8d); box-shadow:0 0 10px rgba(94,252,141,0.3); }
.fill-spoiled { height:100%; border-radius:99px; background:linear-gradient(90deg,#7f1d1d,#ef4444,#ff6b6b); box-shadow:0 0 10px rgba(255,107,107,0.3); }
.fill-low     { height:100%; border-radius:99px; background:linear-gradient(90deg,#78350f,#f59e0b,#fcd34d); }

/* ── Warning ── */
.warn-box {
    background: var(--warn-glow);
    border: 1px solid var(--warn-dim);
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    font-size: 0.79rem;
    color: var(--warn);
    margin-bottom: 1.2rem;
    line-height: 1.6;
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
}

/* ── Breakdown ── */
.bk-title { font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:var(--text-dim); margin-bottom:0.65rem; }
.bk-row   { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.45rem; }
.bk-name  { width:158px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:0.79rem; color:var(--text-muted); flex-shrink:0; }
.bk-name.hi { color:var(--text); font-weight:600; }
.bk-track { flex:1; background:var(--surface3); border-radius:99px; height:5px; overflow:hidden; }
.bk-fill  { height:100%; border-radius:99px; background:var(--border-light); transition:width 0.6s cubic-bezier(0.16,1,0.3,1); }
.bk-fill.hi { background:linear-gradient(90deg,var(--fresh-dim),var(--fresh-mid)); }
.bk-pct   { width:34px; text-align:right; font-size:0.74rem; font-weight:600; color:var(--text-muted); }
.bk-pct.hi { color:var(--text); }

/* ── Image caption ── */
.img-caption {
    font-size: 0.68rem;
    color: var(--text-dim);
    text-align: center;
    padding: 0.5rem 0 0;
    font-weight: 500;
    letter-spacing: 0.03em;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
}

/* ── Divider ── */
.divider { border:none; border-top:1px solid var(--border); margin:2.5rem 0; }

/* ── Empty state ── */
.empty {
    text-align: center;
    padding: 4rem 2rem;
    position: relative;
}
.empty::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 50%, rgba(94,252,141,0.03) 0%, transparent 70%);
    pointer-events: none;
}
.empty-icons {
    font-size: 2.2rem;
    margin-bottom: 1rem;
    letter-spacing: 0.4rem;
    display: block;
    filter: grayscale(0.2);
}
.empty-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.empty-sub   { font-size: 0.82rem; color: var(--text-dim); line-height: 1.6; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--surface2) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: var(--text-muted) !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--surface3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-light) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--surface) 0%, var(--bg) 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small { color: var(--text-muted) !important; }
[data-testid="stSidebar"] h3 {
    color: var(--text) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: var(--text-muted) !important; }

/* ── File uploader button ── */
[data-testid="stFileUploaderDropzoneInstructions"] button,
[data-testid="stBaseButton-secondary"] {
    background: var(--surface3) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: var(--fresh-glow) !important;
    border-color: var(--fresh-dim) !important;
    color: var(--fresh) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 128
MODEL_PATH    = "freshharvest_resnet50_final.pth"
LOW_CONF      = 0.65

FRUIT_EMOJI = {
    "Banana":"🍌","Lemon":"🍋","Lulo":"🟡","Mango":"🥭",
    "Orange":"🍊","Strawberry":"🍓","Tamarillo":"🔴","Tomato":"🍅",
}
HARD_FRUITS = {"Tomato","Tamarillo","Lulo","Strawberry"}

TTA_TRANSFORMS = [
    transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((250,250)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ColorJitter(brightness=0.15,contrast=0.1), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.Pad(20,padding_mode="reflect"), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]),
]

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        return None, None, None, None
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(2048,512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.4),
        nn.Linear(512,256),  nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(256, ckpt["num_classes"]),
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval().to(device)
    return m, ckpt["class_names"], device, ckpt.get("test_accuracy")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_class(raw):
    fruit  = raw[2:]
    status = "fresh" if raw[0].upper() == "F" else "spoiled"
    return status, fruit, FRUIT_EMOJI.get(fruit, "🌿")

def remove_background(image):
    try:
        from rembg import remove as rb
        no_bg    = rb(image.convert("RGBA"))
        white    = Image.new("RGBA", no_bg.size, (255,255,255,255))
        white.paste(no_bg, mask=no_bg.split()[3])
        return white.convert("RGB")
    except Exception:
        return image.convert("RGB")

def predict_tta(model, image, class_names, device):
    clean = remove_background(image)
    avg   = np.zeros(len(class_names))
    with torch.no_grad():
        for tfm in TTA_TRANSFORMS:
            avg += F.softmax(model(tfm(clean).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    avg /= len(TTA_TRANSFORMS)
    idx  = avg.argsort()[::-1][:5]
    return [(class_names[i], float(avg[i])) for i in idx], clean

def render_result(top5):
    name, conf = top5[0]
    status, fruit, emoji = parse_class(name)
    is_fresh  = status == "fresh"
    pct       = int(conf * 100)
    v_cls     = "verdict-fresh"   if is_fresh else "verdict-spoiled"
    b_cls     = "badge-fresh"     if is_fresh else "badge-spoiled"
    badge     = "✦ Fresh"        if is_fresh else "✦ Spoiled"
    fill      = "fill-fresh"      if is_fresh else ("fill-low" if conf < LOW_CONF else "fill-spoiled")

    # Warning
    if fruit in HARD_FRUITS and conf < 0.80:
        st.markdown(f'<div class="warn-box">⚠️ <strong>{fruit}</strong> resembles other red/orange fruits. Verify manually if confidence is below 80%.</div>', unsafe_allow_html=True)
    elif conf < LOW_CONF:
        st.markdown(f'<div class="warn-box">⚠️ <strong>Low confidence ({pct}%)</strong> — use a well-lit single-fruit image on a plain background for best results.</div>', unsafe_allow_html=True)

    # Verdict card
    st.markdown(f"""
    <div class="verdict {v_cls}">
        <div class="verdict-emoji">{emoji}</div>
        <div>
            <div class="verdict-badge {b_cls}"><span class="badge-dot"></span>{badge}</div>
            <div class="verdict-fruit">{fruit}</div>
            <div class="verdict-conf">{pct}% model confidence</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Confidence bar
    st.markdown(f"""
    <div class="cbar-wrap">
        <div class="cbar-header">
            <span class="cbar-lbl">Confidence</span>
            <span class="cbar-val">{pct}%</span>
        </div>
        <div class="cbar-track"><div class="{fill}" style="width:{pct}%"></div></div>
    </div>""", unsafe_allow_html=True)

    # Top-5
    max_p = top5[0][1] or 1.0
    rows  = "".join([f"""
    <div class="bk-row">
        <div class="bk-name {'hi' if i==0 else ''}">{parse_class(n)[2]} {'✅' if parse_class(n)[0]=='fresh' else '❌'} {parse_class(n)[1]}</div>
        <div class="bk-track"><div class="bk-fill {'hi' if i==0 else ''}" style="width:{int(p/max_p*100)}%"></div></div>
        <div class="bk-pct {'hi' if i==0 else ''}">{int(p*100)}%</div>
    </div>""" for i,(n,p) in enumerate(top5)])
    st.markdown(f'<div class="bk-title">All predictions</div>{rows}', unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model, CLASS_NAMES, DEVICE, saved_acc = load_model(MODEL_PATH)

if model is None:
    st.error(f"**Model not found:** `{MODEL_PATH}` — place the checkpoint in the same folder as `app.py`.")
    st.stop()

# ── Top bar ───────────────────────────────────────────────────────────────────
acc_html = f'<span class="pill pill-green"> {saved_acc:.1f}% accuracy</span>' if saved_acc else ""
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">
        <span class="logo-dot"></span>Fresh<span>Harvest</span> AI
    </div>
    <div class="topbar-pills">
        <span class="pill">ResNet50</span>
        <span class="pill">{len(CLASS_NAMES)} classes</span>
        {acc_html}
    </div>
</div>
<div class="main-wrap">
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Quality Detection</div>
    <h1 class="hero-title">Freshness <em>Inspector</em></h1>
    <p class="hero-sub" style="text-align:center;margin-left:auto;margin-right:auto;">Upload fruit images and the AI instantly classifies each one as fresh or spoiled.</p>
</div>
""", unsafe_allow_html=True)

# ── Uploader ──────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-section"><div class="upload-label">Drop images here</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload fruit images",
    type=["jpg","jpeg","png","webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    help="JPG, PNG, WEBP supported. Upload multiple files at once.",
)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
    <div class="empty">
        <div class="empty-title">No images uploaded yet</div>
        <div class="empty-sub">Drag fruit images into the box above — multiple files supported</div>
    </div>""", unsafe_allow_html=True)
else:
    for uploaded in uploaded_files:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        col_img, col_res = st.columns([4, 6], gap="large")

        with col_img:
            with st.spinner(f"Analysing {uploaded.name}…"):
                top5, clean = predict_tta(model, image, CLASS_NAMES, DEVICE)
            tab1, tab2 = st.tabs(["Original", "Processed"])
            with tab1:
                col_l, col_c, col_r = st.columns([1, 4, 1])
                with col_c:
                    st.image(image, use_container_width=True)
            with tab2:
                col_l, col_c, col_r = st.columns([1, 4, 1])
                with col_c:
                    st.image(clean, use_container_width=True)
            st.markdown(f'<div class="img-caption">📁 {uploaded.name}</div>', unsafe_allow_html=True)

        with col_res:
            render_result(top5)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Supported Fruits")
    for fruit, emoji in sorted(FRUIT_EMOJI.items()):
        st.markdown(f"{emoji} &nbsp; {fruit}", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Tips for best results")
    for tip in ["Good lighting, no harsh shadows","Single fruit, fills 70%+ of frame","Plain background works best","Whole fruit, not cut or sliced","In-focus, no motion blur"]:
        st.markdown(f"<small>{tip}</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Model info")
    st.markdown("<small>**Architecture:** ResNet50<br>**Strategy:** Transfer Learning + TTA<br>**Input:** 128×128<br>**Phases:** 3-phase fine-tuning</small>", unsafe_allow_html=True)
