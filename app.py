import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg_start":   "#0f0c29",
    "bg_mid":     "#1a1a3e",
    "bg_end":     "#16213e",
    "surface":    "rgba(255,255,255,0.05)",
    "surface2":   "rgba(255,255,255,0.09)",
    "border":     "rgba(139,92,246,0.25)",
    "border2":    "rgba(99,102,241,0.35)",
    "primary":    "#8b5cf6",
    "secondary":  "#6366f1",
    "accent":     "#f59e0b",
    "accent2":    "#06b6d4",
    "text":       "#e2e8f0",
    "text_muted": "#94a3b8",
    "text_dim":   "#64748b",
    "success":    "#10b981",
    "sidebar_bg": "linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #16213e 100%)",
}

DOT_COLORS = [C["primary"], C["secondary"], C["accent"], C["accent2"], C["primary"], C["secondary"]]

st.set_page_config(
    page_title="Live Object Detection & Tracing",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    :root {{
        --primary:   {C['primary']};
        --secondary: {C['secondary']};
        --accent:    {C['accent']};
        --accent2:   {C['accent2']};
        --text:      {C['text']};
        --muted:     {C['text_muted']};
    }}

    html, body, .stApp {{
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, {C['bg_start']} 0%, {C['bg_mid']} 50%, {C['bg_end']} 100%) !important;
        color: {C['text']};
    }}

    #MainMenu {{ visibility: hidden; }}
    footer     {{ visibility: hidden; }}
    header     {{ visibility: hidden; }}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1140px;
    }}

    /* ══════════════════════════════════════════════════
       SIDEBAR FIXES — THE ONLY THING CHANGED
    ══════════════════════════════════════════════════ */

    /* Sidebar background */
    section[data-testid="stSidebar"] {{
        background: {C['sidebar_bg']} !important;
        border-right: 1px solid {C['border']} !important;
        box-shadow: 6px 0 32px rgba(139,92,246,0.12) !important;
    }}

    /* All text inside sidebar */
    section[data-testid="stSidebar"] * {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: {C['text']} !important;
    }}

    /* ── COLLAPSE BUTTON (the ← arrow inside the open sidebar) ──
       Make it a clearly visible glowing purple button               */
    [data-testid="stSidebarCollapseButton"] {{
        background: rgba(139,92,246,0.15) !important;
        border-radius: 10px !important;
    }}
    [data-testid="stSidebarCollapseButton"] button {{
        background: rgba(139,92,246,0.2) !important;
        border: 1px solid rgba(139,92,246,0.5) !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover {{
        background: rgba(139,92,246,0.45) !important;
        box-shadow: 0 0 14px rgba(139,92,246,0.6) !important;
    }}
    [data-testid="stSidebarCollapseButton"] button svg {{
        stroke: {C['primary']} !important;
        fill: none !important;
        width: 18px !important;
        height: 18px !important;
    }}

    /* ── COLLAPSED CONTROL (the ► tab that reopens the sidebar) ──
       Keep element in DOM flow so React click handlers still work.      */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        /* Cancel Streamlit's hiding transforms */
        transform: translateX(0) !important;
        margin-left: 0 !important;
        left: 0 !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
        overflow: visible !important;

        /* Bright gradient background */
        background: linear-gradient(180deg, {C['primary']} 0%, {C['secondary']} 100%) !important;
        border-radius: 0 16px 16px 0 !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        box-shadow: 4px 0 24px rgba(139,92,246,0.7), 0 0 40px rgba(139,92,246,0.3) !important;
        min-width: 32px !important;
        min-height: 48px !important;
        padding: 12px 8px !important;
        z-index: 999999 !important;
        animation: sidebarPulse 2.5s ease-in-out infinite !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }}
    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover {{
        filter: brightness(1.25) !important;
        box-shadow: 6px 0 32px rgba(139,92,246,0.95), 0 0 60px rgba(139,92,246,0.5) !important;
    }}

    /* Force the inner button to be visible and clickable */
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stSidebarCollapsedControl"] [data-testid="stBaseButton-headerNoPadding"],
    button[kind="headerNoPadding"] {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transform: none !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
        background: transparent !important;
        border: none !important;
        cursor: pointer !important;
        width: 100% !important;
        height: 100% !important;
    }}

    @keyframes sidebarPulse {{
        0%, 100% {{ box-shadow: 4px 0 24px rgba(139,92,246,0.7), 0 0 40px rgba(139,92,246,0.3); }}
        50%       {{ box-shadow: 4px 0 36px rgba(139,92,246,0.95), 0 0 60px rgba(139,92,246,0.5); }}
    }}

    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {{
        fill: white !important;
        stroke: white !important;
        width: 18px !important;
        height: 18px !important;
        filter: drop-shadow(0 0 4px rgba(255,255,255,0.8)) !important;
    }}

    /* ── HIDE STREAMLIT INTERNAL KEYBOARD SHORTCUT LABELS ──
       Streamlit embeds "keyboard_double" text inside collapse buttons.
       Our visibility overrides expose it; aggressively hide ALL text
       while preserving only the SVG arrow icon.                      */

    /* 1. Wipe out any rendered text on the button itself */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        color: transparent !important;
        text-shadow: none !important;
        font-size: 0 !important;
        line-height: 0 !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-indent: -9999px !important;
    }}

    /* 2. Kill pseudo-elements that might inject text */
    [data-testid="stSidebarCollapseButton"]::before,
    [data-testid="stSidebarCollapseButton"]::after,
    [data-testid="collapsedControl"]::before,
    [data-testid="collapsedControl"]::after,
    [data-testid="stSidebarCollapsedControl"]::before,
    [data-testid="stSidebarCollapsedControl"]::after {{
        display: none !important;
        content: none !important;
        opacity: 0 !important;
    }}

    /* 3. Hide every child element except SVG */
    [data-testid="stSidebarCollapseButton"] > *:not(svg),
    [data-testid="collapsedControl"] > *:not(svg),
    [data-testid="stSidebarCollapsedControl"] > *:not(svg),
    [data-testid="stSidebarCollapseButton"] svg ~ *,
    [data-testid="collapsedControl"] svg ~ *,
    [data-testid="stSidebarCollapsedControl"] svg ~ * {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        position: absolute !important;
        clip: rect(0,0,0,0) !important;
    }}

    /* 4. Deep-search: hide common Streamlit text containers */
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] div,
    [data-testid="stSidebarCollapseButton"] p,
    [data-testid="stSidebarCollapseButton"] label,
    [data-testid="stSidebarCollapseButton"] kbd,
    [data-testid="collapsedControl"] span,
    [data-testid="collapsedControl"] div,
    [data-testid="collapsedControl"] p,
    [data-testid="collapsedControl"] label,
    [data-testid="collapsedControl"] kbd,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] div,
    [data-testid="stSidebarCollapsedControl"] p,
    [data-testid="stSidebarCollapsedControl"] label,
    [data-testid="stSidebarCollapsedControl"] kbd {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        clip: rect(0,0,0,0) !important;
    }}

    /* 5. Restore SVG so the arrow stays visible */
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {{
        display: inline-block !important;
        font-size: 18px !important;
        line-height: 1 !important;
        text-indent: 0 !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: static !important;
        clip: auto !important;
        fill: white !important;
        stroke: white !important;
    }}

    /* ── Sidebar inner content styles ── */
    .sb-section {{
        font-size:0.65rem; font-weight:700; letter-spacing:2.5px;
        text-transform:uppercase; color:{C['accent']} !important;
        margin:18px 0 10px; display:flex; align-items:center; gap:7px;
    }}
    .sb-stat {{
        background:{C['surface2']}; border:1px solid {C['border']};
        border-radius:14px; padding:13px 14px; margin-bottom:10px;
        backdrop-filter:blur(8px);
    }}
    .sb-stat-label {{ font-size:0.68rem; color:{C['text_muted']} !important; font-weight:600; margin-bottom:2px; }}
    .sb-stat-value {{ font-size:1.3rem; font-weight:700; color:{C['primary']} !important; line-height:1; }}
    .sb-stat-sub   {{ font-size:0.66rem; color:{C['text_dim']} !important; margin-top:2px; }}
    .sb-tip-box {{
        background:{C['surface']}; border:1px solid {C['border']};
        border-radius:14px; padding:14px; backdrop-filter:blur(8px);
    }}
    .sb-tip-row {{
        display:flex; align-items:flex-start; gap:8px;
        font-size:0.76rem; color:{C['text_muted']} !important;
        margin-bottom:8px; line-height:1.4;
    }}
    .sb-badge {{
        display:inline-flex; align-items:center; gap:5px;
        background:linear-gradient(135deg,{C['primary']},{C['secondary']});
        color:white !important; border-radius:50px; padding:3px 12px;
        font-size:0.66rem; font-weight:700; letter-spacing:0.8px;
    }}
    .sb-divider {{
        height:1px;
        background:linear-gradient(90deg, transparent, {C['border2']}, transparent);
        margin:12px 0;
    }}
    .sb-brand {{
        font-family:'Syne',sans-serif !important;
        font-size:1rem; font-weight:800;
        color:{C['text']} !important; display:block;
    }}
    .sb-tagline {{
        font-size:0.68rem; color:{C['text_muted']} !important;
        display:block; margin-top:2px;
    }}
    .sb-logo {{
        display:flex; flex-direction:column;
        align-items:flex-start; gap:4px; padding:8px 0 4px;
    }}

    /* Slider / Select / Checkbox */
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stCheckbox > label {{
        color:{C['text']} !important; font-size:0.82rem !important; font-weight:500 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {{
        background:linear-gradient(90deg,{C['primary']},{C['accent2']}) !important;
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background:{C['surface2']} !important;
        border:1px solid {C['border2']} !important;
        border-radius:10px !important; color:{C['text']} !important;
    }}

    /* ══ Hero ══ */
    .hero-container {{
        background:linear-gradient(135deg,
            rgba(139,92,246,0.12) 0%,
            rgba(99,102,241,0.08) 50%,
            rgba(6,182,212,0.08) 100%);
        border-radius:28px; padding:50px 40px; margin-bottom:32px;
        text-align:center; border:1px solid {C['border2']};
        box-shadow:0 0 60px rgba(139,92,246,0.12),inset 0 1px 0 rgba(255,255,255,0.06);
        position:relative; overflow:hidden;
    }}
    .hero-container::before {{
        content:''; position:absolute; top:-60px; right:-60px;
        width:220px; height:220px;
        background:radial-gradient(circle,rgba(139,92,246,0.18) 0%,transparent 70%);
        pointer-events:none;
    }}
    .hero-container::after {{
        content:''; position:absolute; bottom:-60px; left:-60px;
        width:220px; height:220px;
        background:radial-gradient(circle,rgba(6,182,212,0.14) 0%,transparent 70%);
        pointer-events:none;
    }}
    .hero-logo {{ display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:10px; }}
    .hero-title {{
        font-family:'Syne',sans-serif; font-size:3rem; font-weight:800;
        background:linear-gradient(135deg,{C['primary']},{C['accent2']},{C['accent']});
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text; margin:0; line-height:1.15;
    }}
    .hero-badge {{
        display:inline-flex; align-items:center; gap:6px;
        background:rgba(139,92,246,0.15); border:1px solid {C['border2']};
        border-radius:50px; padding:5px 16px; font-size:0.72rem;
        font-weight:600; color:{C['accent2']}; letter-spacing:2px;
        text-transform:uppercase; margin-bottom:16px;
    }}
    .hero-desc {{
        font-size:0.95rem; color:{C['text_muted']}; max-width:560px;
        margin:0 auto; line-height:1.75;
    }}
    .dot-row {{ display:flex; justify-content:center; gap:8px; margin:16px 0 4px; }}
    .dot-row span {{ display:inline-block; width:6px; height:6px; border-radius:50%; }}

    /* ══ Feature cards ══ */
    .feat-card {{
        background:{C['surface2']}; border:1px solid {C['border']};
        border-radius:20px; padding:26px 18px; text-align:center;
        transition:transform .3s ease, box-shadow .3s ease, border-color .3s ease;
        height:100%; backdrop-filter:blur(10px);
    }}
    .feat-card:hover {{
        transform:translateY(-6px);
        box-shadow:0 20px 50px rgba(139,92,246,0.2);
        border-color:{C['primary']};
    }}
    .feat-icon-wrap {{
        width:54px; height:54px; border-radius:16px;
        display:flex; align-items:center; justify-content:center;
        margin:0 auto 14px;
    }}
    .feat-title {{ font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:{C['text']}; margin-bottom:6px; }}
    .feat-text  {{ font-size:0.78rem; color:{C['text_muted']}; line-height:1.45; }}

    /* ══ Camera panel ══ */
    .camera-panel {{
        background:{C['surface2']}; border:1px solid {C['border2']};
        border-radius:26px; padding:28px 28px 24px;
        box-shadow:0 0 50px rgba(139,92,246,0.10),inset 0 1px 0 rgba(255,255,255,0.05);
        margin:24px 0 10px; backdrop-filter:blur(12px);
    }}
    .camera-panel-header {{
        display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;
    }}
    .camera-panel-title-row {{ display:flex; align-items:center; gap:10px; }}
    .camera-panel-title {{ font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:800; color:{C['text']}; }}
    .camera-panel-subtitle {{ font-size:0.78rem; color:{C['text_muted']}; margin-top:2px; }}
    .status-bar {{
        display:inline-flex; align-items:center; gap:8px;
        background:rgba(16,185,129,0.10); border:1px solid rgba(16,185,129,0.25);
        border-radius:50px; padding:6px 16px;
    }}
    .status-dot {{
        width:8px; height:8px; background:{C['success']};
        border-radius:50%; animation:pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%   {{ box-shadow:0 0 0 0   rgba(16,185,129,0.5); }}
        70%  {{ box-shadow:0 0 0 8px rgba(16,185,129,0);   }}
        100% {{ box-shadow:0 0 0 0   rgba(16,185,129,0);   }}
    }}
    .status-text {{ font-size:0.76rem; color:{C['success']}; font-weight:700; letter-spacing:0.5px; }}

    /* ══ Inner video container ══ */
    .video-inner {{
        background:rgba(15,12,41,0.55); border:1px solid {C['border']};
        border-radius:18px; overflow:hidden; padding:0; position:relative;
    }}
    .video-inner::before, .video-inner::after {{
        content:''; position:absolute;
        width:28px; height:28px;
        border-color:{C['primary']}; border-style:solid;
        z-index:1; pointer-events:none;
    }}
    .video-inner::before {{ top:10px; left:10px; border-width:2px 0 0 2px; border-radius:4px 0 0 0; }}
    .video-inner::after  {{ bottom:10px; right:10px; border-width:0 2px 2px 0; border-radius:0 0 4px 0; }}
    .video-inner > div, .video-inner iframe, .video-inner video {{
        width:100% !important; border-radius:18px;
    }}

    /* ══ Buttons ══ */
    .stButton > button {{
        background:linear-gradient(135deg,{C['primary']},{C['secondary']}) !important;
        color:white !important; border:none !important;
        border-radius:12px !important; padding:10px 26px !important;
        font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important;
        font-size:0.88rem !important; letter-spacing:0.5px !important;
        box-shadow:0 8px 24px rgba(139,92,246,0.35) !important;
        transition:all .3s ease !important;
    }}
    .stButton > button:hover {{
        transform:translateY(-2px) !important;
        box-shadow:0 14px 34px rgba(139,92,246,0.55) !important;
    }}

    /* ══ Info boxes ══ */
    .info-box {{
        background:{C['surface']}; border:1px solid {C['border']};
        border-radius:20px; padding:24px; height:100%; backdrop-filter:blur(10px);
    }}
    .info-box-title {{
        font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
        color:{C['text']}; margin-bottom:16px; display:flex; align-items:center; gap:8px;
    }}
    .info-row {{
        display:flex; align-items:flex-start; gap:9px;
        margin-bottom:11px; font-size:0.82rem; color:{C['text_muted']}; line-height:1.45;
    }}

    /* ══ Divider ══ */
    .main-divider {{
        height:1px;
        background:linear-gradient(90deg,transparent,{C['primary']},{C['accent2']},transparent);
        margin:30px 0; opacity:0.4;
    }}

    /* ══ Section heading ══ */
    .section-heading {{
        text-align:center; margin-bottom:18px;
        display:flex; align-items:center; justify-content:center; gap:10px;
    }}
    .section-heading span {{
        font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:800; color:{C['text']};
    }}

    /* ══ Tag cloud ══ */
    .tag-cloud {{ display:flex; flex-wrap:wrap; gap:10px; justify-content:center; }}
    .tag-pill {{
        display:inline-flex; align-items:center; gap:7px;
        background:{C['surface2']}; border:1px solid {C['border']};
        border-radius:50px; padding:6px 16px;
        font-size:0.78rem; color:{C['text_muted']}; font-weight:500;
        transition:all .2s ease;
    }}
    .tag-pill:hover {{ border-color:{C['primary']}; color:{C['text']}; }}

    /* ══ Footer ══ */
    .footer-container {{ text-align:center; padding:28px 10px 10px; }}
    .footer-dots {{ display:flex; justify-content:center; gap:6px; margin-bottom:14px; }}
    .footer-text {{ font-size:0.78rem; color:{C['text_dim']}; letter-spacing:0.5px; }}
</style>
""", unsafe_allow_html=True)


# ── SVG helpers ───────────────────────────────────────────────────────────────
def svg_camera(size=32, color=C["primary"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"
            stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="12" cy="13" r="4" stroke="{color}" stroke-width="1.8"/></svg>"""

def svg_zap(size=20, color=C["accent"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"
               stroke="{color}" stroke-width="1.8"
               stroke-linecap="round" stroke-linejoin="round"/></svg>"""

def svg_eye(size=20, color=C["accent2"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"
            stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="12" cy="12" r="3" stroke="{color}" stroke-width="1.8"/></svg>"""

def svg_cpu(size=20, color=C["secondary"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <rect x="9" y="9" width="6" height="6" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M15 9V5h-2M9 9V5H7M15 15v4h-2M9 15v4H7
               M9 9H5V7M9 15H5v2M15 9h4V7M15 15h4v2"
            stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="2" y="2" width="20" height="20" rx="2" stroke="{color}" stroke-width="1.8"/></svg>"""

def svg_target(size=20, color=C["primary"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="1.8"/>
      <circle cx="12" cy="12" r="6"  stroke="{color}" stroke-width="1.8"/>
      <circle cx="12" cy="12" r="2"  stroke="{color}" stroke-width="1.8"/></svg>"""

def svg_layers(size=20, color=C["accent2"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <polygon points="12 2 2 7 12 12 22 7 12 2"
               stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <polyline points="2 17 12 22 22 17"
                stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <polyline points="2 12 12 17 22 12"
                stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

def svg_star(size=16, color=C["accent"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}"
        xmlns="http://www.w3.org/2000/svg">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02
                       12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>"""

def svg_check(size=14, color=C["success"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <polyline points="20 6 9 17 4 12" stroke="{color}" stroke-width="2.5"
                stroke-linecap="round" stroke-linejoin="round"/></svg>"""

def svg_info(size=14, color=C["secondary"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="1.8"/>
      <line x1="12" y1="8"  x2="12"    y2="12" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="12" y1="16" x2="12.01" y2="16" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/></svg>"""

def svg_heart(size=14, color=C["primary"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}"
        xmlns="http://www.w3.org/2000/svg">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06
               a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78
               1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>"""

def svg_sliders(size=14, color=C["primary"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <line x1="4"  y1="21" x2="4"  y2="14" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="4"  y1="10" x2="4"  y2="3"  stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="12" y1="21" x2="12" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="12" y1="8"  x2="12" y2="3"  stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="20" y1="21" x2="20" y2="16" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="20" y1="12" x2="20" y2="3"  stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="1"  y1="14" x2="7"  y2="14" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="9"  y1="8"  x2="15" y2="8"  stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <line x1="17" y1="16" x2="23" y2="16" stroke="{color}" stroke-width="2" stroke-linecap="round"/></svg>"""

def svg_activity(size=14, color=C["accent"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"
                stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

def svg_help(size=14, color=C["accent2"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="1.8"/>
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"
            stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <line x1="12" y1="17" x2="12.01" y2="17"
            stroke="{color}" stroke-width="2.2" stroke-linecap="round"/></svg>"""

def svg_box(size=18, color=C["accent2"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <polyline points="21 16 21 8 12 3 3 8 3 16 12 21 21 16"
                stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <polyline points="3.27 6.96 12 12.01 20.73 6.96"
                stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <line x1="12" y1="22.08" x2="12" y2="12"
            stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

def svg_video(size=20, color=C["primary"]):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg">
      <polygon points="23 7 16 12 23 17 23 7" stroke="{color}" stroke-width="1.8"
               stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="1" y="5" width="15" height="14" rx="2" ry="2"
            stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round"/></svg>"""

def tag_dot(color=C["primary"], size=10):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="{color}" opacity="0.7"/></svg>'

def dot_row_html():
    dots = "".join(f'<span style="background:{c};"></span>' for c in DOT_COLORS)
    return f'<div class="dot-row">{dots}</div>'


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(f"""
    <div class="sb-logo">
        {svg_camera(42, C['primary'])}
        <span class="sb-brand">Vision AI Studio</span>
        <span class="sb-tagline">Real-Time Object Detection</span>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sb-section">{svg_sliders(14,C["primary"])} Detection Settings</div>',
                unsafe_allow_html=True)

    confidence = st.slider(
        "Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
        help="Minimum confidence score for a detection to be shown"
    )
    model_choice = st.selectbox(
        "Model Variant",
        ["YOLOv8n (Fastest)", "YOLOv8s (Balanced)", "YOLOv8m (Accurate)"],
        index=0
    )
    tracker_choice = st.selectbox(
        "Tracker Algorithm", ["ByteTrack", "BoT-SORT"], index=0
    )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-section">{svg_eye(14,C["accent2"])} Display Options</div>',
                unsafe_allow_html=True)

    show_labels     = st.checkbox("Show Labels",         value=True)
    show_confidence = st.checkbox("Show Confidence",     value=True)
    show_tracking   = st.checkbox("Show Track IDs",      value=True)
    show_boxes      = st.checkbox("Show Bounding Boxes", value=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-section">{svg_activity(14,C["accent"])} Live Stats</div>',
                unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("""<div class="sb-stat">
            <div class="sb-stat-label">Model</div>
            <div class="sb-stat-value">v8n</div>
            <div class="sb-stat-sub">YOLOv8 Nano</div>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class="sb-stat">
            <div class="sb-stat-label">Classes</div>
            <div class="sb-stat-value">80</div>
            <div class="sb-stat-sub">COCO dataset</div>
        </div>""", unsafe_allow_html=True)

    cc, cd = st.columns(2)
    with cc:
        st.markdown(f"""<div class="sb-stat">
            <div class="sb-stat-label">Confidence</div>
            <div class="sb-stat-value">{int(confidence*100)}%</div>
            <div class="sb-stat-sub">threshold</div>
        </div>""", unsafe_allow_html=True)
    with cd:
        st.markdown("""<div class="sb-stat">
            <div class="sb-stat-label">Mode</div>
            <div class="sb-stat-value">Live</div>
            <div class="sb-stat-sub">async stream</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-section">{svg_help(14,C["accent2"])} Quick Tips</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-tip-box">
        <div class="sb-tip-row">{svg_check(13)} Good lighting improves accuracy</div>
        <div class="sb-tip-row">{svg_check(13)} Keep objects within the frame</div>
        <div class="sb-tip-row">{svg_check(13)} Move slowly for stable tracking</div>
        <div class="sb-tip-row">{svg_check(13)} Lower confidence = more detections</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:6px 0 12px;">
        <span class="sb-badge">v1.0.0 &nbsp;·&nbsp; stable</span>
    </div>
    """, unsafe_allow_html=True)


# ── Video callback ────────────────────────────────────────────────────────────
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.track(img, persist=True, conf=confidence, verbose=False)
    annotated = results[0].plot(
        labels=show_labels, conf=show_confidence, boxes=show_boxes
    )
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-container">
    {dot_row_html()}
    <div class="hero-logo">
        {svg_camera(46, C['primary'])}
        <h1 class="hero-title">Vision AI Studio</h1>
    </div>
    <div class="hero-badge">
        {svg_zap(14, C['accent2'])}
        Powered by YOLOv8 Neural Network
    </div>
    <p class="hero-desc">
        Experience real-time object detection powered by state-of-the-art AI.
        Adjust settings in the sidebar, point your camera, and watch every
        object get identified and tracked instantly.
    </p>
    {dot_row_html()}
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CARDS
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
cards = [
    (svg_target(22,C["primary"]),   "rgba(139,92,246,0.15)", "Smart Detection",
     "Identifies 80+ object categories with high precision"),
    (svg_zap(22,C["accent"]),       "rgba(245,158,11,0.15)", "Real-Time",
     "Lightning-fast async processing for smooth live video"),
    (svg_eye(22,C["accent2"]),      "rgba(6,182,212,0.15)",  "AI Tracking",
     "Persistent object IDs traced smoothly across frames"),
    (svg_layers(22,C["secondary"]), "rgba(99,102,241,0.15)", "YOLOv8",
     "State-of-the-art neural network optimised for speed"),
]
for col, (icon, bg, title, text) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(f"""
        <div class="feat-card">
            <div class="feat-icon-wrap" style="background:{bg};">{icon}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="camera-panel">
    <div class="camera-panel-header">
        <div>
            <div class="camera-panel-title-row">
                {svg_video(20, C['accent2'])}
                <div class="camera-panel-title">Live Camera Feed</div>
            </div>
            <div class="camera-panel-subtitle">
                Allow camera access and press <b>START</b> — tune settings in the sidebar
            </div>
        </div>
        <div class="status-bar">
            <div class="status-dot"></div>
            <div class="status-text">AI Ready &nbsp;·&nbsp; {int(confidence*100)}% conf</div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="video-inner">', unsafe_allow_html=True)

# ── WebRTC streamer with library-bug guard ──
# streamlit-webrtc has a race condition in its shutdown cleanup where
# _polling_thread can be None when stop() is called. We catch the
# resulting AttributeError so the app survives reruns gracefully.
try:
    webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
except AttributeError as e:
    if "is_alive" in str(e):
        st.warning("Camera stream was interrupted. Click **START** to reconnect.")
    else:
        raise

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIVIDER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TIPS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([3, 2])

tips = [
    "Ensure good lighting for the most accurate detection",
    "Keep objects clearly within the camera frame",
    "Move the camera slowly for stable object tracking",
    "Lower the confidence threshold to detect more objects",
    "Switch to YOLOv8m in the sidebar for higher accuracy",
]
model_rows = [
    ("Model",      model_choice.split(" ")[0]),
    ("Tracker",    tracker_choice),
    ("Confidence", f"{int(confidence*100)}% threshold"),
    ("Labels",     "On" if show_labels else "Off"),
    ("Track IDs",  "On" if show_tracking else "Off"),
]

with left:
    rows = "".join(
        f'<div class="info-row">{svg_check(14)}<span>{t}</span></div>'
        for t in tips
    )
    st.markdown(f"""
    <div class="info-box">
        <div class="info-box-title">{svg_star(16)} Tips for Best Results</div>
        {rows}
    </div>
    """, unsafe_allow_html=True)

with right:
    rows = "".join(
        f'<div class="info-row">{svg_info(14)}'
        f'<span><b style="color:{C["text"]};">{k}:</b>&nbsp;{v}</span></div>'
        for k, v in model_rows
    )
    st.markdown(f"""
    <div class="info-box">
        <div class="info-box-title">{svg_cpu(16)} Current Configuration</div>
        {rows}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAG CLOUD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="section-heading">
    {svg_box(20, C['accent2'])}
    <span>Detectable Objects</span>
</div>
""", unsafe_allow_html=True)

tag_data = [
    ("Person",   C["primary"]),   ("Car",      C["secondary"]),
    ("Dog",      C["accent2"]),   ("Cat",      C["primary"]),
    ("Bicycle",  C["secondary"]), ("Laptop",   C["accent2"]),
    ("Phone",    C["accent"]),    ("Chair",    C["primary"]),
    ("Airplane", C["secondary"]), ("Bus",      C["accent2"]),
    ("Umbrella", C["accent"]),    ("Handbag",  C["primary"]),
    ("Bottle",   C["secondary"]), ("Cup",      C["accent2"]),
    ("Plant",    C["accent"]),
]
pills = "".join(
    f'<div class="tag-pill">{tag_dot(color)}{label}</div>'
    for label, color in tag_data
)
st.markdown(f'<div class="tag-cloud">{pills}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
footer_dots = "".join(
    f'<div style="width:7px;height:7px;border-radius:50%;background:{c};opacity:0.6;"></div>'
    for c in DOT_COLORS
)
st.markdown(f"""
<div class="footer-container">
    <div class="footer-dots">{footer_dots}</div>
    <p class="footer-text">
        Crafted with &nbsp;{svg_heart(13)}&nbsp; using YOLOv8 &amp; Streamlit
        &nbsp;·&nbsp; Vision AI Studio 2026
    </p>
    <p class="footer-text" style="margin-top:5px; font-size:0.72rem;">
        Making AI powerful, one detection at a time
    </p>
</div>
""", unsafe_allow_html=True)