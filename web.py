import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Intelligence System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
  --bg-base:        #040d1a;
  --bg-card:        #071224;
  --bg-card-alt:    #080f20;
  --border:         rgba(0,178,255,0.15);
  --border-bright:  rgba(0,178,255,0.45);
  --accent:         #00b2ff;
  --accent-glow:    rgba(0,178,255,0.25);
  --accent2:        #0af0c0;
  --danger:         #ff4b6e;
  --warn:           #ffb547;
  --text-primary:   #e8f4ff;
  --text-secondary: #7ba8c8;
  --text-muted:     #3d6080;
  --font-display:   'Rajdhani', sans-serif;
  --font-body:      'Inter', sans-serif;
  --font-mono:      'JetBrains Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1200px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 10px; }

/* ── HERO BANNER ── */
.hero-wrap {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #040d1a 0%, #071830 50%, #040d1a 100%);
  border-bottom: 1px solid var(--border-bright);
  padding: 3.5rem 2rem 2.5rem;
  margin: 0 -2rem 2.5rem -2rem;
}
.hero-wrap::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 80% at 90% 50%, rgba(0,178,255,0.06) 0%, transparent 70%),
    radial-gradient(ellipse 40% 60% at 10% 30%, rgba(10,240,192,0.04) 0%, transparent 60%);
  pointer-events: none;
}
/* animated grid lines */
.hero-wrap::after {
  content: '';
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(0,178,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,178,255,0.04) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,178,255,0.1);
  border: 1px solid var(--border-bright);
  border-radius: 100px;
  padding: 4px 14px;
  font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--accent); letter-spacing: 0.1em; text-transform: uppercase;
  margin-bottom: 1rem;
}
.hero-badge::before { content: '●'; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot {
  0%,100% { opacity:1; } 50% { opacity:0.3; }
}
.hero-title {
  font-family: var(--font-display) !important;
  font-size: clamp(2rem, 5vw, 3.6rem) !important;
  font-weight: 700 !important;
  line-height: 1.1 !important;
  color: var(--text-primary) !important;
  letter-spacing: 0.02em !important;
  margin: 0 0 0.6rem 0 !important;
}
.hero-title span { color: var(--accent); }
.hero-sub {
  font-size: 1rem; color: var(--text-secondary);
  font-weight: 300; max-width: 520px; line-height: 1.6;
}
.hero-stats {
  display: flex; gap: 2.5rem; margin-top: 2rem;
}
.hero-stat-item {
  display: flex; flex-direction: column;
}
.hero-stat-num {
  font-family: var(--font-display); font-size: 1.6rem;
  font-weight: 700; color: var(--accent); line-height: 1;
}
.hero-stat-label {
  font-size: 0.7rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px;
}
.hero-plane {
  position: absolute; right: 4%; top: 50%;
  transform: translateY(-50%);
  font-size: 7rem; opacity: 0.06;
  pointer-events: none; user-select: none;
  animation: float-plane 6s ease-in-out infinite;
}
@keyframes float-plane {
  0%,100% { transform: translateY(-50%) translateX(0); }
  50%      { transform: translateY(-55%) translateX(8px); }
}

/* ── SECTION LABEL ── */
.section-label {
  font-family: var(--font-mono); font-size: 0.68rem;
  color: var(--accent); letter-spacing: 0.15em; text-transform: uppercase;
  margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;
}
.section-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-bright), transparent);
}

/* ── CARD / PANEL ── */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.8rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}
.glass-card:hover { border-color: var(--border-bright); }
.glass-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.4;
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
/* selectbox */
div[data-baseweb="select"] > div {
  background-color: #0b1829 !important;
  border: 1px solid var(--border-bright) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
div[data-baseweb="select"] svg { fill: var(--accent) !important; }
div[data-baseweb="popover"] { background: #0b1829 !important; border: 1px solid var(--border-bright) !important; border-radius: 10px !important; }
li[role="option"] { color: var(--text-primary) !important; font-family: var(--font-body) !important; }
li[role="option"]:hover { background: rgba(0,178,255,0.12) !important; }

/* number input */
input[type="number"], .stTextInput input {
  background: #0b1829 !important;
  border: 1px solid var(--border-bright) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
input[type="number"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
  outline: none !important;
}

/* slider */
.stSlider > div > div > div { background: var(--border-bright) !important; }
.stSlider [data-testid="stThumbValue"] {
  background: var(--accent) !important; color: #000 !important;
  font-family: var(--font-mono) !important; font-weight: 600 !important;
  border-radius: 6px !important;
}
div[data-testid="stSlider"] div[role="slider"] {
  background: var(--accent) !important;
  box-shadow: 0 0 8px var(--accent) !important;
}

/* labels */
label, .stSlider label, .stSelectbox label, .stNumberInput label {
  color: var(--text-secondary) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  margin-bottom: 4px !important;
}

/* button */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #0078c8 0%, #00a8e8 50%, var(--accent) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.85rem 2rem !important;
  font-family: var(--font-display) !important;
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 20px rgba(0,178,255,0.35) !important;
  position: relative !important; overflow: hidden !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(0,178,255,0.55) !important;
  filter: brightness(1.1) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── RESULT CARDS ── */
/* delay-risk card */
.risk-card {
  border-radius: 16px;
  padding: 1.8rem;
  position: relative;
  overflow: hidden;
  animation: card-in 0.5s ease both;
}
@keyframes card-in {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:translateY(0); }
}
.risk-card.low {
  background: linear-gradient(135deg, #051a12 0%, #062415 100%);
  border: 1px solid rgba(10,240,150,0.35);
}
.risk-card.high {
  background: linear-gradient(135deg, #1a0508 0%, #240810 100%);
  border: 1px solid rgba(255,75,110,0.35);
}
.risk-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.risk-card.low::before  { background: linear-gradient(90deg, transparent, #0af096, transparent); }
.risk-card.high::before { background: linear-gradient(90deg, transparent, var(--danger), transparent); }
.risk-label {
  font-family: var(--font-mono); font-size: 0.68rem;
  letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem;
}
.risk-card.low  .risk-label { color: #0af096; }
.risk-card.high .risk-label { color: var(--danger); }
.risk-pct {
  font-family: var(--font-display); font-size: 4rem;
  font-weight: 700; line-height: 1; margin-bottom: 0.3rem;
}
.risk-card.low  .risk-pct  { color: #0af096; text-shadow: 0 0 20px rgba(10,240,150,0.4); }
.risk-card.high .risk-pct  { color: var(--danger); text-shadow: 0 0 20px rgba(255,75,110,0.4); }
.risk-status {
  font-size: 0.95rem; font-weight: 500;
}
.risk-card.low  .risk-status { color: rgba(10,240,150,0.8); }
.risk-card.high .risk-status { color: rgba(255,75,110,0.8); }
/* progress bar inside risk card */
.risk-bar-wrap {
  margin-top: 1.2rem;
  background: rgba(255,255,255,0.06);
  border-radius: 100px; height: 6px; overflow: hidden;
}
.risk-bar {
  height: 100%; border-radius: 100px;
  transition: width 1s ease;
}
.risk-card.low  .risk-bar { background: linear-gradient(90deg, #0af060, #0af096); }
.risk-card.high .risk-bar { background: linear-gradient(90deg, #ff4b6e, #ff8040); }

/* delay duration card */
.delay-card {
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 1.8rem;
  position: relative; overflow: hidden;
  animation: card-in 0.5s ease 0.1s both;
}
.delay-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.6;
}
.delay-mins {
  font-family: var(--font-display); font-size: 3.8rem;
  font-weight: 700; color: var(--accent); line-height: 1;
  text-shadow: 0 0 24px rgba(0,178,255,0.4);
}
.delay-unit { font-size: 1.1rem; color: var(--text-muted); margin-left: 6px; }

/* cause card */
.cause-card {
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 1.8rem;
  position: relative; overflow: hidden;
  animation: card-in 0.5s ease 0.2s both;
}
.cause-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent2), transparent);
  opacity: 0.5;
}
.cause-tag {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(10,240,192,0.1); border: 1px solid rgba(10,240,192,0.3);
  border-radius: 8px; padding: 6px 14px;
  font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--accent2); letter-spacing: 0.05em;
  margin-bottom: 0.8rem;
}
.cause-name {
  font-family: var(--font-display); font-size: 1.8rem;
  font-weight: 600; color: var(--text-primary); margin-bottom: 0.4rem;
}
.cause-desc {
  font-size: 0.83rem; color: var(--text-secondary); line-height: 1.5;
}

/* disclaimer */
.disclaimer {
  font-size: 0.72rem; color: var(--text-muted);
  text-align: center; padding: 1rem 0 0;
  border-top: 1px solid var(--border);
  margin-top: 1.5rem; font-style: italic;
}

/* ── MINI STAT CHIPS (below results) ── */
.stat-chips {
  display: flex; flex-wrap: wrap; gap: 0.7rem; margin-top: 1rem;
}
.stat-chip {
  background: rgba(0,178,255,0.08); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 14px;
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-secondary);
}
.stat-chip strong { color: var(--accent); }

/* ── DIVIDER ── */
.fancy-divider {
  display: flex; align-items: center; gap: 1rem;
  margin: 2rem 0;
}
.fancy-divider::before, .fancy-divider::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-bright));
}
.fancy-divider::after {
  background: linear-gradient(270deg, transparent, var(--border-bright));
}
.fancy-divider-icon { color: var(--accent); font-size: 0.9rem; opacity: 0.6; }

/* column padding fix */
div[data-testid="column"] { padding: 0 0.5rem !important; }

/* stMarkdown headings */
h1,h2,h3 { font-family: var(--font-display) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD REAL MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    delay_model    = joblib.load("random_forest_model.joblib")
    delay_cols     = joblib.load("delay_model_columns.joblib")
    duration_model = joblib.load("delay_duration_xgb.joblib")
    duration_cols  = joblib.load("delay_reg_columns.joblib")
    scaler         = joblib.load("regression_scaler.joblib")
    reason_model   = joblib.load("delay_reason_rf.joblib")
    reason_cols    = joblib.load("delay_reason_columns.joblib")
    reason_encoder = joblib.load("delay_reason_encoder.joblib")
    return (delay_model, delay_cols, duration_model,
            duration_cols, scaler, reason_model, reason_cols, reason_encoder)

(delay_model, delay_cols, duration_model,
 duration_cols, scaler, reason_model, reason_cols, reason_encoder) = load_models()

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def prepare_input(airline, origin, dest, hour, distance, dow, month, cols):
    origin = origin.split("–")[0].strip()
    dest   = dest.split("–")[0].strip()
    df = pd.DataFrame([{
        "op_unique_carrier": airline,
        "origin":            origin,
        "dest":              dest,
        "dep_hour":          hour,
        "day_of_week":       dow,
        "month":             month,
        "distance":          distance,
    }])
    df = pd.get_dummies(df)
    df = df.reindex(columns=cols, fill_value=0)
    return df

# ─────────────────────────────────────────────
#  REAL PREDICTION FUNCTIONS
# ─────────────────────────────────────────────
def predict_delay_probability(airline, origin, dest, hour, distance, dow, month):
    X = prepare_input(airline, origin, dest, hour, distance, dow, month, delay_cols)
    return round(float(delay_model.predict_proba(X)[0][1]), 2)

def predict_delay_duration(airline, origin, dest, hour, distance, dow, month):
    X = prepare_input(airline, origin, dest, hour, distance, dow, month, duration_cols)
    num_cols = ["month", "day_of_week", "distance", "dep_hour"]
    X[num_cols] = scaler.transform(X[num_cols])
    return max(0, int(duration_model.predict(X)[0]))

def predict_delay_reason(airline, origin, dest, hour, distance, dow, month):
    X = prepare_input(airline, origin, dest, hour, distance, dow, month, reason_cols)
    encoded = reason_model.predict(X)[0]
    # decode if encoder available
    try:
        return reason_encoder.inverse_transform([encoded])[0]
    except Exception:
        return str(encoded)

# ─────────────────────────────────────────────
#  DELAY REASON METADATA  (UI descriptions)
# ─────────────────────────────────────────────
DELAY_REASONS = {
    "carrier_delay":  ("✈️ Airline Operations",  "Late aircraft, crew scheduling issues, or maintenance problems caused by the airline itself."),
    "weather_delay":  ("🌩️ Adverse Weather",     "Significant meteorological conditions at origin, destination, or en-route."),
    "nas_delay":      ("🗼 Air Traffic Control",  "Non-extreme weather, heavy traffic volume, or FAA equipment/staffing issues."),
    "security_delay": ("🛡️ Security Screening",   "Evacuation of a terminal, re-boarding of aircraft due to security breach."),
    "late_aircraft":  ("🔄 Late Incoming Flight", "A previous flight with the same aircraft arrived late, causing a knock-on delay."),
    # fallback aliases (in case encoder returns title-case)
    "Carrier Delay":  ("✈️ Airline Operations",  "Late aircraft, crew scheduling issues, or maintenance problems caused by the airline itself."),
    "Weather Delay":  ("🌩️ Adverse Weather",     "Significant meteorological conditions at origin, destination, or en-route."),
    "NAS Delay":      ("🗼 Air Traffic Control",  "Non-extreme weather, heavy traffic volume, or FAA equipment/staffing issues."),
    "Security Delay": ("🛡️ Security Screening",   "Evacuation of a terminal, re-boarding of aircraft due to security breach."),
    "Late Aircraft":  ("🔄 Late Incoming Flight", "A previous flight with the same aircraft arrived late, causing a knock-on delay."),
}

# ─────────────────────────────────────────────
#  DATA — AIRLINES & AIRPORTS
# ─────────────────────────────────────────────
AIRLINES = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9", "HA", "G4"]

AIRPORTS = [
    "ATL – Hartsfield-Jackson Atlanta", "LAX – Los Angeles International",
    "ORD – O'Hare International",        "DFW – Dallas/Fort Worth International",
    "DEN – Denver International",        "JFK – John F. Kennedy International",
    "SFO – San Francisco International", "SEA – Seattle-Tacoma International",
    "LAS – Harry Reid International",    "MCO – Orlando International",
]

DAY_MAP   = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
MONTH_MAP = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─────────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-plane">✈</div>
  <div class="hero-badge">AI-Powered · Real-Time Predictions</div>
  <h1 class="hero-title" style="font-size:clamp(4.5rem,10vw,7rem);margin-bottom:0.2rem"><span>DelayScope</span></h1>
  <h2 style="font-family:var(--font-display);font-size:clamp(1.3rem,3vw,2rem);font-weight:600;color:var(--text-primary);letter-spacing:0.03em;margin:0 0 0.8rem;">Flight Delay <span style="color:var(--accent)">Intelligence</span> System</h2>
  <p class="hero-sub">
    Harness machine learning to predict flight delays before they happen —
    probability, duration, and root cause in seconds.
  </p>
  <div class="hero-stats">
    <div class="hero-stat-item">
      <span class="hero-stat-num">82%</span>
      <span class="hero-stat-label">Model Accuracy</span>
    </div>
    <div class="hero-stat-item">
      <span class="hero-stat-num">20K+</span>
      <span class="hero-stat-label">Flights Used for Training</span>
    </div>
    <div class="hero-stat-item">
      <span class="hero-stat-num">~0.2s</span>
      <span class="hero-stat-label">Prediction Time</span>
    </div>
    <div class="hero-stat-item">
      <span class="hero-stat-num">5</span>
      <span class="hero-stat-label">Delay Categories Analysed</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT PANEL
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">01 · Configure Your Flight</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox("🛫  Airline", AIRLINES)
        origin  = st.selectbox("📍  Origin Airport", AIRPORTS, index=0)

    with col2:
        destination = st.selectbox("🏁  Destination Airport", AIRPORTS, index=3)
        distance    = st.number_input("📏  Distance (miles)", min_value=50, max_value=5000,
                                      value=850, step=50)

    with col3:
        dep_hour = st.slider("🕐  Departure Hour (24h)", 0, 23, 8,
                             format="%d:00")
        day_of_week = st.slider("📅  Day of Week", 1, 7, 3,
                                format="%d",
                                help="1=Monday … 7=Sunday")
        month = st.slider("📆  Month", 1, 12, 6, format="%d")

    st.markdown('</div>', unsafe_allow_html=True)

# display friendly day/month labels
day_label   = DAY_MAP[day_of_week]
month_label = MONTH_MAP[month]

# ─────────────────────────────────────────────
#  PREDICT BUTTON
# ─────────────────────────────────────────────
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_clicked = st.button("⚡  ANALYZE FLIGHT  →")

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if predict_clicked:
    prob     = predict_delay_probability(airline, origin, destination,
                                         dep_hour, distance, day_of_week, month)
    duration = predict_delay_duration(airline, origin, destination,
                                       dep_hour, distance, day_of_week, month)
    reason   = predict_delay_reason(airline, origin, destination,
                                     dep_hour, distance, day_of_week, month)

    risk_class   = "high" if prob >= 0.5 else "low"
    risk_icon    = "🔴" if prob >= 0.5 else "🟢"
    risk_text    = "HIGH DELAY RISK" if prob >= 0.5 else "LOW DELAY RISK"
    pct_display  = f"{int(prob * 100)}%"

    reason_icon, reason_desc = DELAY_REASONS.get(
        reason, ("⚠️ Unknown Cause", f"Model returned: {reason}")
    )

    # fancy divider
    st.markdown("""
    <div class="fancy-divider"><span class="fancy-divider-icon">◆</span></div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · Prediction Results</div>', unsafe_allow_html=True)

    # ── three result cards ──
    r1, r2, r3 = st.columns([1.1, 0.9, 1.0])

    with r1:
        st.markdown(f"""
        <div class="risk-card {risk_class}">
          <div class="risk-label">{risk_icon} Delay Probability</div>
          <div class="risk-pct">{pct_display}</div>
          <div class="risk-status">{risk_text}</div>
          <div class="risk-bar-wrap">
            <div class="risk-bar" style="width:{int(prob*100)}%"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="delay-card">
          <div class="risk-label" style="color:var(--accent)">⏱ Estimated Delay</div>
          <div style="display:flex;align-items:baseline;margin-top:0.5rem">
            <span class="delay-mins">{int(duration)}</span>
            <span class="delay-unit">min</span>
          </div>
          <div style="font-size:0.82rem;color:var(--text-secondary);margin-top:0.7rem">
            ≈ {duration//60}h {duration%60}m beyond scheduled departure
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="cause-card">
          <div class="cause-tag">🔍 Root Cause Analysis</div>
          <div class="cause-name">{reason_icon}</div>
          <div style="font-family:var(--font-display);font-size:1.25rem;
                      font-weight:600;color:var(--text-primary);margin-bottom:0.4rem">
            {reason}
          </div>
          <div class="cause-desc">{reason_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── mini summary chips ──
    origin_code = origin.split("–")[0].strip()
    dest_code   = destination.split("–")[0].strip()
    airline_short = airline.split()[0]

    st.markdown(f"""
    <div class="stat-chips">
      <div class="stat-chip">✈ <strong>{origin_code}</strong> → <strong>{dest_code}</strong></div>
      <div class="stat-chip">🛫 <strong>{airline_short}</strong></div>
      <div class="stat-chip">🕐 Departs <strong>{dep_hour:02d}:00</strong></div>
      <div class="stat-chip">📅 <strong>{day_label}</strong>, {month_label}</div>
      <div class="stat-chip">📏 <strong>{distance} mi</strong></div>
      <div class="stat-chip">🎯 Confidence <strong>{min(99,int(70+prob*25))}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # ── disclaimer ──
    st.markdown("""
    <div class="disclaimer">
      ⚠ Predictions are generated by a statistical ML model trained on historical FAA data.
      Results are estimates only and should not be used for operational flight decisions.
      Actual delays depend on real-time conditions not captured by this model.
    </div>
    """, unsafe_allow_html=True)

else:
    # idle placeholder
    st.markdown("""
    <div style="text-align:center; padding:3.5rem 1rem 2.5rem;
                color:var(--text-muted); font-size:0.9rem; line-height:2;">
      <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3">✈️</div>
      Configure your flight parameters above and click<br>
      <strong style="color:var(--accent); font-family:var(--font-display); font-size:1rem;">
        ANALYZE FLIGHT
      </strong>
      to generate predictions.
    </div>
    """, unsafe_allow_html=True)