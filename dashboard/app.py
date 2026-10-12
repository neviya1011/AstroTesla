"""
Dustinel dashboard.
Run from the repo root (the AstroTesla folder):
    streamlit run dashboard/app.py

Optional: put your own photo at dashboard/assets/bg.jpg and it becomes the
background instead of the built-in space scene.
"""
import base64
import pathlib
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backend as b

LIMIT = 0.85
HERE = pathlib.Path(__file__).resolve().parent

st.set_page_config(page_title="Dustinel", page_icon="🛰️", layout="wide")


# ======================================================================
# LOOK AND FEEL (background art, solar panel picture, CSS)
# ======================================================================
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def background_url() -> str:
    """Your own photo if dashboard/assets/bg.jpg exists, else a drawn space scene."""
    photo = HERE / "assets" / "bg.jpg"
    if photo.exists():
        return "data:image/jpeg;base64," + _b64(photo.read_bytes())

    rnd = random.Random(7)
    stars = "".join(
        f'<circle cx="{rnd.randint(0, 1600)}" cy="{rnd.randint(0, 900)}" '
        f'r="{rnd.choice([0.6, 0.9, 1.3])}" fill="#fff" opacity="{rnd.uniform(.3, .9):.2f}"/>'
        for _ in range(200)
    )
    # a solar-panel wing seen at an angle, bottom-left
    grid = "".join(f'<line x1="{x}" y1="0" x2="{x}" y2="170" stroke="#7fb4ff" stroke-width="1"/>'
                   for x in range(60, 420, 60))
    grid += "".join(f'<line x1="0" y1="{y}" x2="420" y2="{y}" stroke="#7fb4ff" stroke-width="1"/>'
                    for y in range(42, 170, 42))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
<defs>
<radialGradient id="sp" cx="50%" cy="40%" r="80%"><stop offset="0" stop-color="#0d1838"/><stop offset="1" stop-color="#02030a"/></radialGradient>
<radialGradient id="moon" cx="35%" cy="30%" r="80%"><stop offset="0" stop-color="#f1f2f6"/><stop offset=".55" stop-color="#9aa0ad"/><stop offset="1" stop-color="#2a2f3d"/></radialGradient>
<radialGradient id="moonglow" cx="50%" cy="50%" r="50%"><stop offset=".84" stop-color="#cbd5e1" stop-opacity=".30"/><stop offset="1" stop-color="#cbd5e1" stop-opacity="0"/></radialGradient>
<radialGradient id="sun" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#fffbe0"/><stop offset=".6" stop-color="#ffc233"/><stop offset="1" stop-color="#ff8a00"/></radialGradient>
<radialGradient id="sunglow" cx="50%" cy="50%" r="50%"><stop offset=".2" stop-color="#ffb703" stop-opacity=".55"/><stop offset="1" stop-color="#ff8a00" stop-opacity="0"/></radialGradient>
<clipPath id="ec"><circle cx="1330" cy="800" r="300"/></clipPath>
</defs>
<rect width="1600" height="900" fill="url(#sp)"/>
{stars}
<circle cx="170" cy="140" r="130" fill="url(#sunglow)"/>
<circle cx="170" cy="140" r="46" fill="url(#sun)"/>
<circle cx="1330" cy="800" r="350" fill="url(#moonglow)"/>
<circle cx="1330" cy="800" r="300" fill="url(#moon)"/>
<g clip-path="url(#ec)" fill="#4b5263" opacity=".45">
<circle cx="1230" cy="690" r="55"/><circle cx="1400" cy="640" r="34"/><circle cx="1420" cy="770" r="70"/>
<circle cx="1290" cy="840" r="42"/><circle cx="1180" cy="790" r="26"/><circle cx="1350" cy="560" r="20"/>
</g>
<g transform="translate(70 620) skewY(-14)" opacity=".5">
<rect width="420" height="170" fill="#0e2a5c" stroke="#9cc7ff" stroke-width="2"/>{grid}
</g>
</svg>"""
    return "data:image/svg+xml;base64," + _b64(svg.encode())


def panel_html(T: float) -> str:
    """A little solar panel picture that gets dustier as T drops."""
    dust = min(max((1 - T) / 0.3, 0.0), 1.0)
    rnd = random.Random(3)
    cells = "".join(
        f'<rect x="{12 + c * 46}" y="{12 + r * 44}" width="42" height="40" rx="3" fill="url(#cell)"/>'
        for r in range(4) for c in range(6)
    )
    specks = "".join(
        f'<circle cx="{rnd.uniform(10, 290):.0f}" cy="{rnd.uniform(10, 190):.0f}" '
        f'r="{rnd.uniform(1, 4):.1f}" fill="#d8c3a0" opacity="{rnd.uniform(.4, .9):.2f}"/>'
        for _ in range(int(dust * 220))
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200">
<defs><linearGradient id="cell" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#2f6fd6"/><stop offset="1" stop-color="#0f2f7a"/></linearGradient></defs>
<rect width="300" height="200" rx="8" fill="#9aa7b8"/>{cells}
<rect width="300" height="200" rx="8" fill="#b89f7a" opacity="{dust * 0.45:.2f}"/>{specks}
</svg>"""
    return f'<img class="panelimg" src="data:image/svg+xml;base64,{_b64(svg.encode())}"/>'


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, .stApp, p, label, h1, h2, h3, h4, li, input, textarea { font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp {
  background: linear-gradient(rgba(2,3,10,.30), rgba(2,3,10,.55)), url("__BG__") center / cover fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton, #MainMenu, [data-testid="stMainMenu"] { display: none; }
.block-container { padding-top: 3rem; max-width: 1400px; }
section[data-testid="stSidebar"] > div {
  background: rgba(8,12,28,.72); backdrop-filter: blur(12px);
  border-right: 1px solid rgba(148,163,184,.18);
}
div[data-testid="stMetric"], div[data-testid="stPlotlyChart"], .card {
  background: rgba(10,16,36,.58); backdrop-filter: blur(10px);
  border: 1px solid rgba(148,163,184,.2); border-radius: 16px; padding: 14px 18px;
}
div[data-testid="stPlotlyChart"] { padding: 0; overflow: hidden; }
div[data-testid="stMetricValue"] { font-weight: 700; }
.brand { font-size: 3.2rem; font-weight: 700; letter-spacing: .01em; line-height: 1.05; color: #f1f5f9; }
.tag { color: #9fb3d1; font-size: 1.05rem; margin: .2rem 0 1.2rem 0; }
.pill { display:flex; align-items:center; gap:.7rem; padding:.85rem 1.2rem; border-radius:14px;
        font-size:1.15rem; font-weight:500; margin: .9rem 0 1rem 0; border:1px solid; }
.pill .dot { width:.8rem; height:.8rem; border-radius:50%; animation: pulse 1.6s infinite; }
.pill.ok   { background:rgba(22,101,52,.35);  border-color:#22c55e; color:#bbf7d0; }
.pill.ok .dot   { background:#22c55e; color:#22c55e; }
.pill.warn { background:rgba(133,77,14,.40);  border-color:#f59e0b; color:#fde68a; }
.pill.warn .dot { background:#f59e0b; color:#f59e0b; }
.pill.clean{ background:rgba(127,29,29,.45);  border-color:#ef4444; color:#fecaca; }
.pill.clean .dot{ background:#ef4444; color:#ef4444; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 currentColor;} 70%{box-shadow:0 0 0 10px transparent;} 100%{box-shadow:0 0 0 0 transparent;} }
.panelimg { width:100%; border-radius:10px; display:block; }
.small { color:#9fb3d1; font-size:.9rem; margin-top:.4rem; }
.key { color:#cbd5e1; font-size:.95rem; line-height:1.6; margin-top:.8rem; }
.sw { display:inline-block; width:1.4rem; height:.28rem; border-radius:2px; margin-right:.4rem; vertical-align:middle; }
@media (prefers-reduced-motion: reduce) { .pill .dot { animation: none; } }
</style>
"""
st.markdown(CSS.replace("__BG__", background_url()), unsafe_allow_html=True)

st.markdown(
    '<div class="brand">Dustinel</div>'
    '<div class="tag">Tells a lunar base exactly when to clean its solar panels.</div>',
    unsafe_allow_html=True,
)

# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.header("Controls")
mode = st.sidebar.radio("Data source", ["Simulator", "CSV replay", "Live Arduino"])
dust_rate = st.sidebar.slider("Dust rate", 0.3, 3.0, 1.0)
noise = st.sidebar.slider("Sensor noise", 0.005, 0.05, 0.02)

if "hist" not in st.session_state:
    st.session_state.hist = []      # live readings
    st.session_state.pos = 10       # CSV replay position
    st.session_state.play_hour = 20  # simulator "play" position

play = False
hour = None
if mode == "Simulator":
    play = st.sidebar.toggle("Play live simulation", value=False)
    if play:
        st.sidebar.caption("Dust builds up one hour per second.")
    else:
        hour = st.sidebar.slider("Current hour", 20, 353, 150)

default_csv_path = str(HERE.parent / "data" / "fake_run.csv")
csv_path = st.sidebar.text_input("CSV file (replay)", default_csv_path)
port = st.sidebar.text_input("Serial port (live)", "COM3")

if st.sidebar.button("Reset replay / live history"):
    st.session_state.hist = []
    st.session_state.pos = 10
    st.session_state.play_hour = 20

if st.sidebar.button("Run dust shield (EDS)"):
    st.session_state.hist = []
    st.session_state.pos = 10
    st.session_state.play_hour = 20
    st.toast("Dust shield ran. Panel is clean again.", icon="🧹")

real = [k for k, v in b.USING_REAL.items() if v]
st.sidebar.caption(f"Teammate code loaded: {', '.join(real) or 'none yet (stand-ins)'}")


# ======================================================================
# DATA SOURCES
# ======================================================================
def read_arduino():
    """Read one 'A0,A1' line from the Arduino and return T = dusty / reference."""
    import serial
    if "ser" not in st.session_state:                     # open the port once
        st.session_state.ser = serial.Serial(port, 9600, timeout=2)
    line = st.session_state.ser.readline().decode(errors="ignore").strip()
    try:
        a0, a1 = map(float, line.split(","))
        return a0 / a1 if a1 > 0 else None
    except ValueError:
        return None


def get_raw():
    """Return the raw T readings as a pandas Series, from whichever source is chosen."""
    if mode == "Simulator":
        n = st.session_state.play_hour if play else hour
        return b.simulate_day(dust_rate, noise).raw_T.iloc[:n].reset_index(drop=True)

    if mode == "CSV replay":
        try:
            df = pd.read_csv(csv_path)                    # needs columns: a0,a1
        except FileNotFoundError:
            st.error(f"File not found: {csv_path}")
            return pd.Series(dtype=float)
        st.session_state.pos = min(st.session_state.pos + 1, len(df))
        df = df.iloc[:st.session_state.pos]
        return (df.a0 / df.a1).reset_index(drop=True)

    # Live Arduino
    try:
        t = read_arduino()
    except Exception as e:
        st.error(f"Arduino not reachable on {port}: {e}")
        return pd.Series(dtype=float)
    if t is not None:
        st.session_state.hist.append(t)
    return pd.Series(st.session_state.hist, dtype=float)


STATUS_UI = {
    "OK":        ("ok",    "All clear: the panel is clean enough"),
    "WARN":      ("warn",  "Limit coming soon: get ready to clean"),
    "CLEAN_NOW": ("clean", "Below the limit: run the dust shield now"),
}


# ======================================================================
# THE LIVE PANEL (refreshes every second when something is "live")
# ======================================================================
@st.fragment(run_every=1 if (mode != "Simulator" or play) else None)
def panel():
    if mode == "Simulator" and play:                      # move time forward
        st.session_state.play_hour = min(st.session_state.play_hour + 1, 353)

    raw = get_raw()
    if len(raw) < 3:
        st.info("Waiting for data...")
        return

    filt = b.filter_series(raw)                 # teammate: noise filter
    t_now = float(filt.iloc[-1])
    hours_left = float(b.predict_hours(filt))   # teammate: random forest forecast
    glitches = int((abs(raw - filt) > 0.1).sum())
    light_lost = max(0.0, 1 - t_now) * 100
    status = b.decide(t_now, hours_left)        # teammate: scheduler
    if t_now <= 0.85:
        status = "CLEAN_NOW"
    elif t_now <= 0.90:
        status = "WARN"

    css_class, message = STATUS_UI.get(status, ("ok", status))
    st.markdown(
        f'<div class="pill {css_class}"><span class="dot"></span>{status}: {message}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current T", f"{t_now:.3f}")
    c2.metric(f"Hours to {b.MODEL_TARGET_TEXT}", f"{hours_left:.0f} h")
    c3.metric("Light lost to dust", f"{light_lost:.1f} %")
    c4.metric("Glitches ignored", glitches)

    left, right = st.columns([3, 1.2])
    with left:
        x = list(range(len(raw)))
        fig = go.Figure()
        fig.add_hrect(y0=0.6, y1=LIMIT, fillcolor="rgba(239,68,68,.12)", line_width=0)
        fig.add_scatter(x=x, y=raw, name="Raw sensor", mode="lines",
                        line=dict(color="rgba(148,163,184,.55)", width=1))
        fig.add_scatter(x=x, y=filt, name="Filtered T", mode="lines",
                        line=dict(color="#38bdf8", width=3))
        if t_now > LIMIT and 0 < hours_left < 72:       # forecast to the limit
            fig.add_scatter(x=[x[-1], x[-1] + hours_left], y=[t_now, LIMIT],
                            name="Forecast", mode="lines+markers",
                            line=dict(color="#fbbf24", width=2, dash="dot"))
        fig.add_hline(y=LIMIT, line_dash="dash", line_color="#f87171",
                      annotation_text="limit 0.85", annotation_position="top left",
                      annotation_font_color="#fca5a5")
        fig.update_layout(
            template="plotly_dark", height=430, hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Transmission T", range=[0.6, 1.05]),
            xaxis_title="Time step (hours)",
            legend=dict(orientation="h", y=1.1), margin=dict(t=40, b=40, l=60, r=30),
        )
        st.plotly_chart(fig)
        st.markdown(
            '<div class="card key">'
            '<div><span class="sw" style="background:#94a3b8"></span><b>Raw sensor:</b> the noisy live reading, glitches included.</div>'
            '<div><span class="sw" style="background:#38bdf8"></span><b>Filtered T:</b> the cleaned value used for decisions.</div>'
            '<div><span class="sw" style="background:#f87171"></span><b>Limit 0.85:</b> below it (red zone) the panel is too dusty.</div>'
            '<div><span class="sw" style="background:#fbbf24"></span><b>Forecast:</b> when T is predicted to hit the limit.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="card">{panel_html(t_now)}'
            f'<div class="small">Panel A: dust haze grows as T falls.</div></div>',
            unsafe_allow_html=True,
        )
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=t_now, number=dict(font=dict(color="#e2e8f0")),
            gauge={"axis": {"range": [0.6, 1.0]}, "bar": {"color": "#e2e8f0", "thickness": .25},
                   "steps": [{"range": [0.6, LIMIT], "color": "rgba(239,68,68,.55)"},
                             {"range": [LIMIT, 1.0], "color": "rgba(34,197,94,.45)"}],
                   "threshold": {"line": {"color": "#fff", "width": 3}, "value": LIMIT}}))
        gauge.update_layout(height=260, margin=dict(t=50, b=20, l=50, r=50),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
        st.plotly_chart(gauge)

    # tell the rig which LED to light: 'O', 'W' or 'C'
    if mode == "Live Arduino" and "ser" in st.session_state:
        st.session_state.ser.write(status[0].encode())


panel()

st.caption("Prototype: talc is not charged lunar dust, and simulation parameters are illustrative.")