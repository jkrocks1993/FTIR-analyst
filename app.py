#!/usr/bin/env python3
"""
FTIR Spectra Processor - Fixed & Improved Version
=================================================
Interactive Streamlit app for FTIR data processing.

Fixes included:
- Robust ALS baseline correction (no more "inconsistent shapes" error)
- Automatic fallback when spectrum is too short after cropping
- Linear baseline set as default (safer)
- All previous features preserved (upload, demo, preprocessing, peak detection, export)
- Comprehensive functional group library

Run:
    pip install streamlit pandas numpy scipy plotly
    streamlit run ftir_processor_app_fixed.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
import plotly.graph_objects as go
import plotly.express as px
import zipfile
import io
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="FTIR Spectra Processor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ROBUST HELPER FUNCTIONS
# ============================================================

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction.
    Fixed version that avoids shape mismatch errors.
    """
    y = np.asarray(y, dtype=float)
    L = len(y)

    if L < 10:   # Too short after cropping → return flat baseline
        return np.zeros_like(y)

    D = sparse.diags([1, -2, 1], [0, -1, 1], shape=(L - 2, L))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


def linear_baseline(y):
    return np.linspace(y[0], y[-1], len(y))


def polynomial_baseline(wn, y, deg=3):
    coeffs = np.polyfit(wn, y, deg)
    return np.polyval(coeffs, wn)


def normalize_spectrum(y, method="minmax"):
    y = np.asarray(y, dtype=float)
    if method == "minmax":
        return (y - y.min()) / (y.ptp() + 1e-12)
    elif method == "snv":
        return (y - y.mean()) / (y.std() + 1e-12)
    elif method == "vector":
        return y / (np.linalg.norm(y) + 1e-12)
    return y


def load_ftir_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, header=None, sep=None, engine="python", comment="#")
        if df.shape[1] < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine="python", comment="#")
        df = df.iloc[:, :2].copy()
        df.columns = ["wavenumber", "intensity"]
        df["wavenumber"] = pd.to_numeric(df["wavenumber"], errors="coerce")
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df = df.dropna().sort_values("wavenumber", ascending=False).reset_index(drop=True)
        if len(df) < 50:
            raise ValueError("Too few data points after cleaning")
        return df
    except Exception as e:
        st.error(f"Failed to parse **{uploaded_file.name}**: {str(e)}")
        return None


def generate_example_spectrum():
    np.random.seed(42)
    wn = np.linspace(4000, 400, 3601)
    y = np.zeros_like(wn)
    y += 0.08 * np.sin(wn / 800) + 0.03 * (wn - 2200)**2 / 4e6

    def add_gaussian(wn_arr, y_arr, center, fwhm, height):
        sigma = fwhm / 2.355
        return y_arr + height * np.exp(-((wn_arr - center)**2) / (2 * sigma**2))

    y = add_gaussian(wn, y, 3400, 280, 0.75)
    y = add_gaussian(wn, y, 2924, 45, 0.55)
    y = add_gaussian(wn, y, 2853, 35, 0.40)
    y = add_gaussian(wn, y, 1728, 28, 0.92)
    y = add_gaussian(wn, y, 1605, 35, 0.25)
    y = add_gaussian(wn, y, 1455, 40, 0.30)
    y = add_gaussian(wn, y, 1378, 25, 0.22)
    y = add_gaussian(wn, y, 1245, 80, 0.45)
    y = add_gaussian(wn, y, 1050, 70, 0.65)
    y = add_gaussian(wn, y, 870, 30, 0.18)
    y += 0.015 * np.random.randn(len(wn))
    y = np.clip(y, 0, None)
    return wn, y


# Comprehensive functional group library
FUNCTIONAL_GROUPS = [
    ("O-H stretch (free, sharp)", 3580, 3650, "strong, sharp — alcohols, phenols (dilute)"),
    ("O-H stretch (H-bonded, broad)", 3200, 3550, "strong, broad — alcohols, carboxylic acids, water"),
    ("N-H stretch (primary amine)", 3300, 3500, "medium, often two bands — primary amines"),
    ("N-H stretch (secondary amine)", 3300, 3400, "medium — secondary amines"),
    ("N-H stretch (amides)", 3100, 3500, "medium-strong, broad — amides"),
    ("O-H stretch (carboxylic acids)", 2500, 3300, "very broad, strong — carboxylic acids (dimer)"),
    ("C-H stretch (aromatic)", 3000, 3100, "medium — aromatic rings"),
    ("=C-H stretch (alkenes)", 3020, 3080, "medium — alkenes"),
    ("C-H stretch (asym CH₃)", 2950, 2975, "strong — methyl groups"),
    ("C-H stretch (asym CH₂)", 2915, 2935, "strong — methylene groups"),
    ("C-H stretch (sym CH₃)", 2865, 2885, "strong — methyl groups"),
    ("C-H stretch (sym CH₂)", 2845, 2865, "strong — methylene groups"),
    ("C-H stretch (aldehyde)", 2700, 2900, "medium, two bands — aldehydes"),
    ("C≡C stretch (alkynes)", 2100, 2260, "weak-medium — terminal alkynes"),
    ("C≡N stretch (nitriles)", 2200, 2260, "sharp, medium-strong — nitriles"),
    ("N=C=O (isocyanates)", 2250, 2275, "very strong, sharp — isocyanates"),
    ("C=O stretch (acid chlorides)", 1780, 1820, "very strong — acid chlorides"),
    ("C=O stretch (esters)", 1730, 1750, "very strong — esters"),
    ("C=O stretch (aldehydes)", 1720, 1740, "strong — aldehydes"),
    ("C=O stretch (ketones)", 1705, 1725, "strong — ketones"),
    ("C=O stretch (carboxylic acids)", 1700, 1725, "strong, broad — carboxylic acids"),
    ("C=O stretch (amides I)", 1630, 1690, "strong — primary/secondary amides"),
    ("C=O stretch (conjugated)", 1650, 1680, "strong — conjugated carbonyls"),
    ("C=C stretch (alkenes)", 1620, 1680, "variable — alkenes"),
    ("C=C stretch (aromatic)", 1580, 1600, "strong — aromatic rings"),
    ("Amide II (N-H bend + C-N)", 1510, 1570, "strong — amides"),
    ("C-H bend (CH₂ scissor)", 1440, 1470, "medium — alkanes"),
    ("C-H bend (CH₃ asym)", 1445, 1465, "medium — methyl"),
    ("C-H bend (CH₃ sym umbrella)", 1365, 1390, "medium-strong — methyl groups"),
    ("C-O stretch (esters)", 1150, 1250, "very strong — esters"),
    ("C-O stretch (alcohols/ethers)", 1000, 1200, "strong — alcohols, ethers"),
    ("C-N stretch (amines)", 1020, 1250, "medium — amines"),
    ("C-O-C stretch (ethers)", 1050, 1150, "strong — ethers"),
    ("S-H stretch (thiols)", 2550, 2600, "weak, sharp — thiols"),
    ("S=O stretch (sulfoxides)", 1030, 1070, "strong — sulfoxides"),
    ("S=O stretch (sulfones)", 1120, 1160, "strong — sulfones"),
    ("S=O stretch (sulfonamides)", 1150, 1180, "strong — sulfonamides"),
    ("S=O stretch (sulfonates)", 1350, 1370, "very strong — sulfonates"),
    ("P=O stretch", 1200, 1300, "strong — phosphates, phosphonates"),
    ("P-O-C stretch", 950, 1050, "strong — phosphate esters"),
    ("C-F stretch", 1000, 1400, "very strong, multiple — fluorinated compounds"),
    ("C-Cl stretch", 600, 800, "strong — chlorinated compounds"),
    ("C-Br stretch", 500, 650, "strong — brominated compounds"),
    ("Carbonate (CO₃²⁻)", 1400, 1450, "strong — carbonates"),
    ("Sulfate (SO₄²⁻)", 1100, 1150, "strong — sulfates"),
    ("Phosphate (PO₄³⁻)", 1000, 1100, "strong — phosphates"),
    ("Nitrate (NO₃⁻)", 1350, 1380, "strong — nitrates"),
    ("Silicate / Si-O", 900, 1100, "very strong — silicates, quartz"),
    ("Aromatic C-H oop (monosubstituted)", 690, 710, "strong — monosubstituted benzene"),
    ("Aromatic C-H oop (ortho)", 735, 770, "strong — ortho-disubstituted"),
    ("Aromatic C-H oop (meta)", 680, 710, "strong — meta-disubstituted"),
    ("Aromatic C-H oop (para)", 810, 840, "strong — para-disubstituted"),
    ("N=O stretch (nitro)", 1500, 1550, "strong — nitro compounds"),
    # ========== ORGANIC FUNCTIONAL GROUPS ==========
    # O-H / N-H
    ("O-H stretch (free, sharp)", 3580, 3650, "strong, sharp — alcohols, phenols (dilute)"),
    ("O-H stretch (H-bonded, broad)", 3200, 3550, "strong, broad — alcohols, carboxylic acids, water"),
    ("N-H stretch (primary amine)", 3300, 3500, "medium, often two bands"),
    ("N-H stretch (secondary amine)", 3300, 3400, "medium"),
    ("N-H stretch (amides)", 3100, 3500, "medium-strong, broad"),
    ("O-H stretch (carboxylic acids, dimer)", 2500, 3300, "very broad, strong"),

    # C-H
    ("C-H stretch (aromatic)", 3000, 3100, "medium"),
    ("=C-H stretch (alkenes)", 3020, 3080, "medium"),
    ("C-H stretch (asym CH₃)", 2950, 2975, "strong"),
    ("C-H stretch (asym CH₂)", 2915, 2935, "strong"),
    ("C-H stretch (sym CH₃)", 2865, 2885, "strong"),
    ("C-H stretch (sym CH₂)", 2845, 2865, "strong"),
    ("C-H stretch (aldehyde)", 2700, 2900, "medium, characteristic doublet"),

    # Triple bonds
    ("C≡C stretch (alkynes)", 2100, 2260, "weak-medium"),
    ("C≡N stretch (nitriles)", 2200, 2260, "sharp, medium-strong"),
    ("N=C=O (isocyanates)", 2250, 2275, "very strong, sharp"),

    # Carbonyls
    ("C=O stretch (acid chlorides)", 1780, 1820, "very strong"),
    ("C=O stretch (esters)", 1730, 1750, "very strong"),
    ("C=O stretch (aldehydes)", 1720, 1740, "strong"),
    ("C=O stretch (ketones)", 1705, 1725, "strong"),
    ("C=O stretch (carboxylic acids)", 1700, 1725, "strong, broad"),
    ("C=O stretch (amides I)", 1630, 1690, "strong"),
    ("C=O stretch (conjugated)", 1650, 1680, "strong"),
    ("C=O stretch (anhydrides)", 1800, 1850, "strong, two bands"),

    # C=C / Aromatic / Amide II
    ("C=C stretch (alkenes)", 1620, 1680, "variable"),
    ("C=C stretch (aromatic)", 1580, 1600, "strong"),
    ("Amide II (N-H bend + C-N)", 1510, 1570, "strong"),
    ("Aromatic ring breathing", 1450, 1500, "medium"),

    # C-H bending
    ("C-H bend (CH₂ scissor)", 1440, 1470, "medium"),
    ("C-H bend (CH₃ asym)", 1445, 1465, "medium"),
    ("C-H bend (CH₃ sym umbrella)", 1365, 1390, "medium-strong"),

    # C-O / C-N
    ("C-O stretch (esters)", 1150, 1250, "very strong"),
    ("C-O stretch (alcohols/ethers)", 1000, 1200, "strong"),
    ("C-N stretch (amines)", 1020, 1250, "medium"),
    ("C-O-C stretch (ethers)", 1050, 1150, "strong"),

    # Sulfur
    ("S-H stretch (thiols)", 2550, 2600, "weak, sharp"),
    ("S=O stretch (sulfoxides)", 1030, 1070, "strong"),
    ("S=O stretch (sulfones)", 1120, 1160, "strong"),
    ("S=O stretch (sulfonamides)", 1150, 1180, "strong"),
    ("S=O stretch (sulfonates)", 1350, 1370, "very strong"),

    # Phosphorus
    ("P=O stretch", 1200, 1300, "strong"),
    ("P-O-C stretch", 950, 1050, "strong"),

    # Halogens
    ("C-F stretch", 1000, 1400, "very strong, multiple bands"),
    ("C-Cl stretch", 600, 800, "strong"),
    ("C-Br stretch", 500, 650, "strong"),

    # Nitro & others
    ("N=O stretch (nitro)", 1500, 1550, "strong"),
    ("C≡C-H (terminal alkyne)", 3300, 3320, "strong, sharp"),

    # ========== INORGANIC / METAL OXIDES / MINERALS ==========
    # Common Metal Oxides
    ("Ti-O (TiO₂)", 500, 700, "strong — anatase/rutile"),
    ("Fe-O (Fe₂O₃ hematite)", 500, 650, "strong"),
    ("Fe-O (Fe₃O₄ magnetite)", 550, 650, "strong"),
    ("Zn-O (ZnO)", 400, 600, "strong"),
    ("Cu-O (CuO)", 400, 600, "strong"),
    ("Cu₂O", 600, 650, "medium"),
    ("Al-O (Al₂O₃ alumina)", 500, 800, "strong"),
    ("Mg-O (MgO)", 400, 600, "medium-strong"),
    ("Ni-O (NiO)", 400, 550, "strong"),
    ("Co-O (Co₃O₄)", 550, 670, "strong"),
    ("Mn-O (MnO₂ / Mn₃O₄)", 500, 650, "strong"),
    ("Cr-O (Cr₂O₃)", 500, 650, "strong"),
    ("General M-O (metal oxides)", 400, 700, "variable — many transition metal oxides"),

    # Mixed oxides & Spinels
    ("Spinel structure (M-O)", 500, 700, "strong — ferrites, aluminates"),
    ("Perovskite (M-O)", 500, 700, "strong"),

    # Silicates & Silica
    ("Si-O stretch (SiO₂ / quartz)", 950, 1100, "very strong"),
    ("Si-O-Si (siloxanes/silicates)", 1000, 1100, "very strong"),
    ("Zeolite framework (T-O-T)", 950, 1100, "strong"),

    # Clays & Minerals
    ("Clay minerals (Al-O-Si / Si-O)", 900, 1100, "strong — kaolinite, montmorillonite"),
    ("Structural OH in clays/micas", 3600, 3700, "sharp"),
    ("Carbonate (CO₃²⁻) asymmetric", 1400, 1450, "strong — calcite, dolomite"),
    ("Carbonate out-of-plane bend", 850, 880, "medium"),
    ("Sulfate (SO₄²⁻)", 1100, 1150, "strong — gypsum, metal sulfates"),
    ("Phosphate (PO₄³⁻)", 1000, 1100, "strong — apatite, metal phosphates"),
    ("Nitrate (NO₃⁻)", 1350, 1380, "strong"),
    ("Perchlorate (ClO₄⁻)", 1080, 1120, "strong"),

    # Hydroxides & Water
    ("M-OH (metal hydroxides)", 800, 1100, "medium-strong"),
    ("Adsorbed/structural water (H-O-H bend)", 1600, 1650, "medium"),
]


def assign_functional_groups(peak_wn):
    assignments = []
    for name, low, high, note in FUNCTIONAL_GROUPS:
        if low <= peak_wn <= high:
            assignments.append(f"{name} ({note})")
    return "; ".join(assignments) if assignments else "No common match"


def apply_processing_pipeline(wn, y, params):
    log = []
    y = np.asarray(y, dtype=float).copy()
    wn = np.asarray(wn, dtype=float).copy()

    # 1. Crop
    if params.get("crop", True):
        mask = (wn >= params["min_wn"]) & (wn <= params["max_wn"])
        wn = wn[mask]
        y = y[mask]
        log.append(f"Cropped to {params['min_wn']:.0f}–{params['max_wn']:.0f} cm⁻¹")

    if len(y) < 10:
        return wn, y, "ERROR: Too few points after crop"

    # 2. Smoothing
    if params.get("smooth", True):
        try:
            y = savgol_filter(y, window_length=params["sg_window"],
                              polyorder=params["sg_poly"], mode="nearest")
            log.append(f"Smoothed (SavGol wl={params['sg_window']})")
        except Exception as e:
            log.append(f"Smoothing failed: {e}")

    # 3. Baseline correction (with safety)
    if params.get("baseline", True) and params.get("baseline_method") != "None":
        method = params["baseline_method"]
        if len(y) < 15:
            log.append("Skipped baseline (spectrum too short after cropping)")
        elif "ALS" in method:
            try:
                baseline = baseline_als(y, lam=params["als_lam"], p=params["als_p"], niter=params["als_niter"])
                y = y - baseline
                log.append("ALS baseline correction")
            except Exception as e:
                log.append(f"ALS failed, using Linear instead ({e})")
                y = y - linear_baseline(y)
        elif "Linear" in method:
            y = y - linear_baseline(y)
            log.append("Linear baseline removed")
        elif "Polynomial" in method:
            baseline = polynomial_baseline(wn, y, deg=params.get("poly_deg", 3))
            y = y - baseline
            log.append(f"Polynomial baseline (deg={params.get('poly_deg', 3)})")

    # 4. Normalization
    if params.get("normalize", True) and params.get("norm_method") != "None":
        method = params["norm_method"].lower().split()[0]
        y = normalize_spectrum(y, method=method)
        log.append(f"Normalized ({params['norm_method']})")

    return wn, y, " | ".join(log) if log else "No processing"


def create_spectrum_plot(spectra_data, selected_names, use_processed=True, peak_data=None):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

    for idx, name in enumerate(selected_names):
        if name not in spectra_data:
            continue
        d = spectra_data[name]

        if use_processed and d.get("processed_int") is not None:
            x = d.get("processed_wn", d["wn"])
            y = d["processed_int"]
            label = f"{name} (processed)"
        else:
            x = d["wn"]
            y = d["raw_int"]
            label = f"{name} (raw)"

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name=label,
            line=dict(color=colors[idx % len(colors)], width=1.8),
            hovertemplate="<b>%{fullData.name}</b><br>Wavenumber: %{x:.1f} cm⁻¹<br>Intensity: %{y:.4f}<extra></extra>"
        ))

    if peak_data and len(peak_data.get("peak_wn", [])) > 0:
        fig.add_trace(go.Scatter(
            x=peak_data["peak_wn"],
            y=peak_data["peak_int"],
            mode="markers+text",
            marker=dict(size=9, color="#FF2D55", symbol="diamond"),
            text=[f"{w:.0f}" for w in peak_data["peak_wn"]],
            textposition="top center",
            textfont=dict(size=9, color="#FF2D55"),
            name="Detected Peaks",
            hovertemplate="<b>Peak</b><br>%{x:.1f} cm⁻¹<br>Intensity: %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title="FTIR Spectra Overview",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity (a.u.)",
        xaxis=dict(autorange="reversed"),
        hovermode="x unified",
        template="plotly_white",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ============================================================
# MAIN APP
# ============================================================

st.title("🧬 FTIR Spectra Processor")
st.caption("Fixed version • Stable baseline correction • Comprehensive functional group library")

if "spectra_data" not in st.session_state:
    st.session_state.spectra_data = {}
if "last_peak_data" not in st.session_state:
    st.session_state.last_peak_data = None
if "processing_log" not in st.session_state:
    st.session_state.processing_log = {}

# Sidebar
with st.sidebar:
    st.header("⚙️ Quick Controls")
    if st.button("🧪 Load Demo Spectrum", use_container_width=True):
        wn, y = generate_example_spectrum()
        name = "Demo_FTIR_Spectrum"
        st.session_state.spectra_data[name] = {
            "wn": wn, "raw_int": y, "processed_int": None, "processed_wn": None,
            "meta": {"source": "demo"}
        }
        st.rerun()

    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.spectra_data = {}
        st.session_state.last_peak_data = None
        st.session_state.processing_log = {}
        st.rerun()

    st.divider()
    n_loaded = len(st.session_state.spectra_data)
    n_processed = sum(1 for d in st.session_state.spectra_data.values() if d.get("processed_int") is not None)
    st.metric("Spectra Loaded", n_loaded)
    st.metric("Processed", n_processed)

# Tabs
tab_upload, tab_process, tab_peaks, tab_export = st.tabs([
    "📁 Upload & Visualize", "⚙️ Preprocessing", "🔍 Peak Analysis", "📤 Export"
])

# TAB 1: Upload
with tab_upload:
    st.header("Upload FTIR Data")
    uploaded_files = st.file_uploader(
        "Drop CSV/TXT files (wavenumber + intensity columns)",
        type=["csv", "txt", "dat"], accept_multiple_files=True
    )

    if uploaded_files:
        new_count = 0
        for f in uploaded_files:
            if f.name not in st.session_state.spectra_data:
                df = load_ftir_file(f)
                if df is not None:
                    st.session_state.spectra_data[f.name] = {
                        "wn": df["wavenumber"].values,
                        "raw_int": df["intensity"].values,
                        "processed_int": None,
                        "processed_wn": None,
                        "meta": {"source": "upload", "filename": f.name}
                    }
                    new_count += 1
        if new_count:
            st.success(f"Loaded {new_count} new spectrum file(s)")

    if st.session_state.spectra_data:
        names = list(st.session_state.spectra_data.keys())
        selected = st.multiselect("Select spectra to display", names, default=names[:min(6, len(names))])

        if selected:
            fig = create_spectrum_plot(st.session_state.spectra_data, selected, use_processed=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload files or load the demo spectrum from the sidebar.")

# TAB 2: Preprocessing
with tab_process:
    st.header("Preprocessing Pipeline")

    if not st.session_state.spectra_data:
        st.warning("Load spectra first (Tab 1)")
    else:
        names = list(st.session_state.spectra_data.keys())
        selected = st.multiselect("Select spectra to process", names, default=names, key="proc_select")

        if selected:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**1. Crop**")
                crop = st.checkbox("Enable crop", True)
                min_wn = st.number_input("Min cm⁻¹", 200, 4000, 400, 50, disabled=not crop)
                max_wn = st.number_input("Max cm⁻¹", 200, 4000, 4000, 50, disabled=not crop)

            with col2:
                st.markdown("**2. Smoothing**")
                smooth = st.checkbox("Savitzky-Golay", True)
                sg_window = st.slider("Window (odd)", 5, 51, 11, 2, disabled=not smooth)
                sg_poly = st.slider("Poly order", 1, 5, 2, disabled=not smooth)

            with col3:
                st.markdown("**3. Baseline**")
                baseline = st.checkbox("Enable baseline", True)
                baseline_method = st.selectbox(
                    "Method",
                    ["Linear (simple detrend)", "ALS (recommended)", "Polynomial (deg 3)", "None"],
                    index=0,  # Linear is now default (safer)
                    disabled=not baseline
                )
                if "ALS" in baseline_method:
                    als_lam = st.number_input("λ (smoothness)", 1e3, 1e8, 1e5, 1e4, format="%.0e")
                    als_p = st.slider("p (asymmetry)", 0.001, 0.2, 0.01, 0.001, format="%.3f")
                    als_niter = st.slider("Iterations", 5, 25, 10)
                else:
                    als_lam, als_p, als_niter = 1e5, 0.01, 10
                poly_deg = st.slider("Poly degree", 1, 6, 3) if "Polynomial" in baseline_method else 3

            st.markdown("**4. Normalization**")
            norm = st.checkbox("Enable normalization", True)
            norm_method = st.selectbox(
                "Method", ["Min-Max (0–1)", "SNV", "Vector (L2)", "None"],
                disabled=not norm
            )

            if st.button("🚀 Apply Preprocessing", type="primary", use_container_width=True):
                params = {
                    "crop": crop, "min_wn": min_wn, "max_wn": max_wn,
                    "smooth": smooth, "sg_window": sg_window, "sg_poly": sg_poly,
                    "baseline": baseline, "baseline_method": baseline_method,
                    "als_lam": als_lam, "als_p": als_p, "als_niter": als_niter,
                    "poly_deg": poly_deg,
                    "normalize": norm, "norm_method": norm_method
                }

                success = 0
                for name in selected:
                    d = st.session_state.spectra_data[name]
                    new_wn, new_y, log = apply_processing_pipeline(d["wn"], d["raw_int"], params)
                    if "ERROR" not in log:
                        d["processed_wn"] = new_wn
                        d["processed_int"] = new_y
                        st.session_state.processing_log[name] = log
                        success += 1
                    else:
                        st.error(f"{name}: {log}")
                if success:
                    st.success(f"Processed {success} spectra successfully!")

            processed_sel = [n for n in selected if st.session_state.spectra_data[n].get("processed_int") is not None]
            if processed_sel:
                st.divider()
                st.subheader("Processed Spectra Preview")
                fig = create_spectrum_plot(st.session_state.spectra_data, processed_sel, use_processed=True)
                st.plotly_chart(fig, use_container_width=True)

# TAB 3: Peak Analysis
with tab_peaks:
    st.header("Peak Detection & Functional Group Assignment")

    if not st.session_state.spectra_data:
        st.warning("Load spectra first")
    else:
        processed_names = [n for n in st.session_state.spectra_data
                           if st.session_state.spectra_data[n].get("processed_int") is not None]

        if not processed_names:
            st.info("Process spectra in Tab 2 first")
        else:
            primary = st.selectbox("Primary spectrum for peak detection", processed_names)

            c1, c2, c3 = st.columns(3)
            with c1:
                height = st.slider("Min peak height", 0.02, 1.0, 0.12, 0.01)
            with c2:
                prominence = st.slider("Prominence", 0.01, 0.8, 0.06, 0.01)
            with c3:
                distance = st.slider("Min distance (points)", 5, 80, 15, 1)

            if st.button("🔎 Detect Peaks", type="primary"):
                d = st.session_state.spectra_data[primary]
                wn = d.get("processed_wn", d["wn"])
                y = d["processed_int"]

                peaks_idx, _ = find_peaks(y, height=height, prominence=prominence, distance=distance)
                if len(peaks_idx) == 0:
                    st.warning("No peaks found with current settings")
                else:
                    peak_wn = wn[peaks_idx]
                    peak_int = y[peaks_idx]
                    assignments = [assign_functional_groups(p) for p in peak_wn]

                    peak_df = pd.DataFrame({
                        "Wavenumber (cm⁻¹)": np.round(peak_wn, 1),
                        "Intensity": np.round(peak_int, 4),
                        "Possible Functional Groups": assignments
                    }).sort_values("Wavenumber (cm⁻¹)", ascending=False)

                    st.session_state.last_peak_data = {
                        "name": primary,
                        "peak_wn": peak_wn,
                        "peak_int": peak_int,
                        "df": peak_df
                    }
                    st.success(f"Detected {len(peaks_idx)} peaks")

            if st.session_state.last_peak_data and st.session_state.last_peak_data["name"] == primary:
                pk = st.session_state.last_peak_data
                st.subheader(f"Detected Peaks — {primary}")
                st.dataframe(pk["df"], use_container_width=True, hide_index=True)

                st.subheader("Annotated Spectrum")
                fig = create_spectrum_plot(
                    st.session_state.spectra_data, [primary],
                    use_processed=True, peak_data=pk
                )
                st.plotly_chart(fig, use_container_width=True)

# TAB 4: Export
with tab_export:
    st.header("Export Processed Data")

    processed_available = [n for n in st.session_state.spectra_data
                           if st.session_state.spectra_data[n].get("processed_int") is not None]

    if not processed_available:
        st.info("Process spectra first (Tab 2)")
    else:
        to_export = st.multiselect("Spectra to export", processed_available, default=processed_available)

        if st.button("📦 Create ZIP of Processed CSVs", disabled=len(to_export) == 0):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in to_export:
                    d = st.session_state.spectra_data[name]
                    df_out = pd.DataFrame({
                        "wavenumber_cm-1": d.get("processed_wn", d["wn"]),
                        "processed_intensity": d["processed_int"]
                    })
                    csv_bytes = df_out.to_csv(index=False).encode()
                    safe_name = name.replace(" ", "_").replace(".csv", "") + "_processed.csv"
                    zf.writestr(safe_name, csv_bytes)
            zip_buffer.seek(0)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇️ Download ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"FTIR_Processed_{ts}.zip",
                mime="application/zip",
                use_container_width=True
            )

# Footer
st.divider()
st.caption("FTIR Processor • Fixed baseline handling • June 2026")
st.markdown(
    "<div style='text-align:center; color:#888;'>Made with ❤️ for researchers • "
    "<a href='https://ko-fi.com/jayakrishnash001' target='_blank'>Support on Ko-fi</a></div>",
    unsafe_allow_html=True
)