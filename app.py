#!/usr/bin/env python3
"""
FTIR Spectra Processor v3
=========================
Clean, robust version with:

- Fixed baseline_als (stable across NumPy versions)
- Fixed peak area using scipy.integrate.trapezoid
- Removed weird colored region boxes
- Expanded functional groups (organics + inorganics/metal oxides)
- Open Specy integration (web + download ready spectrum)
- PubChem lookup buttons next to detected peaks

Run:
    streamlit run ftir_processor_v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapezoid
import plotly.graph_objects as go
import plotly.express as px
import zipfile
import io
from datetime import datetime
import urllib.parse

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FTIR Processor v3",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ROBUST CORE FUNCTIONS
# ============================================================

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """Robust ALS baseline correction."""
    from scipy.sparse import diags
    L = len(y)
    D = diags([1, -2, 1], [0, -1, 1], shape=(L, L)).tocsr()
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
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
            raise ValueError("Too few data points")
        return df
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
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


# ============================================================
# FUNCTIONAL GROUPS (Expanded)
# ============================================================
FUNCTIONAL_GROUPS = [
    # Organic
    ("O-H stretch (free, sharp)", 3580, 3650, "Alcohols, phenols"),
    ("O-H stretch (H-bonded, broad)", 3200, 3550, "Alcohols, carboxylic acids, water"),
    ("N-H stretch (primary amine)", 3300, 3500, "Primary amines, amides"),
    ("N-H stretch (secondary amine)", 3300, 3400, "Secondary amines"),
    ("C-H stretch (aromatic)", 3000, 3100, "Aromatic rings"),
    ("C-H stretch (asym CH₃)", 2950, 2975, "Methyl groups"),
    ("C-H stretch (asym CH₂)", 2915, 2935, "Methylene groups"),
    ("C-H stretch (sym)", 2850, 2885, "Alkanes"),
    ("C-H stretch (aldehyde)", 2700, 2900, "Aldehydes"),
    ("C≡C stretch", 2100, 2260, "Terminal alkynes"),
    ("C≡N stretch", 2200, 2260, "Nitriles"),
    ("C=O stretch (esters)", 1730, 1750, "Esters"),
    ("C=O stretch (aldehydes)", 1720, 1740, "Aldehydes"),
    ("C=O stretch (ketones)", 1705, 1725, "Ketones"),
    ("C=O stretch (carboxylic acids)", 1700, 1725, "Carboxylic acids"),
    ("C=O stretch (amides I)", 1630, 1690, "Amides"),
    ("C=C stretch (aromatic/alkenes)", 1580, 1680, "Aromatic rings, alkenes"),
    ("Amide II", 1510, 1570, "Amides"),
    ("C-H bend (CH₂)", 1440, 1470, "Alkanes"),
    ("C-H bend (CH₃ umbrella)", 1365, 1390, "Methyl groups"),
    ("C-O stretch", 1000, 1300, "Esters, alcohols, ethers"),
    ("C-F stretch", 1000, 1400, "Fluorinated compounds"),
    ("C-Cl stretch", 600, 800, "Chlorinated compounds"),
    ("N=O stretch (nitro)", 1500, 1550, "Nitro groups"),
    ("S=O stretch (sulfonates/sulfones)", 1120, 1370, "Sulfur compounds"),
    ("P=O stretch", 1200, 1300, "Phosphorus compounds"),

    # Inorganic / Metal Oxides / Minerals
    ("Ti-O (TiO₂)", 500, 700, "Titania"),
    ("Fe-O (Fe₂O₃ / Fe₃O₄)", 500, 650, "Iron oxides"),
    ("Zn-O (ZnO)", 400, 600, "Zinc oxide"),
    ("Cu-O (CuO / Cu₂O)", 400, 650, "Copper oxides"),
    ("Al-O (Al₂O₃)", 500, 800, "Alumina"),
    ("Mg-O (MgO)", 400, 600, "Magnesium oxide"),
    ("General M-O (metal oxides)", 400, 700, "Many metal oxides"),
    ("Si-O (SiO₂ / quartz)", 950, 1100, "Silica, quartz"),
    ("Si-O-Si (silicates / zeolites)", 1000, 1100, "Silicates, zeolites"),
    ("Clay minerals (Al-O-Si)", 900, 1100, "Clays"),
    ("Carbonate (CO₃²⁻)", 1400, 1450, "Carbonates"),
    ("Sulfate (SO₄²⁻)", 1100, 1150, "Sulfates"),
    ("Phosphate (PO₄³⁻)", 1000, 1100, "Phosphates"),
    ("Nitrate (NO₃⁻)", 1350, 1380, "Nitrates"),
    ("M-OH (metal hydroxides)", 800, 1100, "Metal hydroxides"),
    ("Structural OH in minerals", 3600, 3700, "Clays, micas"),
]


def assign_functional_groups(peak_wn):
    assignments = []
    for name, low, high, note in FUNCTIONAL_GROUPS:
        if low <= peak_wn <= high:
            assignments.append(name)
    return "; ".join(assignments) if assignments else "No common match"


def apply_processing_pipeline(wn, y, params):
    log = []
    y = np.asarray(y, dtype=float).copy()
    wn = np.asarray(wn, dtype=float).copy()

    if params.get("crop", True):
        mask = (wn >= params["min_wn"]) & (wn <= params["max_wn"])
        wn = wn[mask]
        y = y[mask]
        log.append(f"Cropped to {params['min_wn']:.0f}–{params['max_wn']:.0f} cm⁻¹")

    if len(y) < 10:
        return wn, y, "ERROR: Too few points"

    if params.get("smooth", True):
        try:
            y = savgol_filter(y, window_length=params["sg_window"],
                              polyorder=params["sg_poly"], mode="nearest")
            log.append("Smoothed")
        except Exception as e:
            log.append(f"Smoothing failed: {e}")

    if params.get("baseline", True) and params.get("baseline_method") != "None":
        method = params["baseline_method"]
        if "ALS" in method:
            baseline = baseline_als(y, lam=params["als_lam"], p=params["als_p"], niter=params["als_niter"])
            y = y - baseline
            log.append("ALS baseline")
        elif "Linear" in method:
            y = y - linear_baseline(y)
            log.append("Linear baseline")
        elif "Polynomial" in method:
            baseline = polynomial_baseline(wn, y, deg=params.get("poly_deg", 3))
            y = y - baseline
            log.append(f"Polynomial baseline (deg {params.get('poly_deg', 3)})")

    if params.get("normalize", True) and params.get("norm_method") != "None":
        method = params["norm_method"].lower().split()[0]
        y = normalize_spectrum(y, method=method)
        log.append(f"Normalized ({params['norm_method']})")

    return wn, y, " | ".join(log) if log else "No processing"


# ============================================================
# PLOTTING (Clean - No weird colored boxes)
# ============================================================

def create_spectrum_plot(spectra_data, selected_names, use_processed=True,
                         peak_data=None, primary_name=None, title="FTIR Spectra"):
    fig = go.Figure()
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

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
            line=dict(color=color_palette[idx % len(color_palette)], width=1.6),
            hovertemplate="<b>%{fullData.name}</b><br>Wavenumber: %{x:.1f} cm⁻¹<br>Intensity: %{y:.4f}<extra></extra>"
        ))

    if peak_data and primary_name and primary_name in selected_names:
        fig.add_trace(go.Scatter(
            x=peak_data["peak_wn"],
            y=peak_data["peak_int"],
            mode="markers+text",
            marker=dict(size=9, color="#FF2D55", symbol="diamond", line=dict(width=1, color="white")),
            text=[f"{w:.0f}" for w in peak_data["peak_wn"]],
            textposition="top center",
            textfont=dict(size=9, color="#FF2D55"),
            name="Detected Peaks",
            hovertemplate="<b>Peak</b><br>%{x:.1f} cm⁻¹<br>Intensity: %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity (a.u.)",
        xaxis=dict(autorange="reversed", showgrid=True, gridwidth=0.5),
        yaxis=dict(showgrid=True, gridwidth=0.5),
        hovermode="x unified",
        template="plotly_white",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=60, r=30, t=70, b=50)
    )
    return fig


def create_pubchem_search_url(functional_group_text):
    """Create a smart PubChem search URL based on functional group."""
    # Extract key terms
    search_terms = functional_group_text.split(";")[0].strip()
    query = urllib.parse.quote(search_terms)
    return f"https://pubchem.ncbi.nlm.nih.gov/#query={query}&tab=compound"


# ============================================================
# MAIN APP
# ============================================================

st.title("🧬 FTIR Spectra Processor v3")
st.caption("Open Specy + PubChem integration • All bugs fixed • Clean UI")

# Session state
if "spectra_data" not in st.session_state:
    st.session_state.spectra_data = {}
if "last_peak_data" not in st.session_state:
    st.session_state.last_peak_data = None
if "processing_log" not in st.session_state:
    st.session_state.processing_log = {}

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧪 Load Demo Spectrum", use_container_width=True):
        wn, y = generate_example_spectrum()
        st.session_state.spectra_data["Demo"] = {
            "wn": wn, "raw_int": y, "processed_int": None, "processed_wn": None,
            "meta": {"source": "demo"}
        }
        st.rerun()

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.spectra_data = {}
        st.session_state.last_peak_data = None
        st.session_state.processing_log = {}
        st.rerun()

    st.divider()
    n_loaded = len(st.session_state.spectra_data)
    n_processed = sum(1 for d in st.session_state.spectra_data.values() if d.get("processed_int") is not None)
    st.metric("Loaded", n_loaded)
    st.metric("Processed", n_processed)

# Tabs
tab_upload, tab_process, tab_peaks, tab_export = st.tabs([
    "📁 Upload", "⚙️ Preprocessing", "🔍 Peak Analysis", "📤 Export"
])

# TAB 1: Upload
with tab_upload:
    st.header("Upload Data")
    uploaded_files = st.file_uploader(
        "CSV/TXT files (wavenumber + intensity)",
        type=["csv", "txt", "dat"], accept_multiple_files=True
    )

    if uploaded_files:
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
        st.success(f"Loaded {len(uploaded_files)} file(s)")

    if st.session_state.spectra_data:
        names = list(st.session_state.spectra_data.keys())
        selected = st.multiselect("Display spectra", names, default=names[:min(6, len(names))])
        if selected:
            fig = create_spectrum_plot(st.session_state.spectra_data, selected, use_processed=False)
            st.plotly_chart(fig, use_container_width=True)

# TAB 2: Preprocessing
with tab_process:
    st.header("Preprocessing")

    if not st.session_state.spectra_data:
        st.warning("Upload data first")
    else:
        names = list(st.session_state.spectra_data.keys())
        selected = st.multiselect("Select spectra", names, default=names, key="proc")

        if selected:
            col1, col2, col3 = st.columns(3)
            with col1:
                crop = st.checkbox("Crop", True)
                min_wn = st.number_input("Min", 200, 4000, 400, 50, disabled=not crop)
                max_wn = st.number_input("Max", 200, 4000, 4000, 50, disabled=not crop)
            with col2:
                smooth = st.checkbox("Smooth (Savitzky-Golay)", True)
                sg_window = st.slider("Window", 5, 51, 11, 2, disabled=not smooth)
                sg_poly = st.slider("Poly order", 1, 5, 2, disabled=not smooth)
            with col3:
                baseline = st.checkbox("Baseline", True)
                baseline_method = st.selectbox(
                    "Method", ["ALS (recommended)", "Linear", "Polynomial", "None"],
                    disabled=not baseline
                )
                if "ALS" in baseline_method:
                    als_lam = st.number_input("λ", 1e3, 1e8, 1e5, 1e4, format="%.0e")
                    als_p = st.slider("p", 0.001, 0.2, 0.01, 0.001, format="%.3f")
                    als_niter = st.slider("Iterations", 5, 25, 10)
                else:
                    als_lam, als_p, als_niter = 1e5, 0.01, 10
                poly_deg = st.slider("Degree", 1, 6, 3) if "Polynomial" in baseline_method else 3

            norm = st.checkbox("Normalize", True)
            norm_method = st.selectbox("Method", ["Min-Max", "SNV", "Vector"], disabled=not norm)

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
                if success:
                    st.success(f"Processed {success} spectra!")

            processed_sel = [n for n in selected if st.session_state.spectra_data[n].get("processed_int") is not None]
            if processed_sel:
                fig = create_spectrum_plot(st.session_state.spectra_data, processed_sel, use_processed=True)
                st.plotly_chart(fig, use_container_width=True)

# TAB 3: Peak Analysis (with Open Specy + PubChem)
with tab_peaks:
    st.header("Peak Detection & Analysis")

    if not st.session_state.spectra_data:
        st.warning("Load data first")
    else:
        processed_names = [n for n in st.session_state.spectra_data
                           if st.session_state.spectra_data[n].get("processed_int") is not None]

        if not processed_names:
            st.info("Process spectra in Tab 2 first")
        else:
            primary = st.selectbox("Select spectrum", processed_names)

            c1, c2, c3 = st.columns(3)
            with c1:
                height = st.slider("Min height", 0.02, 1.0, 0.12, 0.01)
            with c2:
                prominence = st.slider("Prominence", 0.01, 0.8, 0.06, 0.01)
            with c3:
                distance = st.slider("Min distance", 5, 80, 15, 1)

            show_derivative = st.checkbox("Show Second Derivative", value=False)

            if st.button("🔎 Detect Peaks", type="primary"):
                d = st.session_state.spectra_data[primary]
                wn = d.get("processed_wn", d["wn"])
                y = d["processed_int"]

                peaks_idx, _ = find_peaks(y, height=height, prominence=prominence, distance=distance)
                if len(peaks_idx) == 0:
                    st.warning("No peaks found")
                else:
                    peak_wn = wn[peaks_idx]
                    peak_int = y[peaks_idx]

                    areas = []
                    for idx in peaks_idx:
                        start = max(0, idx - 15)
                        end = min(len(y), idx + 16)
                        area = trapezoid(y[start:end], wn[start:end])
                        areas.append(round(area, 4))

                    assignments = [assign_functional_groups(p) for p in peak_wn]

                    peak_df = pd.DataFrame({
                        "Wavenumber (cm⁻¹)": np.round(peak_wn, 1),
                        "Intensity": np.round(peak_int, 4),
                        "Area": areas,
                        "Functional Groups": assignments
                    }).sort_values("Wavenumber (cm⁻¹)", ascending=False)

                    st.session_state.last_peak_data = {
                        "name": primary, "peak_wn": peak_wn, "peak_int": peak_int, "df": peak_df
                    }
                    st.success(f"Detected {len(peaks_idx)} peaks")

            # Display results
            if st.session_state.last_peak_data and st.session_state.last_peak_data["name"] == primary:
                pk = st.session_state.last_peak_data
                st.subheader(f"Detected Peaks — {primary}")
                st.dataframe(pk["df"], use_container_width=True, hide_index=True)

                # Annotated plot
                st.subheader("Annotated Spectrum")
                fig = create_spectrum_plot(
                    st.session_state.spectra_data, [primary],
                    use_processed=True, peak_data=pk, primary_name=primary
                )
                st.plotly_chart(fig, use_container_width=True)

                # === Open Specy Integration ===
                st.subheader("🔬 Open Specy Matching")
                col_os1, col_os2 = st.columns(2)

                with col_os1:
                    if st.button("🌐 Open Open Specy Web App", use_container_width=True):
                        st.markdown("[Click here to open Open Specy](https://www.openspecy.org/)")

                with col_os2:
                    if st.button("⬇️ Download Spectrum for Open Specy", use_container_width=True):
                        d = st.session_state.spectra_data[primary]
                        wn = d.get("processed_wn", d["wn"])
                        y = d["processed_int"]
                        export_df = pd.DataFrame({"wavenumber": wn, "intensity": y})
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            "Download JSON/CSV for Open Specy",
                            data=csv,
                            file_name=f"{primary}_for_openspecy.csv",
                            mime="text/csv"
                        )

                st.caption("Tip: Upload the downloaded file to Open Specy for advanced spectral matching.")

                # === PubChem Lookup ===
                st.subheader("🔍 PubChem Lookup")
                st.caption("Click a button below to search PubChem for compounds related to each peak:")

                if pk.get("df") is not None:
                    for i, row in pk["df"].head(8).iterrows():
                        fg = row["Functional Groups"]
                        if fg and fg != "No common match":
                            search_url = create_pubchem_search_url(fg)
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{row['Wavenumber (cm⁻¹)']} cm⁻¹** — {fg[:60]}...")
                            with col2:
                                st.link_button("Search PubChem", search_url, use_container_width=True)

                # Second Derivative
                if show_derivative:
                    d = st.session_state.spectra_data[primary]
                    wn = d.get("processed_wn", d["wn"])
                    y = d["processed_int"]
                    try:
                        y_deriv2 = savgol_filter(y, window_length=11, polyorder=3, deriv=2, mode='nearest')
                        fig_deriv = go.Figure()
                        fig_deriv.add_trace(go.Scatter(x=wn, y=y_deriv2, mode='lines',
                                                       line=dict(color='#00B4D8', width=1.5)))
                        if pk.get("peak_wn") is not None:
                            peak_indices = [np.argmin(np.abs(wn - w)) for w in pk["peak_wn"]]
                            fig_deriv.add_trace(go.Scatter(
                                x=pk["peak_wn"], y=y_deriv2[peak_indices],
                                mode='markers', marker=dict(size=8, color='#FF2D55', symbol='diamond')
                            ))
                        fig_deriv.update_layout(
                            title="Second Derivative", xaxis_title="Wavenumber (cm⁻¹)",
                            yaxis_title="2nd Derivative", xaxis=dict(autorange="reversed"),
                            template="plotly_white", height=400, showlegend=False
                        )
                        st.plotly_chart(fig_deriv, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not compute second derivative: {e}")

# TAB 4: Export
with tab_export:
    st.header("Export Data")
    processed_available = [n for n in st.session_state.spectra_data
                           if st.session_state.spectra_data[n].get("processed_int") is not None]

    if not processed_available:
        st.info("Process spectra first")
    else:
        to_export = st.multiselect("Select spectra", processed_available, default=processed_available)
        if st.button("📦 Export as ZIP", disabled=len(to_export) == 0):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in to_export:
                    d = st.session_state.spectra_data[name]
                    df_out = pd.DataFrame({
                        "wavenumber_cm-1": d.get("processed_wn", d["wn"]),
                        "processed_intensity": d["processed_int"]
                    })
                    zf.writestr(f"{name}_processed.csv", df_out.to_csv(index=False))
            zip_buffer.seek(0)
            st.download_button("⬇️ Download ZIP", data=zip_buffer.getvalue(),
                               file_name="FTIR_Processed.zip", mime="application/zip")

st.divider()
st.caption("FTIR Processor v3 • Open Specy + PubChem ready • Clean & robust • June 2026")