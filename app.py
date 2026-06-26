#!/usr/bin/env python3
"""
FTIR Spectra Processor - Interactive Streamlit App
==================================================
A complete, self-contained Streamlit application for uploading, visualizing,
preprocessing, peak detection, functional group assignment, and exporting
FTIR (Fourier Transform Infrared) spectroscopy data.

Features:
- Multiple CSV/TXT file upload (wavenumber + intensity columns)
- Synthetic demo spectrum loader
- Interactive Plotly visualizations (overlay, reversed x-axis like real IR)
- Preprocessing pipeline: crop, Savitzky-Golay smoothing, baseline correction (ALS/Linear/Poly), normalization (MinMax/SNV/Vector)
- Peak detection with adjustable parameters + automatic functional group suggestions
- Batch processing + ZIP export of processed spectra
- Session persistence (no data loss on widget interaction)
- Clean, researcher-friendly UI

Requirements (install once):
    pip install streamlit pandas numpy scipy plotly

Usage:
    streamlit run ftir_processor_app.py

Author: Generated for Jayakrishna (environmental chemistry PhD workflow)
License: MIT (free to use/modify)
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
# CONFIG & PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="FTIR Processor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit/streamlit',
        'Report a bug': None,
        'About': "# FTIR Spectra Processor\nBuilt for environmental & analytical chemistry workflows."
    }
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares (ALS) baseline correction.
    Robust version that avoids shape mismatch errors.
    """
    from scipy.sparse import diags, csr_matrix
    from scipy.sparse.linalg import spsolve
    import numpy as np

    L = len(y)
    
    # Create second difference matrix and convert to CSR format for stability
    D = diags([1, -2, 1], [0, -1, 1], shape=(L, L)).tocsr()
    
    w = np.ones(L)
    
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return z
    
def linear_baseline(y):
    """Simple linear detrend connecting first and last points."""
    return np.linspace(y[0], y[-1], len(y))


def polynomial_baseline(wn, y, deg=3):
    """Polynomial baseline fit (global)."""
    coeffs = np.polyfit(wn, y, deg)
    return np.polyval(coeffs, wn)


def normalize_spectrum(y, method="minmax"):
    """Normalization methods common in spectroscopy."""
    y = np.asarray(y, dtype=float)
    if method == "minmax":
        return (y - y.min()) / (y.ptp() + 1e-12)
    elif method == "snv":
        return (y - y.mean()) / (y.std() + 1e-12)
    elif method == "vector":
        return y / (np.linalg.norm(y) + 1e-12)
    return y


def load_ftir_file(uploaded_file):
    """
    Robust loader for FTIR CSV/TXT files.
    Tries to auto-detect delimiter and header.
    Expects at least two numeric columns (wavenumber, intensity).
    """
    try:
        # First attempt: flexible read
        df = pd.read_csv(uploaded_file, header=None, sep=None, engine="python", comment="#")
        
        if df.shape[1] < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine="python", comment="#")
        
        # Take first two columns
        df = df.iloc[:, :2].copy()
        df.columns = ["wavenumber", "intensity"]
        
        # Coerce to numeric
        df["wavenumber"] = pd.to_numeric(df["wavenumber"], errors="coerce")
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        
        # Drop NaNs and sort descending (standard IR convention)
        df = df.dropna().sort_values("wavenumber", ascending=False).reset_index(drop=True)
        
        if len(df) < 50:
            raise ValueError("Too few data points after cleaning (<50). Check file format.")
        
        return df
    except Exception as e:
        st.error(f"Failed to parse **{uploaded_file.name}**: {str(e)}")
        st.info("Tip: Your file should have two numeric columns (wavenumber in cm⁻¹ and intensity/absorbance). "
                "Headers are ignored. Delimiters: comma, tab, or space are supported.")
        return None


def generate_example_spectrum():
    """
    Create a realistic synthetic mid-IR absorbance spectrum for demo/testing.
    Includes common functional group peaks + realistic baseline + noise.
    """
    np.random.seed(42)
    wn = np.linspace(4000, 400, 3601)
    y = np.zeros_like(wn)
    
    # Slow curved baseline (common in real FTIR)
    y += 0.08 * np.sin(wn / 800) + 0.03 * (wn - 2200)**2 / 4e6
    
    def add_gaussian(wn_arr, y_arr, center, fwhm, height):
        """Add a Gaussian peak (positive for absorbance mode)."""
        sigma = fwhm / 2.355
        return y_arr + height * np.exp(-((wn_arr - center)**2) / (2 * sigma**2))
    
    # === Common functional groups (realistic positions & shapes) ===
    # Broad O-H (H-bonded)
    y = add_gaussian(wn, y, 3400, 280, 0.75)
    # C-H stretches (alkanes)
    y = add_gaussian(wn, y, 2924, 45, 0.55)
    y = add_gaussian(wn, y, 2853, 35, 0.40)
    # C=O carbonyl (ester/ketone/acid)
    y = add_gaussian(wn, y, 1728, 28, 0.92)
    # C=C / aromatic
    y = add_gaussian(wn, y, 1605, 35, 0.25)
    # CH2/CH3 bends
    y = add_gaussian(wn, y, 1455, 40, 0.30)
    y = add_gaussian(wn, y, 1378, 25, 0.22)
    # C-O stretch region (alcohols/esters)
    y = add_gaussian(wn, y, 1245, 80, 0.45)
    y = add_gaussian(wn, y, 1050, 70, 0.65)
    # Fingerprint region example
    y = add_gaussian(wn, y, 870, 30, 0.18)
    
    # Add realistic noise
    y += 0.015 * np.random.randn(len(wn))
    y = np.clip(y, 0, None)  # absorbance can't be negative
    
    return wn, y


# Functional group library (mid-IR) - Expanded with more organics + inorganics/metal oxides
FUNCTIONAL_GROUPS = [
    # === Organic ===
    ("O-H stretch (free, sharp)", 3580, 3650, "Alcohols, phenols (dilute)"),
    ("O-H stretch (H-bonded, broad)", 3200, 3550, "Alcohols, carboxylic acids, water"),
    ("N-H stretch (primary amine)", 3300, 3500, "Primary amines, amides"),
    ("N-H stretch (secondary amine)", 3300, 3400, "Secondary amines"),
    ("C-H stretch (aromatic)", 3000, 3100, "Aromatic rings"),
    ("C-H stretch (asym CH₃)", 2950, 2975, "Methyl groups"),
    ("C-H stretch (asym CH₂)", 2915, 2935, "Methylene groups"),
    ("C-H stretch (sym CH₃/CH₂)", 2850, 2885, "Alkanes"),
    ("C-H stretch (aldehyde)", 2700, 2900, "Aldehydes (characteristic)"),
    ("C≡C stretch (alkynes)", 2100, 2260, "Terminal alkynes"),
    ("C≡N stretch (nitriles)", 2200, 2260, "Nitriles"),
    ("C=O stretch (acid chlorides)", 1780, 1820, "Acid chlorides"),
    ("C=O stretch (esters)", 1730, 1750, "Esters"),
    ("C=O stretch (aldehydes)", 1720, 1740, "Aldehydes"),
    ("C=O stretch (ketones)", 1705, 1725, "Ketones"),
    ("C=O stretch (carboxylic acids)", 1700, 1725, "Carboxylic acids"),
    ("C=O stretch (amides I)", 1630, 1690, "Amides"),
    ("C=C stretch (alkenes/aromatics)", 1580, 1680, "Alkenes, aromatic rings"),
    ("Amide II (N-H bend)", 1510, 1570, "Amides"),
    ("C-H bend (CH₂ scissor)", 1440, 1470, "Alkanes"),
    ("C-H bend (CH₃ umbrella)", 1365, 1390, "Methyl groups"),
    ("C-O stretch (esters/alcohols)", 1000, 1300, "Esters, alcohols, ethers"),
    ("C-F stretch", 1000, 1400, "Fluorinated compounds"),
    ("C-Cl stretch", 600, 800, "Chlorinated compounds"),
    ("N=O stretch (nitro)", 1500, 1550, "Nitro groups"),
    ("S=O stretch (sulfonates/sulfones)", 1120, 1370, "Sulfur compounds"),
    ("P=O stretch", 1200, 1300, "Phosphorus compounds"),

    # === Inorganic / Metal Oxides / Minerals ===
    ("Ti-O (TiO₂)", 500, 700, "Titania (anatase/rutile)"),
    ("Fe-O (Fe₂O₃ / Fe₃O₄)", 500, 650, "Iron oxides (hematite/magnetite)"),
    ("Zn-O (ZnO)", 400, 600, "Zinc oxide"),
    ("Cu-O (CuO / Cu₂O)", 400, 650, "Copper oxides"),
    ("Al-O (Al₂O₃)", 500, 800, "Alumina"),
    ("Mg-O (MgO)", 400, 600, "Magnesium oxide"),
    ("Ni-O / Co-O", 400, 650, "Nickel / Cobalt oxides"),
    ("Mn-O (MnO₂)", 500, 650, "Manganese oxides"),
    ("General M-O (metal oxides)", 400, 700, "Many transition metal oxides"),
    ("Si-O stretch (SiO₂ / quartz)", 950, 1100, "Silica, quartz"),
    ("Si-O-Si (silicates / zeolites)", 1000, 1100, "Silicates, zeolites"),
    ("Clay minerals (Al-O-Si)", 900, 1100, "Kaolinite, montmorillonite"),
    ("Carbonate (CO₃²⁻)", 1400, 1450, "Calcite, metal carbonates"),
    ("Sulfate (SO₄²⁻)", 1100, 1150, "Gypsum, metal sulfates"),
    ("Phosphate (PO₄³⁻)", 1000, 1100, "Apatite, metal phosphates"),
    ("Nitrate (NO₃⁻)", 1350, 1380, "Metal nitrates"),
    ("M-OH (metal hydroxides)", 800, 1100, "Metal hydroxides"),
    ("Structural OH in minerals", 3600, 3700, "Clays, micas"),
    # === Expanded Inorganic / Metal Oxides / Minerals ===
("Ti-O (TiO₂ anatase)", 500, 700, "Strong — common in pigments, photocatalysts"),
("Ti-O (TiO₂ rutile)", 400, 600, "Strong"),
("Fe-O (α-Fe₂O₃ hematite)", 500, 650, "Strong — rust, pigments"),
("Fe-O (Fe₃O₄ magnetite)", 550, 650, "Strong — magnetic nanoparticles"),
("Fe-O (γ-Fe₂O₃ maghemite)", 550, 650, "Strong"),
("Zn-O (ZnO)", 400, 600, "Strong — semiconductors, sunscreens"),
("Cu-O (CuO)", 400, 600, "Strong — catalysts"),
("Cu₂O", 600, 650, "Medium"),
("Al-O (Al₂O₃ corundum)", 500, 800, "Strong — abrasives, catalysts"),
("Mg-O (MgO periclase)", 400, 600, "Medium-strong"),
("Ni-O (NiO)", 400, 550, "Strong"),
("Co₃O₄ (spinel)", 550, 670, "Strong"),
("Mn-O (MnO₂)", 500, 650, "Strong — battery materials"),
("Cr₂O₃", 500, 650, "Strong — pigments"),
("General M-O (transition metal oxides)", 400, 700, "Broad region for many MOx"),

# Mixed oxides & Spinels
("Spinel (MFe₂O₄ type)", 500, 700, "Ferrites, aluminates"),
("Perovskite (ABO₃)", 500, 700, "Many functional ceramics"),

# Silicates & Framework Materials
("Si-O (quartz / SiO₂)", 950, 1100, "Very strong"),
("Si-O-Si (amorphous silica)", 1000, 1100, "Very strong"),
("Zeolites (T-O-T asymmetric)", 950, 1100, "Strong"),
("Feldspars", 900, 1100, "Strong"),
("Clay minerals (kaolinite, montmorillonite)", 900, 1100, "Strong"),

# Common Minerals & Salts
("Carbonate (calcite, dolomite)", 1400, 1450, "Very strong"),
("Carbonate (out-of-plane bend)", 850, 880, "Medium"),
("Sulfate (gypsum, anhydrite)", 1100, 1150, "Strong"),
("Phosphate (apatite, metal phosphates)", 1000, 1100, "Strong"),
("Nitrate", 1350, 1380, "Strong"),
("Perchlorate", 1080, 1120, "Strong"),
("Borate", 1200, 1400, "Variable"),

# Hydroxides & Water-related
("M-OH (metal hydroxides)", 800, 1100, "Medium-strong"),
("Structural OH (clays, micas, brucite)", 3600, 3700, "Sharp"),
("Adsorbed water (H-O-H)", 1600, 1650, "Medium"),
("Free water (O-H)", 3200, 3600, "Broad"),

# Other important inorganics
("Ti-O in titanates", 500, 700, "Strong"),
("Zr-O (ZrO₂)", 400, 600, "Strong — zirconia"),
("Ce-O (CeO₂)", 500, 600, "Strong — ceria, catalysts"),
("Sn-O (SnO₂)", 500, 700, "Strong — gas sensors"),
("WO₃", 600, 800, "Strong — electrochromic materials"),
("Mo-O (MoO₃)", 800, 1000, "Strong"),
("V-O (V₂O₅)", 800, 1000, "Strong"),
# O-H and N-H Stretching
    ("O-H stretch (free, sharp)", 3580, 3650, "Alcohols, phenols (dilute solution)"),
    ("O-H stretch (H-bonded, broad)", 3200, 3550, "Alcohols, carboxylic acids, water (broad)"),
    ("N-H stretch (primary amine)", 3300, 3500, "Primary amines (often two bands)"),
    ("N-H stretch (secondary amine)", 3300, 3400, "Secondary amines"),
    ("N-H stretch (amides)", 3100, 3500, "Amides (broad)"),
    ("O-H stretch (carboxylic acid dimer)", 2500, 3300, "Carboxylic acids (very broad)"),

    # C-H Stretching
    ("C-H stretch (aromatic)", 3000, 3100, "Aromatic rings"),
    ("=C-H stretch (alkenes)", 3020, 3080, "Alkenes"),
    ("C-H stretch (asym CH₃)", 2950, 2975, "Methyl groups"),
    ("C-H stretch (asym CH₂)", 2915, 2935, "Methylene groups"),
    ("C-H stretch (sym CH₃)", 2865, 2885, "Methyl groups"),
    ("C-H stretch (sym CH₂)", 2845, 2865, "Methylene groups"),
    ("C-H stretch (aldehyde)", 2700, 2900, "Aldehydes (characteristic doublet)"),

    # Triple Bonds
    ("C≡C stretch (alkynes)", 2100, 2260, "Terminal alkynes (weak-medium)"),
    ("C≡N stretch (nitriles)", 2200, 2260, "Nitriles (sharp)"),
    ("N=C=O (isocyanates)", 2250, 2275, "Isocyanates (very strong)"),

    # Carbonyl Region (Detailed)
    ("C=O stretch (acid chlorides)", 1780, 1820, "Acid chlorides (very strong)"),
    ("C=O stretch (esters)", 1730, 1750, "Esters (very strong)"),
    ("C=O stretch (aldehydes)", 1720, 1740, "Aldehydes (strong)"),
    ("C=O stretch (ketones)", 1705, 1725, "Ketones (strong)"),
    ("C=O stretch (carboxylic acids)", 1700, 1725, "Carboxylic acids (strong, broad)"),
    ("C=O stretch (amides I)", 1630, 1690, "Amides (strong)"),
    ("C=O stretch (conjugated)", 1650, 1680, "Conjugated carbonyls"),
    ("C=O stretch (anhydrides)", 1800, 1850, "Anhydrides (two bands)"),

    # C=C, Aromatic, Amide II
    ("C=C stretch (alkenes)", 1620, 1680, "Alkenes"),
    ("C=C stretch (aromatic)", 1580, 1600, "Aromatic rings"),
    ("Amide II (N-H bend + C-N)", 1510, 1570, "Amides (strong)"),
    ("Aromatic ring breathing", 1450, 1500, "Aromatic rings"),

    # C-H Bending
    ("C-H bend (CH₂ scissor)", 1440, 1470, "Alkanes"),
    ("C-H bend (CH₃ asym)", 1445, 1465, "Methyl groups"),
    ("C-H bend (CH₃ sym umbrella)", 1365, 1390, "Methyl groups (characteristic)"),

    # C-O, C-N, Fingerprint
    ("C-O stretch (esters)", 1150, 1250, "Esters (very strong)"),
    ("C-O stretch (alcohols/ethers)", 1000, 1200, "Alcohols, ethers (strong)"),
    ("C-N stretch (amines)", 1020, 1250, "Amines"),
    ("C-O-C stretch (ethers)", 1050, 1150, "Ethers"),

    # Sulfur Compounds
    ("S-H stretch (thiols)", 2550, 2600, "Thiols (weak, sharp)"),
    ("S=O stretch (sulfoxides)", 1030, 1070, "Sulfoxides (strong)"),
    ("S=O stretch (sulfones)", 1120, 1160, "Sulfones (strong)"),
    ("S=O stretch (sulfonamides)", 1150, 1180, "Sulfonamides"),
    ("S=O stretch (sulfonates)", 1350, 1370, "Sulfonates (very strong)"),

    # Phosphorus Compounds
    ("P=O stretch", 1200, 1300, "Phosphates, phosphonates (strong)"),
    ("P-O-C stretch", 950, 1050, "Phosphate esters"),

    # Halogens & Nitro
    ("C-F stretch", 1000, 1400, "Fluorinated compounds (very strong)"),
    ("C-Cl stretch", 600, 800, "Chlorinated compounds"),
    ("C-Br stretch", 500, 650, "Brominated compounds"),
    ("N=O stretch (nitro)", 1500, 1550, "Nitro groups (strong)"),

    # ========== INORGANIC / METAL OXIDES / MINERALS ==========
    
    # Common Metal Oxides
    ("Ti-O (TiO₂ anatase)", 500, 700, "Titania (pigments, photocatalysts)"),
    ("Ti-O (TiO₂ rutile)", 400, 600, "Titania"),
    ("Fe-O (α-Fe₂O₃ hematite)", 500, 650, "Hematite, rust, pigments"),
    ("Fe-O (Fe₃O₄ magnetite)", 550, 650, "Magnetite, magnetic nanoparticles"),
    ("Fe-O (γ-Fe₂O₃ maghemite)", 550, 650, "Maghemite"),
    ("Zn-O (ZnO)", 400, 600, "Zinc oxide (semiconductors)"),
    ("Cu-O (CuO)", 400, 600, "Copper(II) oxide"),
    ("Cu₂O", 600, 650, "Copper(I) oxide"),
    ("Al-O (Al₂O₃)", 500, 800, "Alumina, corundum"),
    ("Mg-O (MgO)", 400, 600, "Magnesium oxide (periclase)"),
    ("Ni-O (NiO)", 400, 550, "Nickel oxide"),
    ("Co₃O₄", 550, 670, "Cobalt oxide (spinel)"),
    ("Mn-O (MnO₂)", 500, 650, "Manganese dioxide"),
    ("Cr₂O₃", 500, 650, "Chromium(III) oxide"),
    ("Zr-O (ZrO₂)", 400, 600, "Zirconia"),
    ("Ce-O (CeO₂)", 500, 600, "Ceria (catalysts)"),
    ("Sn-O (SnO₂)", 500, 700, "Tin dioxide (gas sensors)"),
    ("WO₃", 600, 800, "Tungsten trioxide"),
    ("Mo-O (MoO₃)", 800, 1000, "Molybdenum trioxide"),
    ("V-O (V₂O₅)", 800, 1000, "Vanadium pentoxide"),

    # Mixed Oxides & Structures
    ("Spinel (MFe₂O₄ type)", 500, 700, "Ferrites, aluminates"),
    ("Perovskite (ABO₃)", 500, 700, "Perovskite oxides"),
    ("General M-O (metal oxides)", 400, 700, "Most transition metal oxides"),

    # Silicates & Framework Materials
    ("Si-O (quartz / SiO₂)", 950, 1100, "Quartz, crystalline silica"),
    ("Si-O-Si (amorphous silica)", 1000, 1100, "Amorphous silica, glass"),
    ("Zeolites (T-O-T)", 950, 1100, "Zeolites, molecular sieves"),
    ("Feldspars", 900, 1100, "Feldspar minerals"),
    ("Clay minerals (kaolinite, montmorillonite)", 900, 1100, "Clays and layered silicates"),

    # Common Minerals & Salts
    ("Carbonate (calcite, dolomite)", 1400, 1450, "Calcite, dolomite, metal carbonates"),
    ("Carbonate (out-of-plane bend)", 850, 880, "Carbonates (medium)"),
    ("Sulfate (gypsum, anhydrite)", 1100, 1150, "Gypsum, metal sulfates"),
    ("Phosphate (apatite, metal phosphates)", 1000, 1100, "Apatite, phosphates"),
    ("Nitrate (NO₃⁻)", 1350, 1380, "Metal nitrates"),
    ("Perchlorate (ClO₄⁻)", 1080, 1120, "Perchlorates"),
    ("Borate", 1200, 1400, "Borates"),

    # Hydroxides & Water
    ("M-OH (metal hydroxides)", 800, 1100, "Metal hydroxides"),
    ("Structural OH (clays, micas, brucite)", 3600, 3700, "Structural hydroxyl in minerals"),
    ("Adsorbed/structural water (H-O-H)", 1600, 1650, "Water bending mode"),
    ("Free/bound water (O-H)", 3200, 3600, "Water stretching (broad)"),
]


def assign_functional_groups(peak_wn):
    """Return comma-separated list of possible functional groups for a given wavenumber."""
    assignments = []
    for name, low, high, note in FUNCTIONAL_GROUPS:
        if low <= peak_wn <= high:
            assignments.append(f"{name}")
    return "; ".join(assignments) if assignments else "No common match (check fingerprint region or library)"


def apply_processing_pipeline(wn, y, params):
    """
    Apply the full preprocessing chain in logical order.
    Returns processed wn, y and a short log string.
    """
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
            y = savgol_filter(y, 
                              window_length=params["sg_window"], 
                              polyorder=params["sg_poly"], 
                              mode="nearest")
            log.append(f"Smoothed (SavGol wl={params['sg_window']}, poly={params['sg_poly']})")
        except Exception as e:
            log.append(f"Smoothing failed: {e}")
    
    # 3. Baseline correction
    if params.get("baseline", True) and params.get("baseline_method") != "None":
        method = params["baseline_method"]
        if "ALS" in method:
            baseline = baseline_als(y, lam=params["als_lam"], p=params["als_p"], niter=params["als_niter"])
            y = y - baseline
            log.append(f"ALS baseline (λ={params['als_lam']:.0e}, p={params['als_p']})")
        elif "Linear" in method:
            baseline = linear_baseline(y)
            y = y - baseline
            log.append("Linear baseline removed")
        elif "Polynomial" in method:
            baseline = polynomial_baseline(wn, y, deg=params.get("poly_deg", 3))
            y = y - baseline
            log.append(f"Polynomial baseline (deg={params.get('poly_deg', 3)})")
    
    # 4. Normalization (after baseline!)
    if params.get("normalize", True) and params.get("norm_method") != "None":
        method = params["norm_method"]
        y = normalize_spectrum(y, method=method.lower().split()[0] if " " in method else method.lower())
        log.append(f"Normalized ({method})")
    
    return wn, y, " | ".join(log) if log else "No processing applied"

def create_spectrum_plot(spectra_data, selected_names, use_processed=True, 
                         peak_data=None, primary_name=None, title="FTIR Spectra"):
    """Improved interactive Plotly figure for FTIR spectra."""
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
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(color=color_palette[idx % len(color_palette)], width=1.6),
            hovertemplate="<b>%{fullData.name}</b><br>Wavenumber: %{x:.1f} cm⁻¹<br>Intensity: %{y:.4f}<extra></extra>"
        ))

    # Add detected peaks
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=30, t=70, b=50)
    )

    # Subtle region annotations
    fig.add_vrect(x0=2800, x1=3100, fillcolor="rgba(255,200,0,0.07)", layer="below", line_width=0)
    fig.add_vrect(x0=1650, x1=1750, fillcolor="rgba(0,200,255,0.07)", layer="below", line_width=0)

    return fig
# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("🧬 FTIR Spectra Processor")
st.markdown(
    "**Interactive tool for environmental chemistry, materials characterization, and analytical workflows.**<br>"
    "Upload your FTIR data → Preprocess → Detect peaks & assign functional groups → Export ready-to-use files."
)

# Initialize session state
if "spectra_data" not in st.session_state:
    st.session_state.spectra_data = {}
if "last_peak_data" not in st.session_state:
    st.session_state.last_peak_data = None
if "processing_log" not in st.session_state:
    st.session_state.processing_log = {}

# ============================================================
# SIDEBAR - Quick Controls & Info
# ============================================================
with st.sidebar:
    st.header("⚙️ Quick Controls")
    
    if st.button("🧪 Load Demo Example Spectrum", use_container_width=True, type="secondary"):
        wn, y = generate_example_spectrum()
        name = "Demo_FTIR_Spectrum"
        st.session_state.spectra_data[name] = {
            "wn": wn,
            "raw_int": y,
            "processed_int": None,
            "processed_wn": None,
            "meta": {"source": "demo", "loaded_at": datetime.now().isoformat()}
        }
        st.success("Demo spectrum loaded!")
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
    st.metric("Fully Processed", n_processed)
    
    st.divider()
    st.caption("**Typical mid-IR range:** 4000–400 cm⁻¹")
    st.caption("**Recommended order:** Crop → Smooth → Baseline → Normalize → Peak pick")
    st.caption("Made for researchers • Works offline after install")

# ============================================================
# TABS
# ============================================================
tab_upload, tab_process, tab_peaks, tab_export = st.tabs([
    "📁 Upload & Visualize", 
    "⚙️ Preprocessing", 
    "🔍 Peak Analysis & ID", 
    "📤 Export"
])

# ============================================================
# TAB 1: UPLOAD & VISUALIZE
# ============================================================
with tab_upload:
    st.header("Upload FTIR Data Files")
    
    uploaded_files = st.file_uploader(
        "Drag & drop CSV, TXT or DAT files here",
        type=["csv", "txt", "dat"],
        accept_multiple_files=True,
        help="Each file should contain at least two columns: wavenumber (cm⁻¹) and intensity (absorbance or %T). Headers are automatically skipped."
    )
    
    if uploaded_files:
        new_files = 0
        for f in uploaded_files:
            if f.name not in st.session_state.spectra_data:
                df = load_ftir_file(f)
                if df is not None:
                    st.session_state.spectra_data[f.name] = {
                        "wn": df["wavenumber"].values,
                        "raw_int": df["intensity"].values,
                        "processed_int": None,
                        "processed_wn": None,
                        "meta": {"source": "upload", "filename": f.name, "loaded_at": datetime.now().isoformat()}
                    }
                    new_files += 1
        if new_files > 0:
            st.success(f"Successfully loaded {new_files} new spectrum file(s).")
    
    # Show current loaded spectra
    if st.session_state.spectra_data:
        st.subheader("Loaded Spectra")
        names = list(st.session_state.spectra_data.keys())
        
        # Selection for visualization
        selected_for_viz = st.multiselect(
            "Select spectra to display (overlay)",
            options=names,
            default=names[:min(6, len(names))],
            help="You can select up to ~8–10 spectra comfortably for overlay."
        )
        
        if selected_for_viz:
            # Quick raw plot
            fig = create_spectrum_plot(
                st.session_state.spectra_data, 
                selected_for_viz, 
                use_processed=False
            )
            st.plotly_chart(fig, use_container_width=True, key="raw_overview")
            
            # Mini data table for first selected
            first = selected_for_viz[0]
            d = st.session_state.spectra_data[first]
            with st.expander(f"📋 Raw data preview — {first} ({len(d['wn'])} points)", expanded=False):
                preview_df = pd.DataFrame({
                    "wavenumber_cm-1": d["wn"][:10],
                    "intensity": d["raw_int"][:10]
                })
                st.dataframe(preview_df, hide_index=True, use_container_width=True)
                st.caption(f"Total points: {len(d['wn'])} | Range: {d['wn'].max():.0f} – {d['wn'].min():.0f} cm⁻¹")
    else:
        st.info("👆 Upload your FTIR files or load the demo spectrum from the sidebar to get started.")

# ============================================================
# TAB 2: PREPROCESSING
# ============================================================
with tab_process:
    st.header("Preprocessing Pipeline")
    
    if not st.session_state.spectra_data:
        st.warning("Please upload or load spectra first (Tab 1).")
    else:
        names = list(st.session_state.spectra_data.keys())
        selected = st.multiselect(
            "Select spectra to preprocess",
            options=names,
            default=names,
            key="process_select"
        )
        
        if selected:
            st.subheader("Pipeline Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**1. Crop Range**")
                crop = st.checkbox("Enable cropping", value=True, key="crop_enable")
                min_wn = st.number_input("Min (cm⁻¹)", 200, 4000, 400, 50, disabled=not crop)
                max_wn = st.number_input("Max (cm⁻¹)", 200, 4000, 4000, 50, disabled=not crop)
            
            with col2:
                st.markdown("**2. Smoothing**")
                smooth = st.checkbox("Savitzky-Golay filter", value=True, key="smooth_enable")
                sg_window = st.slider("Window length (odd)", 5, 51, 11, 2, disabled=not smooth)
                sg_poly = st.slider("Polynomial order", 1, 5, 2, 1, disabled=not smooth)
            
            with col3:
                st.markdown("**3. Baseline Correction**")
                baseline = st.checkbox("Enable baseline correction", value=True, key="baseline_enable")
                baseline_method = st.selectbox(
                    "Method",
                    ["ALS (recommended for FTIR)", "Linear (simple detrend)", "Polynomial (deg 3)", "None"],
                    index=0,
                    disabled=not baseline
                )
                
                if "ALS" in baseline_method:
                    als_lam = st.number_input("λ (smoothness)", 1e3, 1e8, 1e5, 1e4, format="%.0e")
                    als_p = st.slider("p (asymmetry)", 0.001, 0.2, 0.01, 0.001, format="%.3f")
                    als_niter = st.slider("Iterations", 5, 25, 10)
                else:
                    als_lam, als_p, als_niter = 1e5, 0.01, 10
                
                poly_deg = 3
                if "Polynomial" in baseline_method:
                    poly_deg = st.slider("Polynomial degree", 1, 6, 3)
            
            st.markdown("**4. Normalization** (applied after baseline)")
            norm = st.checkbox("Enable normalization", value=True, key="norm_enable")
            norm_method = st.selectbox(
                "Method",
                ["Min-Max (0–1)", "SNV (Standard Normal Variate)", "Vector (L2 norm)", "None"],
                index=0,
                disabled=not norm
            )
            
            # Apply button
            if st.button("🚀 Apply Preprocessing to Selected Spectra", type="primary", use_container_width=True):
                params = {
                    "crop": crop,
                    "min_wn": min_wn,
                    "max_wn": max_wn,
                    "smooth": smooth,
                    "sg_window": sg_window,
                    "sg_poly": sg_poly,
                    "baseline": baseline,
                    "baseline_method": baseline_method,
                    "als_lam": als_lam,
                    "als_p": als_p,
                    "als_niter": als_niter,
                    "poly_deg": poly_deg,
                    "normalize": norm,
                    "norm_method": norm_method
                }
                
                success_count = 0
                for name in selected:
                    d = st.session_state.spectra_data[name]
                    new_wn, new_y, log_msg = apply_processing_pipeline(d["wn"], d["raw_int"], params)
                    
                    if "ERROR" not in log_msg:
                        d["processed_wn"] = new_wn
                        d["processed_int"] = new_y
                        st.session_state.processing_log[name] = log_msg
                        success_count += 1
                    else:
                        st.error(f"{name}: {log_msg}")
                
                if success_count > 0:
                    st.success(f"✅ Preprocessing completed for {success_count} spectrum/spectra!")
                    st.balloons()
            
            # Show processed plot if any processed data exists
            processed_selected = [n for n in selected if st.session_state.spectra_data[n].get("processed_int") is not None]
            if processed_selected:
                st.divider()
                st.subheader("Processed Spectra Preview")
                fig = create_spectrum_plot(
                    st.session_state.spectra_data,
                    processed_selected,
                    use_processed=True
                )
                st.plotly_chart(fig, use_container_width=True, key="processed_preview")
                
                # Show processing log for one
                with st.expander("📜 Processing log for selected spectra"):
                    for name in processed_selected[:3]:
                        if name in st.session_state.processing_log:
                            st.code(f"{name}:\n{st.session_state.processing_log[name]}", language="text")

# ============================================================
# TAB 3: PEAK ANALYSIS & FUNCTIONAL GROUP ID
# ============================================================
with tab_peaks:
    st.header("Peak Detection & Functional Group Assignment")
    
    if not st.session_state.spectra_data:
        st.warning("Load spectra first.")
    else:
        names = list(st.session_state.spectra_data.keys())
        processed_names = [n for n in names if st.session_state.spectra_data[n].get("processed_int") is not None]
        
        if not processed_names:
            st.info("👉 Go to the **Preprocessing** tab and apply processing first. Peak detection works best on processed data.")
        else:
            primary = st.selectbox(
                "Choose primary spectrum for peak detection",
                options=processed_names,
                index=0
            )
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                height = st.slider("Minimum peak height", 0.02, 1.0, 0.12, 0.01, 
                                   help="After normalization this is usually 0.05–0.3")
            with col_b:
                prominence = st.slider("Prominence", 0.01, 0.8, 0.06, 0.01)
            with col_c:
                distance = st.slider("Minimum distance (data points)", 5, 80, 15, 1)

            show_derivative = st.checkbox("Show Second Derivative (helps resolve overlapping peaks)", value=False)
            
            if st.button("🔎 Detect Peaks & Assign Groups", type="primary"):
                d = st.session_state.spectra_data[primary]
                wn = d.get("processed_wn", d["wn"])
                y = d["processed_int"]
                
                peaks_idx, props = find_peaks(
                    y,
                    height=height,
                    prominence=prominence,
                    distance=distance
                )
                
                if len(peaks_idx) == 0:
                    st.warning("No peaks detected with current settings. Try lowering height or prominence.")
                else:
                    peak_wn = wn[peaks_idx]
                    peak_int = y[peaks_idx]
                    
                    # Calculate approximate peak area (using a window of ±15 points)
                    areas = []
                    for idx in peaks_idx:
                        start = max(0, idx - 15)
                        end = min(len(y), idx + 16)
                        area = np.trapz(y[start:end], wn[start:end])
                        areas.append(round(area, 4))
                    
                    # Build assignment table
                    assignments = [assign_functional_groups(p) for p in peak_wn]
                    
                    peak_df = pd.DataFrame({
                        "Wavenumber (cm⁻¹)": np.round(peak_wn, 1),
                        "Intensity": np.round(peak_int, 4),
                        "Area": areas,
                        "Possible Functional Groups": assignments
                    }).sort_values("Wavenumber (cm⁻¹)", ascending=False)
                    
                    st.session_state.last_peak_data = {
                        "name": primary,
                        "peak_wn": peak_wn,
                        "peak_int": peak_int,
                        "df": peak_df
                    }
                    
                    st.success(f"Detected {len(peaks_idx)} peaks in **{primary}**")
            
            # Display results if available
            if st.session_state.last_peak_data and st.session_state.last_peak_data["name"] == primary:
                pk = st.session_state.last_peak_data
                st.subheader(f"Detected Peaks — {primary}")
                
                st.dataframe(
                    pk["df"],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Possible Functional Groups": st.column_config.TextColumn(width="large")
                    }
                )
                
                # Annotated plot
                st.subheader("Annotated Spectrum with Peaks")
                fig = create_spectrum_plot(
                    st.session_state.spectra_data,
                    [primary],
                    use_processed=True,
                    peak_data=pk,
                    primary_name=primary
                )
                st.plotly_chart(fig, use_container_width=True, key="peaks_annotated")
                
                # Second Derivative (if enabled)
                if show_derivative:
                    d = st.session_state.spectra_data[primary]
                    wn = d.get("processed_wn", d["wn"])
                    y = d["processed_int"]
                    
                    try:
                        y_deriv2 = savgol_filter(y, window_length=11, polyorder=3, deriv=2, mode='nearest')
                        
                        fig_deriv = go.Figure()
                        fig_deriv.add_trace(go.Scatter(
                            x=wn, y=y_deriv2, mode='lines',
                            line=dict(color='#00B4D8', width=1.5),
                            name='Second Derivative'
                        ))
                        
                        # Mark the same peaks on derivative
                        if 'peak_wn' in pk:
                            fig_deriv.add_trace(go.Scatter(
                                x=pk['peak_wn'], y=[y_deriv2[np.argmin(np.abs(wn - w))] for w in pk['peak_wn']],
                                mode='markers',
                                marker=dict(size=8, color='#FF2D55', symbol='diamond'),
                                name='Detected Peaks'
                            ))
                        
                        fig_deriv.update_layout(
                            title="Second Derivative Spectrum",
                            xaxis_title="Wavenumber (cm⁻¹)",
                            yaxis_title="Second Derivative",
                            xaxis=dict(autorange="reversed"),
                            template="plotly_white",
                            height=450,
                            showlegend=False
                        )
                        st.plotly_chart(fig_deriv, use_container_width=True, key="second_derivative")
                        st.caption("Second derivative helps resolve overlapping peaks and shoulders.")
                    except Exception as e:
                        st.warning(f"Could not compute second derivative: {e}")

                st.caption("💡 Tip: Peaks are labeled with their wavenumber. Hover for exact intensity. "
                           "Multiple groups may match — use your knowledge of sample chemistry to disambiguate.")

                # === Online Database Cross-Reference Section ===
                with st.expander("🔍 Cross-Reference with Online FTIR / Mineral Databases", expanded=False):
                    st.markdown("Use these databases to verify and expand your peak assignments:")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.link_button("RRUFF (Minerals & Inorganics)", "https://rruff.info", help="Best for minerals and metal oxides")
                        st.link_button("NIST Chemistry WebBook", "https://webbook.nist.gov/chemistry", help="IR spectra search by name/formula")
                        st.link_button("SDBS (Japan)", "https://sdbs.db.aist.go.jp", help="Good spectral database")
                        st.link_button("Open Specy", "https://www.openspecy.org", help="Community-driven Raman + IR library")

                    with col2:
                        st.link_button("IRUG Spectral Database", "https://www.irug.org", help="High-quality reference spectra")
                        st.link_button("PubChem Spectra", "https://pubchem.ncbi.nlm.nih.gov", help="Search compounds + spectra")
                        st.link_button("Mineral Spectroscopy", "https://www.mineralspec.org", help="Mineral IR reference")
                        st.link_button("Thermo / PerkinElmer Libraries (Commercial)", "https://www.thermofisher.com", help="Commercial libraries (if accessible)")

                    st.caption("Tip: For minerals and metal oxides, start with **RRUFF**. For organics, use **NIST** or **SDBS**.")

# ============================================================
# TAB 4: EXPORT
# ============================================================
with tab_export:
    st.header("Export Processed Data & Results")
    
    if not st.session_state.spectra_data:
        st.warning("No data to export yet.")
    else:
        processed_available = [n for n in st.session_state.spectra_data 
                               if st.session_state.spectra_data[n].get("processed_int") is not None]
        
        if not processed_available:
            st.info("Process some spectra in the Preprocessing tab first.")
        else:
            st.subheader("Batch Export (ZIP)")
            
            to_export = st.multiselect(
                "Select processed spectra to include in ZIP",
                options=processed_available,
                default=processed_available
            )
            
            if st.button("📦 Create ZIP of Processed CSVs", disabled=len(to_export)==0):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name in to_export:
                        d = st.session_state.spectra_data[name]
                        df_out = pd.DataFrame({
                            "wavenumber_cm-1": d.get("processed_wn", d["wn"]),
                            "processed_intensity": d["processed_int"]
                        })
                        csv_str = df_out.to_csv(index=False)
                        safe_name = name.replace(" ", "_").replace(".csv", "") + "_processed.csv"
                        zf.writestr(safe_name, csv_str)
                
                zip_buffer.seek(0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label=f"⬇️ Download FTIR_Processed_{timestamp}.zip",
                    data=zip_buffer.getvalue(),
                    file_name=f"FTIR_Processed_{timestamp}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            st.divider()
            
            # Single spectrum quick download + peak table
            st.subheader("Quick Downloads")
            single_name = st.selectbox("Choose one spectrum", options=processed_available)
            
            col_x, col_y = st.columns(2)
            with col_x:
                if st.button(f"Download {single_name} (CSV)"):
                    d = st.session_state.spectra_data[single_name]
                    df_out = pd.DataFrame({
                        "wavenumber_cm-1": d.get("processed_wn", d["wn"]),
                        "processed_intensity": d["processed_int"]
                    })
                    st.download_button(
                        "⬇️ Download CSV",
                        data=df_out.to_csv(index=False),
                        file_name=f"{single_name}_processed.csv",
                        mime="text/csv"
                    )
            
            with col_y:
                if (st.session_state.last_peak_data and 
                    st.session_state.last_peak_data["name"] == single_name):
                    pk = st.session_state.last_peak_data
                    st.download_button(
                        "⬇️ Download Peak Table (CSV)",
                        data=pk["df"].to_csv(index=False),
                        file_name=f"{single_name}_peaks.csv",
                        mime="text/csv"
                    )
                else:
                    st.caption("Detect peaks in Tab 3 to enable peak table download.")

# ============================================================
# FOOTER / HELP
# ============================================================
st.divider()
with st.expander("📖 How to use this app (quick guide)", expanded=False):
    st.markdown("""
    1. **Upload** your FTIR CSV/TXT files (or load the demo) in the first tab.
    2. Go to **Preprocessing** → adjust parameters → click **Apply Preprocessing**.
    3. Switch to **Peak Analysis** → pick a spectrum → adjust detection sliders → **Detect Peaks**.
    4. Review the functional group suggestions (they are conservative — combine with your sample knowledge).
    5. **Export** processed spectra + peak lists for your reports/thesis.
    
    **Pro tips:**
    - ALS baseline is usually best for real FTIR data.
    - After normalization, peak height ~0.1–0.4 works well for most samples.
    - Always visually inspect the processed spectrum before trusting peak assignments.
    - For ATR spectra, baseline correction is especially important.
    """)

st.caption("Built with ❤️ for analytical chemistry workflows • Streamlit + Plotly + SciPy • June 2026")