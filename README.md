# 🧬 FTIR Spectra Processor

**Interactive Streamlit application for uploading, preprocessing, visualizing, peak detection, and exporting FTIR spectroscopy data.**

Built for researchers in environmental chemistry, analytical chemistry, materials science, and pharmaceutical analysis. Uses **Bokeh** for stable, high-quality interactive visualizations.

---

## ✨ Features

- **Multiple file upload** — Supports CSV, TXT, DAT with automatic parsing
- **Demo mode** — One-click synthetic mid-IR spectrum with realistic peaks for testing
- **Full preprocessing pipeline**:
  - Wavenumber range cropping
  - Savitzky-Golay smoothing
  - Baseline correction (ALS recommended, Linear, Polynomial)
  - Normalization (Min-Max, SNV, Vector)
- **Interactive Bokeh plots**:
  - Overlay multiple spectra
  - Zoom, pan, reset
  - Click legend items to hide/show individual spectra
  - Detected peaks highlighted with labels
- **Peak detection + functional group assignment**:
  - Adjustable parameters (height, prominence, distance)
  - Comprehensive library of common + less common functional groups
  - Automatic suggestions with notes
- **Batch export**:
  - ZIP download of all processed spectra as CSVs
  - Individual spectrum + peak table downloads
- Clean tabbed interface with persistent session state

---

## 📦 Installation

```bash
# 1. Clone or download the files
# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements_ftir_app.txt

# Or install manually:
pip install streamlit pandas numpy scipy bokeh
```

**requirements_ftir_app.txt** should contain:
```
streamlit>=1.28
pandas>=2.0
numpy>=1.26
scipy>=1.11
bokeh>=3.4
```

---

## 🚀 How to Run

```bash
streamlit run ftir_processor_app_bokeh.py
```

The app will open in your default browser (usually at `http://localhost:8501`).

---

## 🖥️ Interface Overview

The app has four main tabs:

### 1. 📁 Upload & Visualize
- Drag & drop multiple FTIR files
- Load demo spectrum from sidebar
- Interactive overlay plot of raw spectra
- Quick data preview

### 2. ⚙️ Preprocessing
- Apply the full processing pipeline with live parameter control
- **ALS baseline correction** is recommended for most real FTIR data
- See processed spectra immediately in the interactive Bokeh plot
- Processing log is saved per spectrum

### 3. 🔍 Peak Analysis
- Select a processed spectrum
- Adjust peak detection parameters
- View detected peaks with **automatic functional group suggestions**
- Annotated plot with peak markers and labels

### 4. 📤 Export
- Download processed spectra as individual CSVs or as a single ZIP
- Export peak detection tables
- Ready for reports, thesis figures, or further analysis (Python/R)

**Sidebar controls**:
- Load demo spectrum
- Clear all data
- Quick metrics (loaded / processed count)

---

## 📚 Functional Group Library

The app includes a comprehensive curated library of FTIR absorption bands covering:

- O-H, N-H, C-H stretches (detailed variations)
- Carbonyl region (esters, ketones, aldehydes, acids, amides I/II)
- C=C, aromatic, fingerprint region
- Sulfur compounds (S=O, S-H, sulfonates)
- Phosphorus compounds
- Halogens (C-F, C-Cl, C-Br)
- Common inorganic ions (carbonates, sulfates, phosphates, nitrates, silicates)
- Aromatic substitution patterns

> **Note**: FTIR interpretation is contextual. Always cross-validate with other techniques (NMR, MS, Raman) and reference standards when possible.

---

## 💡 Tips for Best Results

- Use **ALS baseline correction** for most real-world FTIR data (especially ATR)
- After normalization, peak height values between **0.08 – 0.25** usually work well
- Zoom into the **fingerprint region** (1500–500 cm⁻¹) for more specific identification
- Click legend items in Bokeh plots to toggle individual spectra on/off
- Process spectra first before running peak detection

---

## 📁 Project Structure

```
ftir_processor_app_bokeh.py   # Main Streamlit application (Bokeh version)
requirements_ftir_app.txt     # Python dependencies
README.md                     # This file
```

---

## 🛠️ Extending the App

You can easily add new features:

- More advanced baseline methods
- Second derivative spectra
- PCA / clustering on multiple spectra
- Custom spectral library matching
- Additional export formats (Excel, PDF reports)
- Dark theme or custom styling

The code is modular — the main logic lives in clear functions (`apply_processing_pipeline`, `create_spectrum_plot`, etc.).

---

## 📜 License

Free to use for research and personal purposes.  
Feel free to modify and adapt for your own workflows.

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [Bokeh](https://bokeh.org/)
- [SciPy](https://scipy.org/) / [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/)

Special thanks to the analytical chemistry community for the many excellent FTIR reference tables that informed the functional group library.

---

**Happy peak hunting!** 🧪

If you have suggestions, feature requests, or run into issues, feel free to reach out. Meow~ 

*Last updated: June 2026*
