# 🧬 FTIR Spectra Processor

**Interactive Streamlit application for uploading, preprocessing, visualizing, peak detection, and exporting FTIR spectroscopy data.**

Built for researchers in environmental chemistry, analytical chemistry, materials science, and pharmaceutical analysis. Features a clean tabbed interface with powerful interactive visualizations.

---

## ✨ Features

- **Multiple file upload** — Supports CSV, TXT, DAT with robust automatic parsing
- **Demo mode** — One-click synthetic mid-IR spectrum with realistic peaks for immediate testing
- **Full preprocessing pipeline**:
  - Wavenumber range cropping
  - Savitzky-Golay smoothing
  - Baseline correction (ALS recommended, Linear, Polynomial)
  - Normalization (Min-Max, SNV, Vector)
- **Interactive visualizations**:
  - Overlay multiple spectra with zoom, pan, and hover
  - Detected peaks clearly marked with labels
- **Peak detection + functional group assignment**:
  - Adjustable detection parameters (height, prominence, distance)
  - Comprehensive library of common and important functional groups
  - Automatic suggestions with notes
- **Batch export**:
  - ZIP download of processed spectra as CSVs
  - Individual spectrum and peak table downloads
- Clean, researcher-friendly tabbed interface with persistent session state

---

## 📦 Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install streamlit pandas numpy scipy plotly

# Or use the requirements file:
pip install -r requirements_ftir_app.txt
```

**requirements_ftir_app.txt**:
```
streamlit>=1.28
pandas>=2.0
numpy>=1.26
scipy>=1.11
plotly>=5.18
```

---

## 🚀 How to Run

```bash
streamlit run ftir_processor_app.py
```

The app will open automatically in your default browser (usually at `http://localhost:8501`).

---

## 🖥️ Interface Overview

The app is organized into four main tabs:

### 1. 📁 Upload & Visualize
- Drag and drop multiple FTIR files
- Load the built-in demo spectrum from the sidebar
- Interactive overlay plot of raw spectra
- Quick data preview table

### 2. ⚙️ Preprocessing
- Configure and apply the full processing pipeline
- **ALS baseline correction** is recommended for most real FTIR data
- Immediately preview the processed spectra
- Processing steps are logged per spectrum

### 3. 🔍 Peak Analysis
- Choose a processed spectrum
- Fine-tune peak detection settings
- View detected peaks with **automatic functional group suggestions**
- Annotated plot showing peak positions and labels

### 4. 📤 Export
- Download all processed spectra as a ZIP archive
- Export individual processed CSVs
- Export peak detection tables for reports or further analysis

**Sidebar**:
- Load demo spectrum
- Clear all loaded data
- Quick metrics (number of spectra loaded and processed)

---

## 📚 Functional Group Library

The app includes a curated and reasonably comprehensive library of FTIR absorption bands, covering:

- Detailed O-H, N-H, and C-H stretching regions
- Full carbonyl region (esters, ketones, aldehydes, carboxylic acids, amides)
- C=C, aromatic, and Amide II bands
- Sulfur- and phosphorus-containing groups
- Halogenated compounds
- Common inorganic ions relevant to environmental samples
- Aromatic substitution patterns in the fingerprint region

> **Note**: FTIR peak assignment is contextual. Peak shape, intensity, and the presence of other bands are important. Always cross-reference with reference standards and complementary techniques (NMR, MS, etc.) when possible.

---

## 💡 Tips for Best Results

- Use **ALS baseline correction** for most real-world and ATR-FTIR spectra
- After normalization, good peak detection thresholds are usually in the **0.08 – 0.25** range
- Zoom into the fingerprint region (1500–500 cm⁻¹) for more specific identification
- Process your spectra before running peak detection for cleaner results
- The demo spectrum is useful for exploring all features without uploading data

---

## 📁 Project Structure

```
ftir_processor_app.py          # Main Streamlit application
requirements_ftir_app.txt      # Python dependencies
README.md                      # This file
```

---

## 🛠️ Extending the App

The code is modular and easy to extend. You can add:

- Second derivative spectra
- PCA or clustering across multiple spectra
- Custom spectral library search
- Additional export formats (Excel reports, PDF)
- More advanced baseline methods
- Dark mode or custom theming

Key functions are clearly separated (`apply_processing_pipeline`, peak detection logic, plotting, etc.).

---

## 📜 License

Free for research and personal use. Feel free to modify and adapt for your own needs.

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [SciPy](https://scipy.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)

The functional group library draws from standard analytical chemistry references.

---

**Happy spectroscopy!** 🧪

If you have feature requests, improvements, or run into any issues, feel free to reach out.

*Last updated: June 2026*
