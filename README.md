# MELVIS — MELt VIScosity

**A volcanic melt viscosity platform**

[![DOI](https://img.shields.io/badge/preprint%20DOI-10.31223%2FX5F806-blue)](https://doi.org/10.31223/X5F806)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20460531.svg)](https://doi.org/10.5281/zenodo.20460531)

Developed by the [GLASS laboratory](https://www.danilodigenova.org/glass-laboratory/), CNR-ISSMC (now CNR-GEO), Rome, Italy  
ERC Consolidator Grant NANOVOLC (grant 101044772)  
**Principal Investigator:** Danilo Di Genova — [danilo.digenova@cnr.it](mailto:danilo.digenova@cnr.it)

> **This repository is frozen at MELVIS v1.2.0**, the first public release and the version described in
> Di Genova, D. (in review), *Computers & Geosciences*; preprint on EarthArXiv,
> [doi:10.31223/X5F806](https://doi.org/10.31223/X5F806).
> The code here stays open source under the MIT licence and is archived on Zenodo
> ([doi:10.5281/zenodo.20460531](https://doi.org/10.5281/zenodo.20460531)).
> Later versions of MELVIS (v1.3 and above: the life of a magma, the world map of volcanic melts, ascent modelling,
> fitting of your own data and more) run as a free web service at
> **https://melvis.streamlit.app** and are not distributed as source code.
> Issues and pull requests are no longer followed here: write to [melvis@danilodigenova.org](mailto:melvis@danilodigenova.org).

---

## 🌐 Online access

MELVIS v1.2.0, the version in this repository, runs at:  
**https://annviscositycalculator.streamlit.app**  
The latest version of MELVIS runs at **https://melvis.streamlit.app**

No installation required to use the online version.

---

## 💻 Local installation

Follow these steps to run MELVIS on your own machine.

### Requirements

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.11

### Step 1 — Create a conda environment

```bash
conda create -n melvis python=3.11
conda activate melvis
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Place the ANN model file

Make sure `model.tflite` (the trained ANN in TensorFlow Lite format) is in the **same directory** as `app_online.py`:

```
MELVIS_v1/
├── app_online.py
├── model.tflite
├── requirements.txt
├── Example_compositions.csv
└── README.md
```

### Step 4 — Launch the app

```bash
cd /path/to/MELVIS_v1
streamlit run app_online.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📋 Modules

| Module | Description |
|--------|-------------|
| 🔥 Viscosity Calculator | Multi-sample ANN-based viscosity calculation with TAS diagram and Excel export |
| 💧 Anhydrous and Hydrous Modelling | Tg(H₂O) and fragility fitting for a single composition |
| 🌋 Specific Composition Models | MYEGA models calibrated on specific volcanic compositions |

---

## 📥 Input format

Upload a CSV file with the following columns:

```
Sample, SiO2, TiO2, Al2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, Cr2O3, Fe2O3, H2O, Reference
```

- Missing oxide columns are set to 0 automatically
- Iron redistribution is handled automatically:
  - Only FeO present → split 50/50 between FeO and Fe₂O₃
  - Only Fe₂O₃ present → split 50/50 between Fe₂O₃ and FeO
  - Both present → no change
- Compositions are normalized so that anhydrous oxides sum to (100 − H₂O wt%), preserving the H₂O content

An example file (`Example_compositions.csv`) is included.

---

## 📖 How to cite

If you use MELVIS in your research, please cite:

> Di Genova, D. (2026). *MELVIS: An open-source web application for computing and modelling the viscosity of volcanic melts*. EarthArXiv (preprint). https://doi.org/10.31223/X5F806

The code of this repository (v1.2.0) is archived on Zenodo:

> Di Genova, D. (2026). *MELVIS — MELt VIScosity: a volcanic melt viscosity platform* (v1.2.0). Zenodo. https://doi.org/10.5281/zenodo.20460531

and the relevant model references for each module used (see the sidebar of the app for details).

### Key references

| Model | Reference |
|-------|-----------|
| ANN viscosity model | [Langhammer et al. (2022)](https://doi.org/10.1029/2022GC010673), *GGG* |
| Tg(H₂O) fitting | [Langhammer et al. (2021)](https://doi.org/10.1029/2021GC009918), *GGG* |
| MYEGA equation | [Mauro et al. (2009)](https://doi.org/10.1073/pnas.0911705106), *PNAS* |
| Stromboli basalt | [Valdivia et al. (2023)](https://doi.org/10.1007/s00410-023-02024-w), *CMP* |
| Peridotite melt | [Di Genova et al. (2023)](https://doi.org/10.1016/j.chemgeo.2023.121440), *Chem. Geol.* |
| Anhydrous andesite | [Valdivia et al. (2025)](https://doi.org/10.1038/s43247-025-02424-9), *Commun. Earth Environ.* |
| Colli Albani tephriphonolite | [Fanesi et al. (2025)](https://doi.org/10.1016/j.jvolgeores.2025.108276), *JVGR* |
| Metaluminous/peralkaline haplogranite | [Stopponi et al. (2026)](https://doi.org/10.1016/j.chemgeo.2025.123196), *Chem. Geol.* |
| Vesuvio phonotephrite (472 CE) | [Dominijanni et al. (2026)](https://doi.org/10.1016/j.epsl.2025.119714), *EPSL* |
| AMS-B1 trachyte (Campi Flegrei) | Abeykoon et al. (2026), *J. Geophys. Res. Solid Earth* |

---

## 🔬 About GLASS laboratory

The **GLASS laboratory** (Gateway Laboratory of Amorphous and Structured Solids and Melts) was established in 2023 in Rome with funding from the European Research Council (ERC) Consolidator Grant NANOVOLC (grant 101044772). GLASS operates within the National Research Council of Italy (CNR) at the Institute of Science, Technology and Sustainability for Ceramics (ISSMC).

**Research focus:** physicochemical behaviour of silicate melts, magmas, and glasses, with emphasis on volcanic processes, eruption dynamics, and advanced materials design.

🌐 [www.danilodigenova.org/glass-laboratory](https://www.danilodigenova.org/glass-laboratory/)

---

## 📄 License

This software is released under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for the full text.
