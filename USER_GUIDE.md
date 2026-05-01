# Viscosity Calculator — User Guide
### Based on Langhammer et al. (2022), *Geochem. Geophys. Geosyst.*

---

## What is this tool?

The Viscosity Calculator predicts the viscosity of silicate melts (magmas) as a function of temperature, using an Artificial Neural Network (ANN) trained on experimental data. It requires no programming skills — just upload a spreadsheet and download the results.

---

## Option A — Use the online web app (recommended, no installation needed)

> *(Link will be provided once the app is deployed online)*

1. Open the link in any browser (Chrome, Firefox, Safari, Edge)
2. Follow the steps in **Section 3** below

---

## Option B — Run locally on Windows (no Python needed)

### Step 1 — Download the tool

Download and unzip the file `ViscosityCalculator.zip`.  
You will get a folder called `ViscosityCalculator\` containing:

```
ViscosityCalculator\
├── ViscosityCalculator.exe   ← double-click this to launch
├── _internal\                ← DO NOT delete or move this folder
└── ...
```

> ⚠️ **Important:** Always keep `ViscosityCalculator.exe` and the `_internal\` folder in the same location. Never move the `.exe` file alone.

### Step 2 — Launch the tool

Double-click `ViscosityCalculator.exe`.  
A black terminal window will open — this is normal, do not close it.  
After a few seconds, a file selection dialog will appear.

### Step 3 — Follow the steps in Section 3 below

---

## Section 3 — How to use the tool (both versions)

### Step 1 — Prepare your input CSV file

Create a CSV file (e.g. in Excel: *File → Save As → CSV*) with the following structure:

| Sample | SiO2 | TiO2 | Al2O3 | FeO | MnO | MgO | CaO | Na2O | K2O | P2O5 | Cr2O3 | Fe2O3 | H2O | Reference |
|--------|------|------|-------|-----|-----|-----|-----|------|-----|------|-------|-------|-----|-----------|
| Sample_A | 48.05 | 0.76 | 17.69 | 6.08 | 0.14 | 3.32 | 9.31 | 3.45 | 7.55 | 0.46 | 0 | 0 | 0 | Giordano 2009 |
| Sample_B | 58.90 | 0.12 | 20.58 | 2.00 | 0.18 | 0.07 | 1.69 | 7.91 | 6.66 | 0.00 | 0 | 0 | 0 | Giordano 2009 |

**Rules:**
- The `Sample` column is **required**
- All oxide columns are in **wt%**
- Missing oxide columns are automatically set to **0**
- The `Reference` and `Cr2O3` columns are **optional**
- Compositions do **not** need to sum to 100% — the tool normalises automatically

**Iron handling:**
- If only **FeO** is given (Fe2O3 = 0): iron is split 50/50 between FeO and Fe2O3
- If only **Fe2O3** is given (FeO = 0): iron is split 50/50 between Fe2O3 and FeO
- If **both** FeO and Fe2O3 are given: no change is applied

---

### Step 2 — Upload your file and calculate

**Web app:** click *Browse files* and select your CSV, then click **▶️ Calculate viscosity**

**Desktop app:** a file dialog will open automatically — navigate to your CSV and select it

The tool will process all samples. A progress bar shows the status.

---

### Step 3 — View the results

After calculation you will see:

- 📈 **Viscosity plot** — MYEGA curves for all samples (temperature in °C vs log₁₀ viscosity)
- 🌡️ **Specific temperatures** (optional) — see Section 4 below

---

### Step 4 — Download the results

Three download buttons will appear:

| File | Contents |
|------|----------|
| `output_viscosity.xlsx` | Sheet 1: full MYEGA curves · Sheet 2: viscosity at specific T (if requested) · Sheet 3: MYEGA calculator with editable Tg and m |
| `chemistry_check.xlsx` | Sheet 1: your original input chemistry · Sheet 2: normalised and iron-redistributed chemistry used for the calculation |
| `viscosity_plot.png` | High-resolution plot (200 dpi) |

---

## Section 4 — Viscosity at specific temperatures (optional)

After the main calculation:

1. Tick the checkbox **"Calculate viscosity at specific temperatures?"**
2. Select the samples you are interested in from the list
3. For each sample, enter the temperatures in °C separated by commas  
   Example: `800, 1000, 1200, 1400`
4. Click **✅ Compute specific temperatures**
5. A table will appear and the results will be included in `output_viscosity.xlsx` (sheet *Viscosity_at_T*)

---

## Section 5 — The MYEGA Calculator sheet

The third sheet in `output_viscosity.xlsx` is an interactive Excel calculator:

- **Yellow cells**: editable — paste your own Tg (K) and fragility index m values
- **Blue cells**: automatically computed by Excel using the MYEGA equation (A = −2.9)
- Temperature range: from Tg (rounded to nearest 25°C) to **1600°C**, step **25°C**
- To make a plot: select the **T (°C)** and **log₁₀(visc)** columns and insert a chart in Excel

---

## Section 6 — Troubleshooting

| Problem | Solution |
|---------|----------|
| The `.exe` crashes immediately | Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and try again |
| A sample is skipped | Check that SiO2 > 0 and the composition is within the model domain |
| Very long calculation time | Normal for large files (>100 samples). The model loads only once |
| Viscosity seems unrealistic | Check your input chemistry — compositions far outside the training domain may produce unreliable results |

---

## Citation

If you use this tool in your research, please cite:

> Langhammer, D., Di Genova, D., & Steinle-Neumann, G. (2022).  
> *Modeling viscosity of volcanic melts with artificial neural networks.*  
> Geochemistry, Geophysics, Geosystems, 23, e2022GC010673.  
> https://doi.org/10.1029/2022GC010673

---

## Contact

For questions or bug reports, please contact: *(your email here)*

