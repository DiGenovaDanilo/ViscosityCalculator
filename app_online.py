"""
app.py  —  ViscosityCalculator  (Streamlit web app)
Langhammer et al. (2022) ANN model
"""

import sys
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import brentq
import streamlit as st
import tensorflow as tf
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import urllib.request
import zipfile

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE_DIR, "model")
sys.path.insert(0, BASE_DIR)

# ── Page config MUST be first Streamlit call ─────────────────────────────────
st.set_page_config(page_title="Viscosity Calculator", page_icon="🌋", layout="wide")

# ── Auto-download model from Zenodo if not present ────────────────────────────
ZENODO_URL = "https://zenodo.org/records/19945909/files/model.zip"

if not os.path.exists(PATH_MODEL):
    with st.spinner("Downloading ANN model from Zenodo (first run only, ~50 MB)..."):
        zip_path = os.path.join(BASE_DIR, "model.zip")
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(BASE_DIR)
        os.remove(zip_path)
    st.success("Model downloaded successfully!")

from wt_to_mol_calc import mol_conv
from myega import myega

OXIDES = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO',
          'Na2O','K2O','P2O5','Cr2O3','Fe2O3','H2O']

# ==============================================================================
# FUNCTIONS
# ==============================================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(PATH_MODEL)

def redistribute_iron(wt):
    wt = wt.copy()
    feo = wt[3]; fe2o3 = wt[11]
    flag = 'unchanged'
    if feo > 0 and fe2o3 == 0:
        wt[3] = feo/2; wt[11] = feo*1.11/2; flag = 'FeO->split'
    elif fe2o3 > 0 and feo == 0:
        wt[11] = fe2o3/2; wt[3] = fe2o3/1.11/2; flag = 'Fe2O3->split'
    return wt, flag

def normalize_to_100(wt):
    s = wt.sum()
    return wt/s*100.0 if s > 0 else wt

def myega_eq(T, Tg, m):
    A = -2.9
    return A + (Tg/T)*(12-A)*np.exp(((m/(12-A))-1)*((Tg/T)-1))

def visc_calc_fast(inp, si, model):
    eta_goal = ([0.0,0.5,1.0,1.5,2.0,9.5,10,10.5,11.0,11.5] if si<=60
                else [2,2.5,3,3.5,4,4.5,9.5,10,10.5,11.0,11.5])
    t_max = 2023.0
    t_top_start = (3000.0/t_max-0.602847523)/np.sqrt(0.031535353)
    t_bot_start = (300.0 /t_max-0.602847523)/np.sqrt(0.031535353)
    t_mid_start = (t_top_start+t_bot_start)/2
    comp = np.c_[t_mid_start,inp[0],inp[1],inp[2],inp[3],
                 inp[4],inp[5],inp[6],inp[7],inp[8],
                 inp[9],inp[11],inp[12],inp[13],inp[14]]
    eta_mid_start = model(comp)
    t_goal, eta = [], []
    for goal in eta_goal:
        t_top=t_top_start; t_bot=t_bot_start
        t_mid=t_mid_start; eta_mid=eta_mid_start
        err=eta_mid-goal
        while np.absolute(err)>1e-3:
            if eta_mid<goal: t_top=t_mid
            else:            t_bot=t_mid
            t_mid=     (t_bot+t_top)/2
            comp[0][0]=t_mid
            eta_mid=model(comp)
            eta_mid=np.concatenate(eta_mid)
            err=eta_mid-goal
        t_goal.append((t_mid*np.sqrt(0.031535353)+0.602847523)*t_max)
        eta.append(float(np.array(eta_mid).flatten()[0]))
    return np.array(t_goal,dtype=float), np.array(eta,dtype=float)

def write_sheet(ws, df_in, hdr_color='1F4E79'):
    hdr_font  = Font(name='Arial',bold=True,color='FFFFFF',size=10)
    hdr_fill  = PatternFill('solid',start_color=hdr_color)
    alt_fill  = PatternFill('solid',start_color='D6E4F0')
    norm_fill = PatternFill('solid',start_color='FFFFFF')
    thin      = Side(style='thin',color='AAAAAA')
    brd       = Border(left=thin,right=thin,top=thin,bottom=thin)
    ctr       = Alignment(horizontal='center',vertical='center')
    cols = list(df_in.columns)
    for c,col in enumerate(cols,1):
        cell=ws.cell(row=1,column=c,value=col)
        cell.font=hdr_font; cell.fill=hdr_fill
        cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
        cell.border=brd
    ws.row_dimensions[1].height=25
    for r,row_data in enumerate(df_in.itertuples(index=False),2):
        fill=alt_fill if r%2==0 else norm_fill
        for c,val in enumerate(row_data,1):
            cell=ws.cell(row=r,column=c,value=val if pd.notna(val) else '')
            cell.border=brd; cell.fill=fill; cell.alignment=ctr
            if isinstance(val,float): cell.number_format='0.000'
    for c,col in enumerate(cols,1):
        ws.column_dimensions[get_column_letter(c)].width=max(len(str(col)),8)+3
    ws.freeze_panes='B2'

# ==============================================================================
# SESSION STATE INIT
# ==============================================================================
for key in ['calc_done','all_curves','rows_recalc','tg_m_dict',
            'skipped','fig','df_input']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'calc_done' not in st.session_state:
    st.session_state['calc_done'] = False

# ==============================================================================
# UI
# ==============================================================================
st.title("🌋 Viscosity Calculator")
st.markdown("""
Calculates silicate melt viscosity using the **Langhammer et al. (2022)** ANN model.  
Upload a CSV file with your compositions and download the results.
""")

with st.sidebar:
    st.header("📋 Input format")
    st.markdown("""
    Your CSV must have a **Sample** column and oxide columns  
    (missing ones are set to 0 automatically):
    
    `SiO2, TiO2, Al2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, Cr2O3, Fe2O3, H2O`
    
    Optional: `Reference` column.
    
    **Iron redistribution:**
    - Only FeO → split 50/50 between FeO and Fe₂O₃
    - Only Fe₂O₃ → split 50/50 between Fe₂O₃ and FeO
    - Both present → no change
    
    **Reference:**  
    Langhammer et al. (2022), *Geochem. Geophys. Geosyst.*  
    [https://doi.org/10.1029/2022GC010673](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GC010673)
    """)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

# Template CSV content
TEMPLATE_CSV = (
    "Sample,SiO2,TiO2,Al2O3,FeO,MnO,MgO,CaO,Na2O,K2O,P2O5,Cr2O3,Fe2O3,H2O,Reference\n"
    "1,78.1,0.1,12.02,1.6,0.04,0.06,0.9,1.0,5.2,0.02,0,0,0,Unknown\n"
    "2,64.04,0.6,19.0,3.2,0.2,2.3,6.0,4.5,1.6,0,0,0,0,Unknown\n"
    "3,53.0,0.88,20.0,4.0,0,3.18,9.1,4.0,1.0,0.2,0,4.0,0,Unknown\n"
    "4,46.1,2.3,13.0,11.2,0.4,11.6,11.13,2.4,1.1,0.6,0,0,0,Unknown\n"
)

st.download_button(
    label="📄 Download CSV template",
    data=TEMPLATE_CSV,
    file_name="Example_compositions.csv",
    mime="text/csv",
    help="Download this template, fill in your compositions and upload it above."
)

if uploaded is None:
    st.info("👆 Upload a CSV file to get started. Download the template above if needed.")
    st.stop()

df = pd.read_csv(uploaded).dropna(how='all').reset_index(drop=True)
for ox in OXIDES:
    if ox not in df.columns:
        df[ox] = 0.0
df[OXIDES] = df[OXIDES].fillna(0.0)
st.success(f"✅ File loaded: **{len(df)} samples** found.")
with st.expander("Preview input data"):
    st.dataframe(df)

# ── Calculate button ──────────────────────────────────────────────────────────
if st.button("▶️ Calculate viscosity", type="primary"):

    model      = load_model()
    all_curves = []
    rows_recalc= []
    skipped    = []
    tg_m_dict  = {}

    progress = st.progress(0, text="Calculating...")
    fig, ax  = plt.subplots(figsize=(12,7))
    colors   = plt.cm.nipy_spectral(np.linspace(0,1,len(df)))

    for idx, row in df.iterrows():
        sname   = row['Sample']
        wt_orig = np.array([row[o] for o in OXIDES], dtype=float)
        wt_fe, fe_flag = redistribute_iron(wt_orig)
        wt_final       = normalize_to_100(wt_fe)

        rec = {'Sample': sname}
        for i,ox in enumerate(OXIDES): rec[ox]=round(wt_final[i],4)
        rec['SUM']=round(wt_final.sum(),4); rec['Fe_treatment']=fe_flag
        if 'Reference' in df.columns: rec['Reference']=row['Reference']
        rows_recalc.append(rec)

        try:
            normalised,_,_ = mol_conv(wt_final)
            t_synth,eta_synth = visc_calc_fast(normalised,wt_final[0],model)
            if not np.all(np.isfinite(t_synth)) or not np.all(np.isfinite(eta_synth)):
                raise ValueError("ANN returned NaN")
            param,_,_,_ = myega(t_synth,eta_synth,np.array([1000.0]))
            Tg=param[0]; m=param[1]
            tg_m_dict[sname]=(Tg,m)
            try: T_max=brentq(myega_eq,Tg,5000.0,args=(Tg,m))
            except: T_max=3000.0
            T_array=np.arange(Tg,T_max+50,50)
            visc_array=myega_eq(T_array,Tg,m)
            ax.plot(T_array-273.15,visc_array,color=colors[idx],
                    linewidth=1.5,label=sname)
            ax.scatter([Tg-273.15],[myega_eq(Tg,Tg,m)],
                       color=colors[idx],marker='D',s=40,zorder=5)
            for i,(T,v) in enumerate(zip(T_array,visc_array)):
                all_curves.append({
                    'Sample':    sname if i==0 else '',
                    'Tg_K':     round(Tg,1)        if i==0 else '',
                    'Tg_C':     round(Tg-273.15,1) if i==0 else '',
                    'm':        round(m,2)          if i==0 else '',
                    'T_K':      round(T,1),
                    'T_C':      round(T-273.15,1),
                    'log10_visc':round(float(v),3),
                })
            all_curves.append({k:'' for k in
                ['Sample','Tg_K','Tg_C','m','T_K','T_C','log10_visc']})
        except Exception as e:
            skipped.append({'Sample':sname,'Error':str(e)})

        progress.progress((idx+1)/len(df),
                          text=f"Processing {idx+1}/{len(df)}: {sname}")

    progress.empty()
    ax.set_xlabel('Temperature (°C)',fontsize=13)
    ax.set_ylabel('log₁₀(Viscosity / Pa·s)',fontsize=13)
    ax.legend(fontsize=6,loc='upper right',ncol=3,framealpha=0.7,handlelength=1.5)
    ax.grid(True,linestyle='--',alpha=0.5)
    plt.tight_layout()

    # Save everything to session state
    st.session_state['calc_done']   = True
    st.session_state['all_curves']  = all_curves
    st.session_state['rows_recalc'] = rows_recalc
    st.session_state['tg_m_dict']   = tg_m_dict
    st.session_state['skipped']     = skipped
    st.session_state['fig']         = fig
    st.session_state['df_input']    = df

# ── Show results (from session state) ────────────────────────────────────────
if st.session_state.get('calc_done'):

    all_curves  = st.session_state['all_curves']
    rows_recalc = st.session_state['rows_recalc']
    tg_m_dict   = st.session_state['tg_m_dict']
    skipped     = st.session_state['skipped']
    fig         = st.session_state['fig']
    df_input    = st.session_state['df_input']

    st.subheader("📈 Viscosity curves")
    st.pyplot(fig)

    # ── Specific temperatures ─────────────────────────────────────────────────
    st.subheader("🌡️ Viscosity at specific temperatures (optional)")
    do_specific = st.checkbox("Calculate viscosity at specific temperatures?")
    rows_specific = []

    if do_specific:
        sample_names = list(tg_m_dict.keys())
        selected     = st.multiselect("Select samples:", sample_names,
                                      default=sample_names)

        if selected:
            st.markdown("**Enter temperatures (°C) for each sample:**")
            temps_per_sample = {}
            for sname in selected:
                t_input = st.text_input(
                    f"Temperatures for **{sname}** (comma-separated):",
                    placeholder="e.g. 800, 1000, 1200, 1400",
                    key=f"temps_{sname}")
                temps_per_sample[sname] = t_input

            if st.button("✅ Compute specific temperatures"):
                all_ok = True
                for sname, t_input in temps_per_sample.items():
                    if not t_input.strip():
                        continue
                    try:
                        temps = [float(t.strip()) for t in t_input.split(',')
                                 if t.strip()]
                        Tg, m = tg_m_dict[sname]
                        for tc in sorted(temps):
                            visc = myega_eq(tc+273.15, Tg, m)
                            rows_specific.append({
                                'Sample':     sname,
                                'T_C':        round(tc, 2),
                                'T_K':        round(tc+273.15, 2),
                                'log10_visc': round(float(visc), 4),
                            })
                    except ValueError:
                        st.error(f"Invalid temperatures for {sname}. "
                                 "Use numbers separated by commas.")
                        all_ok = False
                if rows_specific:
                    st.session_state['rows_specific'] = rows_specific
                    if all_ok:
                        st.success("Done!")

        # Show table if already computed
        if 'rows_specific' in st.session_state and st.session_state['rows_specific']:
            rows_specific = st.session_state['rows_specific']
            st.dataframe(pd.DataFrame(rows_specific))

    # ── Build Excel files ─────────────────────────────────────────────────────
    # output_viscosity.xlsx
    wb = Workbook()
    ws1 = wb.active; ws1.title='Viscosity_Curves'
    write_sheet(ws1,pd.DataFrame(all_curves),hdr_color='1F4E79')
    if rows_specific:
        ws2=wb.create_sheet('Viscosity_at_T')
        write_sheet(ws2,pd.DataFrame(rows_specific),hdr_color='4A235A')
    # MYEGA_Calculator sheet
    ws_calc = wb.create_sheet('MYEGA_Calculator')
    T_STEP_C = 25; T_END_C = 1600
    hdr_font_c  = Font(name='Arial', bold=True, color='FFFFFF', size=10)
    inp_fill_c  = PatternFill('solid', start_color='FFF9C4')
    res_fill_c  = PatternFill('solid', start_color='E3F2FD')
    wht_fill_c  = PatternFill('solid', start_color='FFFFFF')
    drk_fill_c  = PatternFill('solid', start_color='37474F')
    thin_c      = Side(style='thin', color='AAAAAA')
    brd_c       = Border(left=thin_c, right=thin_c, top=thin_c, bottom=thin_c)
    ctr_c       = Alignment(horizontal='center', vertical='center')
    BLOCK_COLS  = 3; GAP_COLS = 1
    sample_list = list(tg_m_dict.keys())
    for s_idx, sname in enumerate(sample_list):
        Tg_val, m_val = tg_m_dict[sname]
        col_start = 1 + s_idx * (BLOCK_COLS + GAP_COLS)
        # Header
        ws_calc.merge_cells(start_row=1, start_column=col_start,
                            end_row=1,   end_column=col_start+BLOCK_COLS-1)
        cell_hdr = ws_calc.cell(row=1, column=col_start, value=sname)
        cell_hdr.font = Font(name='Arial', bold=True, color='FFFFFF', size=10)
        cell_hdr.fill = PatternFill('solid', start_color='1B5E20')
        cell_hdr.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell_hdr.border = brd_c
        ws_calc.row_dimensions[1].height = 22
        # Tg row
        ws_calc.cell(row=2, column=col_start, value='Tg (K)').font = Font(name='Arial', bold=True, size=9)
        tg_cell = ws_calc.cell(row=2, column=col_start+1, value=round(Tg_val,1))
        tg_cell.fill=inp_fill_c; tg_cell.border=brd_c; tg_cell.alignment=ctr_c; tg_cell.number_format='0.0'
        # m row
        ws_calc.cell(row=3, column=col_start, value='m').font = Font(name='Arial', bold=True, size=9)
        m_cell = ws_calc.cell(row=3, column=col_start+1, value=round(m_val,2))
        m_cell.fill=inp_fill_c; m_cell.border=brd_c; m_cell.alignment=ctr_c; m_cell.number_format='0.00'
        tg_addr = f"${get_column_letter(col_start+1)}$2"
        m_addr  = f"${get_column_letter(col_start+1)}$3"
        # Column headers
        DATA_ROW_START = 5
        for ci, label in enumerate(['T (°C)', 'T (K)', 'log10(visc / Pa·s)'], 0):
            cell = ws_calc.cell(row=DATA_ROW_START-1, column=col_start+ci, value=label)
            cell.font = Font(name='Arial', bold=True, color='FFFFFF', size=9)
            cell.fill = drk_fill_c; cell.alignment = ctr_c; cell.border = brd_c
        ws_calc.row_dimensions[DATA_ROW_START-1].height = 28
        # Temperature range from Tg (rounded down to nearest 25) to 1600 C
        Tg_C    = Tg_val - 273.15
        T_first = int(Tg_C // T_STEP_C) * T_STEP_C
        T_values = list(range(T_first, T_END_C + T_STEP_C, T_STEP_C))
        for r_idx, tc in enumerate(T_values):
            data_row = DATA_ROW_START + r_idx
            fill_row = res_fill_c if r_idx % 2 == 0 else wht_fill_c
            tc_col   = get_column_letter(col_start)
            # T in C
            cell_tc = ws_calc.cell(row=data_row, column=col_start, value=tc)
            cell_tc.border=brd_c; cell_tc.alignment=ctr_c; cell_tc.fill=fill_row
            # T in K
            cell_tk = ws_calc.cell(row=data_row, column=col_start+1,
                                   value=f"={tc_col}{data_row}+273.15")
            cell_tk.border=brd_c; cell_tk.alignment=ctr_c
            cell_tk.number_format='0.00'; cell_tk.fill=fill_row
            # log10 visc (MYEGA formula)
            tk_addr2 = f"{get_column_letter(col_start+1)}{data_row}"
            formula  = (f"=-2.9+({tg_addr}/{tk_addr2})*(12-(-2.9))"
                        f"*EXP(({m_addr}/(12-(-2.9))-1)*({tg_addr}/{tk_addr2}-1))")
            cell_v = ws_calc.cell(row=data_row, column=col_start+2, value=formula)
            cell_v.border=brd_c; cell_v.alignment=ctr_c
            cell_v.number_format='0.000'; cell_v.fill=fill_row
        ws_calc.column_dimensions[get_column_letter(col_start)  ].width = 9
        ws_calc.column_dimensions[get_column_letter(col_start+1)].width = 9
        ws_calc.column_dimensions[get_column_letter(col_start+2)].width = 16
        if s_idx < len(sample_list)-1:
            ws_calc.column_dimensions[get_column_letter(col_start+BLOCK_COLS)].width = 2
    ws_calc.freeze_panes = 'A5'
    note_row = DATA_ROW_START + len(T_values) + 2
    ws_calc.cell(row=note_row, column=1,
                 value="Yellow cells = editable (Tg, m). Blue cells = MYEGA formula (A=-2.9). Copy T(C) and log10(visc) columns to plot directly.").font = Font(name='Arial', italic=True, size=8, color='555555')

    buf_visc=io.BytesIO(); wb.save(buf_visc); buf_visc.seek(0)

    # chemistry_check.xlsx
    wb_chem=Workbook()
    ws_in=wb_chem.active; ws_in.title='Input_Chemistry'
    cols_orig=['Sample']+OXIDES
    if 'Reference' in df_input.columns: cols_orig.append('Reference')
    df_orig=df_input[cols_orig].copy()
    df_orig.insert(len(df_orig.columns)-(1 if 'Reference' in df_orig.columns else 0),
                   'SUM_input',df_input[OXIDES].sum(axis=1).round(3))
    write_sheet(ws_in,df_orig,hdr_color='1B5E20')
    ws_rc=wb_chem.create_sheet('Recalculated_Chemistry')
    cols_rec=['Sample']+OXIDES+['SUM','Fe_treatment']
    if 'Reference' in rows_recalc[0]: cols_rec.append('Reference')
    write_sheet(ws_rc,pd.DataFrame(rows_recalc)[cols_rec],hdr_color='1F4E79')
    buf_chem=io.BytesIO(); wb_chem.save(buf_chem); buf_chem.seek(0)

    # plot PNG
    buf_plot=io.BytesIO(); fig.savefig(buf_plot,format='png',dpi=200); buf_plot.seek(0)

    # ── Download buttons ──────────────────────────────────────────────────────
    st.subheader("📥 Download results")
    col1,col2,col3=st.columns(3)
    with col1:
        st.download_button("⬇️ Viscosity Excel",data=buf_visc,
            file_name="output_viscosity.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2:
        st.download_button("⬇️ Chemistry check Excel",data=buf_chem,
            file_name="chemistry_check.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col3:
        st.download_button("⬇️ Plot (PNG)",data=buf_plot,
            file_name="viscosity_plot.png",mime="image/png")

    if skipped:
        st.warning(f"⚠️ {len(skipped)} samples could not be processed:")
        st.dataframe(pd.DataFrame(skipped))
    else:
        st.success(f"✅ All {len(df_input)} samples processed successfully!")
