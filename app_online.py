"""
app_online.py  —  ViscosityCalculator  (Streamlit web app)
Langhammer et al. (2022) ANN model — two modes:
  1. Viscosity Calculator (anhydrous, multi-sample)
  2. Anhydrous and Hydrous Modelling (Tg, m and viscosity vs H2O)
"""

import sys, os, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import brentq, minimize
import streamlit as st
import tensorflow as tf
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import urllib.request, zipfile

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE_DIR, "model")
sys.path.insert(0, BASE_DIR)

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(page_title="Viscosity Calculator", page_icon="🌋", layout="wide")

# ── Auto-download model from Zenodo ──────────────────────────────────────────
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

# ── Constants ─────────────────────────────────────────────────────────────────
OXIDES  = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO',
           'Na2O','K2O','P2O5','Cr2O3','Fe2O3','H2O']
A_FIXED = -2.9

TEMPLATE_CSV = (
    "Sample,SiO2,TiO2,Al2O3,FeO,MnO,MgO,CaO,Na2O,K2O,P2O5,Cr2O3,Fe2O3,H2O,Reference\n"
    "1,78.1,0.1,12.02,1.6,0.04,0.06,0.9,1.0,5.2,0.02,0,0,0,Unknown\n"
    "2,64.04,0.6,19.0,3.2,0.2,2.3,6.0,4.5,1.6,0,0,0,0,Unknown\n"
    "3,53.0,0.88,20.0,4.0,0,3.18,9.1,4.0,1.0,0.2,0,4.0,0,Unknown\n"
    "4,46.1,2.3,13.0,11.2,0.4,11.6,11.13,2.4,1.1,0.6,0,0,0,Unknown\n"
)

# ==============================================================================
# SHARED FUNCTIONS
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

def myega_eq(T, Tg, m, A=A_FIXED):
    return A + (Tg/T)*(12-A)*np.exp(((m/(12-A))-1)*((Tg/T)-1))

def visc_calc_fast(inp, si, model):
    eta_goal = ([0.0,0.5,1.0,1.5,2.0,9.5,10,10.5,11.0,11.5] if si<=60
                else [2,2.5,3,3.5,4,4.5,9.5,10,10.5,11.0,11.5])
    t_max = 2023.0
    t_top = (3000.0/t_max-0.602847523)/np.sqrt(0.031535353)
    t_bot = (300.0 /t_max-0.602847523)/np.sqrt(0.031535353)
    t_mid = (t_top+t_bot)/2
    comp = np.c_[t_mid,inp[0],inp[1],inp[2],inp[3],
                 inp[4],inp[5],inp[6],inp[7],inp[8],
                 inp[9],inp[11],inp[12],inp[13],inp[14]]
    eta_mid = model(comp)
    t_goal, eta = [], []
    for goal in eta_goal:
        t_top2=t_top; t_bot2=t_bot; t_mid2=t_mid; eta_mid2=eta_mid
        err=eta_mid2-goal
        while np.absolute(err)>1e-3:
            if eta_mid2<goal: t_top2=t_mid2
            else:             t_bot2=t_mid2
            t_mid2=     (t_bot2+t_top2)/2
            comp[0][0]=t_mid2
            eta_mid2=model(comp)
            eta_mid2=np.concatenate(eta_mid2)
            err=eta_mid2-goal
        t_goal.append((t_mid2*np.sqrt(0.031535353)+0.602847523)*t_max)
        eta.append(float(np.array(eta_mid2).flatten()[0]))
    return np.array(t_goal,dtype=float), np.array(eta,dtype=float)

def write_sheet(ws, df_in, hdr_color='1F4E79'):
    hdr_font = Font(name='Arial',bold=True,color='FFFFFF',size=10)
    hdr_fill = PatternFill('solid',start_color=hdr_color)
    alt_fill = PatternFill('solid',start_color='D6E4F0')
    nrm_fill = PatternFill('solid',start_color='FFFFFF')
    thin     = Side(style='thin',color='AAAAAA')
    brd      = Border(left=thin,right=thin,top=thin,bottom=thin)
    ctr      = Alignment(horizontal='center',vertical='center')
    cols = list(df_in.columns)
    for c,col in enumerate(cols,1):
        cell=ws.cell(row=1,column=c,value=col)
        cell.font=hdr_font; cell.fill=hdr_fill
        cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
        cell.border=brd
    ws.row_dimensions[1].height=25
    for r,row_data in enumerate(df_in.itertuples(index=False),2):
        fill=alt_fill if r%2==0 else nrm_fill
        for c,val in enumerate(row_data,1):
            cell=ws.cell(row=r,column=c,value=val if pd.notna(val) else '')
            cell.border=brd; cell.fill=fill; cell.alignment=ctr
            if isinstance(val,float): cell.number_format='0.000'
    for c,col in enumerate(cols,1):
        ws.column_dimensions[get_column_letter(c)].width=max(len(str(col)),8)+3
    ws.freeze_panes='B2'

# ── Hydrous modelling helpers ─────────────────────────────────────────────────

def get_Tg_m(wt_anhydrous, h2o_wt, model):
    wt = wt_anhydrous.copy()
    wt[12] = h2o_wt
    wt, _ = redistribute_iron(wt)
    wt     = normalize_to_100(wt)
    normalised, _, _ = mol_conv(wt)
    t_synth, eta_synth = visc_calc_fast(normalised, wt[0], model)
    params, _, _, _ = myega(t_synth, eta_synth, np.array([1000.0]))
    return params[0], params[1]   # Tg (K), m

def wt_to_mol_h2o(h2o_wt_pct, wt_anhydrous):
    wt = wt_anhydrous.copy(); wt[12] = h2o_wt_pct
    wt = normalize_to_100(wt)
    _, _, mol_per = mol_conv(wt)
    return mol_per[12]

def tg_model(x_h2o_mol, b, c, d, Tg_d, Tg_H2O=136.0):
    denom = b*(100.0-x_h2o_mol) + x_h2o_mol
    if denom == 0: return Tg_d
    w1 = x_h2o_mol / denom
    w2 = b*(100.0-x_h2o_mol) / denom
    return (w1*Tg_H2O + w2*Tg_d
            + c*w1*w2*(Tg_d-Tg_H2O)
            + d*w1*w2**2*(Tg_d-Tg_H2O))

def m_from_tg(Tg, Tg_d, m_d, A=A_FIXED):
    return m_d + (12-A)*np.log(Tg/Tg_d)

def fit_tg(x_mol_arr, Tg_arr, Tg_d):
    def rmse_tg(params):
        b, c, d = params
        if b <= 0: return 1e10
        try:
            pred = np.array([tg_model(x, b, c, d, Tg_d) for x in x_mol_arr])
            if not np.all(np.isfinite(pred)): return 1e10
            return np.sqrt(np.mean((pred-Tg_arr)**2))
        except: return 1e10
    best_rmse = np.inf
    best_p    = np.array([0.1, 1.5, -2.0])
    for b0 in [0.05,0.1,0.15,0.2,0.3,0.5,1.0,2.0]:
        for c0 in [0.5,1.0,1.5,2.0,3.0]:
            for d0 in [-3.0,-2.0,-1.5,-1.0,-0.5]:
                try:
                    res = minimize(rmse_tg,[b0,c0,d0],method='Nelder-Mead',
                                   options={'maxiter':20000,'xatol':1e-8,'fatol':1e-8})
                    if res.fun < best_rmse and res.x[0]>0:
                        best_rmse=res.fun; best_p=res.x
                except: pass
    return best_p, best_rmse

def make_visc_sheet_hydrous(wb, sheet_name, results, m_func, tg_model_func, hdr_color):
    ws = wb.create_sheet(sheet_name)
    BLOCK=3; GAP=1; T_STEP=25
    for s_idx, r in enumerate(results):
        col_start = 1+s_idx*(BLOCK+GAP)
        Tg_f = tg_model_func(r['h2o_mol'])
        m_v  = m_func(r['h2o_mol'])
        ws.merge_cells(start_row=1,start_column=col_start,end_row=1,end_column=col_start+BLOCK-1)
        hdr = ws.cell(row=1,column=col_start,
            value='{:.1f} wt% H2O | Tg={:.1f}C | m={:.2f}'.format(r['h2o_wt'],Tg_f-273.15,m_v))
        thin=Side(style='thin',color='AAAAAA')
        brd=Border(left=thin,right=thin,top=thin,bottom=thin)
        ctr=Alignment(horizontal='center',vertical='center')
        hdr.font=Font(name='Arial',bold=True,color='FFFFFF',size=10)
        hdr.fill=PatternFill('solid',start_color=hdr_color)
        hdr.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
        hdr.border=brd
        for ci,label in enumerate(['T (C)','T (K)','log10(visc/Pa.s)'],0):
            cell=ws.cell(row=2,column=col_start+ci,value=label)
            cell.font=Font(name='Arial',bold=True,color='FFFFFF',size=9)
            cell.fill=PatternFill('solid',start_color='37474F')
            cell.alignment=ctr; cell.border=brd
        ws.row_dimensions[2].height=25
        try: T_max=brentq(myega_eq,Tg_f,5000.0,args=(Tg_f,m_v))
        except: T_max=3000.0
        T_first=int((Tg_f-273.15)//T_STEP)*T_STEP
        T_list=list(range(T_first,int(T_max)+T_STEP,T_STEP))
        alt_f=PatternFill('solid',start_color='D6E4F0')
        wht_f=PatternFill('solid',start_color='FFFFFF')
        for ri,tc in enumerate(T_list):
            row_n=3+ri; tk_val=tc+273.15
            visc_v=round(float(myega_eq(tk_val,Tg_f,m_v)),4)
            fill=alt_f if ri%2==0 else wht_f
            for ci,val in enumerate([float(tc),round(tk_val,2),visc_v],0):
                cell=ws.cell(row=row_n,column=col_start+ci,value=val)
                cell.alignment=ctr; cell.border=brd; cell.fill=fill
                if isinstance(val,float): cell.number_format='0.000'
        for ci in range(BLOCK):
            ws.column_dimensions[get_column_letter(col_start+ci)].width=14
        if s_idx<len(results)-1:
            ws.column_dimensions[get_column_letter(col_start+BLOCK)].width=2
    ws.freeze_panes='A3'

# ==============================================================================
# SESSION STATE
# ==============================================================================
for key in ['calc_done','all_curves','rows_recalc','tg_m_dict','skipped',
            'fig','df_input','rows_specific',
            'hyd_done','hyd_results','hyd_fig','hyd_buf_excel']:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state['calc_done'] is None:
    st.session_state['calc_done'] = False
if st.session_state['hyd_done'] is None:
    st.session_state['hyd_done'] = False

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🌋 Viscosity Calculator")
    mode = st.radio(
        "**Select mode:**",
        [
            "🔷 Viscosity Calculator",
            "💧 Anhydrous and Hydrous Modelling",
        ],
        index=0
    )
    st.divider()
    if mode == "🔷 Viscosity Calculator":
        st.markdown("""
**Input format**

CSV with a **Sample** column and oxide columns in wt%
(missing ones set to 0 automatically):

`SiO2, TiO2, Al2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, Cr2O3, Fe2O3, H2O`

Optional: `Reference` column.

**Iron redistribution:**
- Only FeO → split 50/50 FeO / Fe₂O₃
- Only Fe₂O₃ → split 50/50 Fe₂O₃ / FeO
- Both present → no change
        """)
    else:
        st.markdown("""
**Input format**

CSV with a **single anhydrous composition** (H2O = 0).

Same oxide columns as Mode 1.

The tool will automatically add H₂O at the
requested contents, fit Tg(H₂O) with Eq. 9-10
(Langhammer et al. 2021) and compare three
fragility models:
- m constant = m_dry
- m from Eq. 12
- m from polynomial fit on ANN points
        """)
    st.divider()
    st.markdown("""
**Reference:**  
Langhammer et al. (2022), *GGG*  
[doi:10.1029/2022GC010673](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GC010673)

Langhammer et al. (2021), *GGG*  
[doi:10.1029/2021GC009918](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GC009918)
    """)

# ==============================================================================
# MODE 1 — VISCOSITY CALCULATOR
# ==============================================================================
if mode == "🔷 Viscosity Calculator":

    st.title("🔷 Viscosity Calculator")
    st.markdown("Calculate silicate melt viscosity from composition using the **Langhammer et al. (2022)** ANN model. Upload a CSV with multiple samples and download MYEGA curves.")

    st.download_button("📄 Download CSV template", data=TEMPLATE_CSV,
                       file_name="Example_compositions.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is None:
        st.info("👆 Upload a CSV file to get started.")
        st.stop()

    df = pd.read_csv(uploaded).dropna(how='all').reset_index(drop=True)
    for ox in OXIDES:
        if ox not in df.columns: df[ox] = 0.0
    df[OXIDES] = df[OXIDES].fillna(0.0)
    st.success(f"✅ File loaded: **{len(df)} samples** found.")
    with st.expander("Preview input data"): st.dataframe(df)

    if st.button("▶️ Calculate viscosity", type="primary"):
        model = load_model()
        all_curves=[]; rows_recalc=[]; skipped=[]; tg_m_dict={}
        progress=st.progress(0,text="Calculating...")
        fig,ax=plt.subplots(figsize=(12,7))
        colors=plt.cm.nipy_spectral(np.linspace(0,1,len(df)))

        for idx,row in df.iterrows():
            sname=row['Sample']
            wt_orig=np.array([row[o] for o in OXIDES],dtype=float)
            wt_fe,fe_flag=redistribute_iron(wt_orig)
            wt_final=normalize_to_100(wt_fe)
            rec={'Sample':sname}
            for i,ox in enumerate(OXIDES): rec[ox]=round(wt_final[i],4)
            rec['SUM']=round(wt_final.sum(),4); rec['Fe_treatment']=fe_flag
            if 'Reference' in df.columns: rec['Reference']=row['Reference']
            rows_recalc.append(rec)
            try:
                normalised,_,_=mol_conv(wt_final)
                t_synth,eta_synth=visc_calc_fast(normalised,wt_final[0],model)
                if not np.all(np.isfinite(t_synth)) or not np.all(np.isfinite(eta_synth)):
                    raise ValueError("ANN returned NaN")
                param,_,_,_=myega(t_synth,eta_synth,np.array([1000.0]))
                Tg=param[0]; m=param[1]
                tg_m_dict[sname]=(Tg,m)
                try: T_max=brentq(myega_eq,Tg,5000.0,args=(Tg,m))
                except: T_max=3000.0
                T_array=np.arange(Tg,T_max+50,50)
                visc_array=myega_eq(T_array,Tg,m)
                ax.plot(T_array-273.15,visc_array,color=colors[idx],linewidth=1.5,label=sname)
                ax.scatter([Tg-273.15],[myega_eq(Tg,Tg,m)],color=colors[idx],marker='D',s=40,zorder=5)
                for i,(T,v) in enumerate(zip(T_array,visc_array)):
                    all_curves.append({
                        'Sample':   sname if i==0 else '',
                        'Tg_K':    round(Tg,1)        if i==0 else '',
                        'Tg_C':    round(Tg-273.15,1) if i==0 else '',
                        'm':       round(m,2)          if i==0 else '',
                        'T_K':     round(T,1),
                        'T_C':     round(T-273.15,1),
                        'log10_visc': round(float(v),3),
                    })
                all_curves.append({k:'' for k in ['Sample','Tg_K','Tg_C','m','T_K','T_C','log10_visc']})
            except Exception as e:
                skipped.append({'Sample':sname,'Error':str(e)})
            progress.progress((idx+1)/len(df), text=f"Processing {idx+1}/{len(df)}: {sname}")

        progress.empty()
        ax.set_xlabel('Temperature (°C)',fontsize=13)
        ax.set_ylabel('log₁₀(Viscosity / Pa·s)',fontsize=13)
        ax.legend(fontsize=6,loc='upper right',ncol=3,framealpha=0.7,handlelength=1.5)
        ax.grid(True,linestyle='--',alpha=0.5)
        plt.tight_layout()
        st.session_state.update({'calc_done':True,'all_curves':all_curves,
            'rows_recalc':rows_recalc,'tg_m_dict':tg_m_dict,
            'skipped':skipped,'fig':fig,'df_input':df,'rows_specific':[]})

    if st.session_state['calc_done']:
        all_curves=st.session_state['all_curves']
        rows_recalc=st.session_state['rows_recalc']
        tg_m_dict=st.session_state['tg_m_dict']
        skipped=st.session_state['skipped']
        fig=st.session_state['fig']
        df_input=st.session_state['df_input']

        st.subheader("📈 Viscosity curves")
        st.pyplot(fig)

        st.subheader("🌡️ Viscosity at specific temperatures (optional)")
        do_specific=st.checkbox("Calculate viscosity at specific temperatures?")
        rows_specific=st.session_state.get('rows_specific',[]) or []

        if do_specific:
            sample_names=list(tg_m_dict.keys())
            selected=st.multiselect("Select samples:",sample_names,default=sample_names)
            if selected:
                st.markdown("**Enter temperatures (°C) for each sample:**")
                temps_per_sample={}
                for sname in selected:
                    t_input=st.text_input(f"Temperatures for **{sname}** (comma-separated):",
                                          placeholder="e.g. 800, 1000, 1200, 1400",
                                          key=f"temps_{sname}")
                    temps_per_sample[sname]=t_input
                if st.button("✅ Compute specific temperatures"):
                    rows_specific=[]
                    for sname,t_input in temps_per_sample.items():
                        if not t_input.strip(): continue
                        try:
                            temps=[float(t.strip()) for t in t_input.split(',') if t.strip()]
                            Tg,m=tg_m_dict[sname]
                            for tc in sorted(temps):
                                rows_specific.append({'Sample':sname,'T_C':round(tc,2),
                                    'T_K':round(tc+273.15,2),'log10_visc':round(float(myega_eq(tc+273.15,Tg,m)),4)})
                        except ValueError:
                            st.error(f"Invalid temperatures for {sname}.")
                    st.session_state['rows_specific']=rows_specific
            if st.session_state.get('rows_specific'):
                st.dataframe(pd.DataFrame(st.session_state['rows_specific']))

        # ── Build Excel ───────────────────────────────────────────────────────
        rows_specific=st.session_state.get('rows_specific',[]) or []
        wb=Workbook()
        ws1=wb.active; ws1.title='Viscosity_Curves'
        write_sheet(ws1,pd.DataFrame(all_curves),hdr_color='1F4E79')
        if rows_specific:
            ws2=wb.create_sheet('Viscosity_at_T')
            write_sheet(ws2,pd.DataFrame(rows_specific),hdr_color='4A235A')

        # MYEGA_Calculator sheet
        ws_calc=wb.create_sheet('MYEGA_Calculator')
        T_STEP_C=25; T_END_C=1600
        inp_fill=PatternFill('solid',start_color='FFF9C4')
        res_fill=PatternFill('solid',start_color='E3F2FD')
        wht_fill=PatternFill('solid',start_color='FFFFFF')
        drk_fill=PatternFill('solid',start_color='37474F')
        thin_c=Side(style='thin',color='AAAAAA')
        brd_c=Border(left=thin_c,right=thin_c,top=thin_c,bottom=thin_c)
        ctr_c=Alignment(horizontal='center',vertical='center')
        BCOLS=3; GCOLS=1
        for s_idx,sname in enumerate(tg_m_dict):
            Tg_val,m_val=tg_m_dict[sname]
            cs=1+s_idx*(BCOLS+GCOLS)
            ws_calc.merge_cells(start_row=1,start_column=cs,end_row=1,end_column=cs+BCOLS-1)
            ch=ws_calc.cell(row=1,column=cs,value=sname)
            ch.font=Font(name='Arial',bold=True,color='FFFFFF',size=10)
            ch.fill=PatternFill('solid',start_color='1B5E20')
            ch.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
            ch.border=brd_c; ws_calc.row_dimensions[1].height=22
            ws_calc.cell(row=2,column=cs,value='Tg (K)').font=Font(name='Arial',bold=True,size=9)
            tc2=ws_calc.cell(row=2,column=cs+1,value=round(Tg_val,1))
            tc2.fill=inp_fill; tc2.border=brd_c; tc2.alignment=ctr_c; tc2.number_format='0.0'
            ws_calc.cell(row=3,column=cs,value='m').font=Font(name='Arial',bold=True,size=9)
            mc2=ws_calc.cell(row=3,column=cs+1,value=round(m_val,2))
            mc2.fill=inp_fill; mc2.border=brd_c; mc2.alignment=ctr_c; mc2.number_format='0.00'
            tga=f"${get_column_letter(cs+1)}$2"; ma=f"${get_column_letter(cs+1)}$3"
            DR=5
            for ci,label in enumerate(['T (°C)','T (K)','log10(visc / Pa·s)'],0):
                cell=ws_calc.cell(row=DR-1,column=cs+ci,value=label)
                cell.font=Font(name='Arial',bold=True,color='FFFFFF',size=9)
                cell.fill=drk_fill; cell.alignment=ctr_c; cell.border=brd_c
            ws_calc.row_dimensions[DR-1].height=28
            Tg_C=Tg_val-273.15; T_first=int(Tg_C//T_STEP_C)*T_STEP_C
            T_values=list(range(T_first,T_END_C+T_STEP_C,T_STEP_C))
            for r_idx,tc in enumerate(T_values):
                dr=DR+r_idx; fill=res_fill if r_idx%2==0 else wht_fill
                tcc=get_column_letter(cs)
                ct=ws_calc.cell(row=dr,column=cs,value=tc)
                ct.border=brd_c; ct.alignment=ctr_c; ct.fill=fill
                ctk=ws_calc.cell(row=dr,column=cs+1,value=f"={tcc}{dr}+273.15")
                ctk.border=brd_c; ctk.alignment=ctr_c; ctk.number_format='0.00'; ctk.fill=fill
                tka=f"{get_column_letter(cs+1)}{dr}"
                formula=f"=-2.9+({tga}/{tka})*(12-(-2.9))*EXP(({ma}/(12-(-2.9))-1)*({tga}/{tka}-1))"
                cv=ws_calc.cell(row=dr,column=cs+2,value=formula)
                cv.border=brd_c; cv.alignment=ctr_c; cv.number_format='0.000'; cv.fill=fill
            ws_calc.column_dimensions[get_column_letter(cs)].width=9
            ws_calc.column_dimensions[get_column_letter(cs+1)].width=9
            ws_calc.column_dimensions[get_column_letter(cs+2)].width=16
            if s_idx<len(tg_m_dict)-1:
                ws_calc.column_dimensions[get_column_letter(cs+BCOLS)].width=2
        ws_calc.freeze_panes='A5'

        # Chemistry sheet
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

        buf_visc=io.BytesIO(); wb.save(buf_visc); buf_visc.seek(0)
        buf_chem=io.BytesIO(); wb_chem.save(buf_chem); buf_chem.seek(0)
        buf_plot=io.BytesIO(); fig.savefig(buf_plot,format='png',dpi=200); buf_plot.seek(0)

        st.subheader("📥 Download results")
        c1,c2,c3=st.columns(3)
        with c1: st.download_button("⬇️ Viscosity Excel",data=buf_visc,
            file_name="output_viscosity.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2: st.download_button("⬇️ Chemistry check Excel",data=buf_chem,
            file_name="chemistry_check.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3: st.download_button("⬇️ Plot (PNG)",data=buf_plot,
            file_name="viscosity_plot.png",mime="image/png")

        if skipped:
            st.warning(f"⚠️ {len(skipped)} samples could not be processed:")
            st.dataframe(pd.DataFrame(skipped))
        else:
            st.success(f"✅ All {len(df_input)} samples processed successfully!")

# ==============================================================================
# MODE 2 — ANHYDROUS AND HYDROUS MODELLING
# ==============================================================================
else:

    st.title("💧 Anhydrous and Hydrous Modelling")
    st.markdown("""
Model how **Tg, fragility index m and viscosity** evolve as a function of H₂O content,
using the **Langhammer et al. (2021)** framework (Eq. 9-10, 12).  
Upload a CSV with **one anhydrous composition** (H₂O = 0).
    """)

    with st.expander("📖 How to use this mode — read first", expanded=False):
        st.markdown("""
### What this mode does
This mode takes a single anhydrous melt composition and models how viscosity changes
as a function of dissolved water content, following the physically-motivated approach
of Langhammer et al. (2021).

### Step-by-step
1. **Upload a CSV** with one row containing your anhydrous composition in wt% oxides (H₂O = 0).
2. **Set H₂O contents** — the tool will recalculate the ANN viscosity at each water content
   you specify (e.g. 0, 0.5, 1, 2, 3, 4, 5 wt%). Always include 0.
3. **Click Run** — the ANN calculates Tg and m at each water content. Then:
   - **Tg(H₂O)** is fitted using Eq. 9-10 (Schneider/Gordon-Taylor model, three free parameters b, c, d)
   - **m(H₂O)** is modelled in three ways (see below)
   - **Viscosity curves** are generated for each m model
4. **Download** the Excel file with all results and the plot as PNG.

### Compositional limits
The ANN was trained on compositions with SiO₂ ranging from ~43 to 79 wt% and total
alkalis from 0 to 17 wt%. Results outside this range may be unreliable.
Always verify that your composition falls within the training domain of the model.

### What the diamond markers mean
The ♦ diamond on each viscosity curve marks the **glass transition temperature Tg**,
defined as the temperature at which log₁₀(η) = 12 Pa·s.
        """)

    # ── Input method selector ─────────────────────────────────────────────────
    input_method = st.radio("**How do you want to provide the composition?**",
                            ["📂 Upload CSV file", "⌨️ Type composition manually"],
                            horizontal=True, key="hyd_input_method")

    df_h = None

    if input_method == "📂 Upload CSV file":
        st.download_button("📄 Download CSV template (single composition)", data=TEMPLATE_CSV,
                           file_name="Example_compositions.csv", mime="text/csv")
        uploaded_h = st.file_uploader("Upload anhydrous composition CSV", type=["csv"],
                                       key="hyd_upload")
        if uploaded_h is None:
            st.info("👆 Upload a CSV file with one anhydrous composition.")
            st.stop()
        df_h = pd.read_csv(uploaded_h).dropna(how='all').reset_index(drop=True)
        for ox in OXIDES:
            if ox not in df_h.columns: df_h[ox] = 0.0
        df_h[OXIDES] = df_h[OXIDES].fillna(0.0)
        if len(df_h) == 0:
            st.error("No valid rows found in the CSV.")
            st.stop()
        if len(df_h) > 1:
            st.warning(f"Found {len(df_h)} rows — using only the first row.")
        df_h = df_h.iloc[[0]]

    else:  # Manual input
        st.markdown("**Enter your anhydrous composition in wt% (H₂O = 0):**")
        sample_name_manual = st.text_input("Sample name:", value="My_sample", key="hyd_sname")
        # Arrange oxides in a grid: 4 columns
        OXIDES_NO_H2O = [o for o in OXIDES if o != "H2O"]
        cols_manual = st.columns(4)
        manual_vals = {}
        defaults = {
            "SiO2": 48.05, "TiO2": 0.76, "Al2O3": 17.69, "FeO": 6.08,
            "MnO": 0.14, "MgO": 3.32, "CaO": 9.31, "Na2O": 3.45,
            "K2O": 7.55, "P2O5": 0.46, "Cr2O3": 0.0, "Fe2O3": 0.0,
        }
        for i, ox in enumerate(OXIDES_NO_H2O):
            with cols_manual[i % 4]:
                manual_vals[ox] = st.number_input(
                    ox, min_value=0.0, max_value=100.0,
                    value=float(defaults.get(ox, 0.0)),
                    step=0.01, format="%.3f",
                    key=f"hyd_manual_{ox}")
        manual_vals["H2O"] = 0.0
        total = sum(manual_vals[o] for o in OXIDES_NO_H2O)
        st.caption(f"Sum of oxides (anhydrous): **{total:.2f} wt%** — will be normalised to 100%")
        # Build df_h from manual input
        df_h = pd.DataFrame([{'Sample': sample_name_manual, **manual_vals}])

    st.success(f"✅ Composition ready: **{df_h['Sample'].values[0]}**")
    with st.expander("Preview composition"): st.dataframe(df_h)

    # H2O contents input
    st.subheader("⚙️ Settings")
    h2o_input = st.text_input(
        "H₂O contents to model (wt%, comma-separated):",
        value="0, 0.5, 1, 2, 3, 4, 5",
        help="Always include 0 as the anhydrous reference.")

    try:
        h2o_list = sorted(set([float(x.strip()) for x in h2o_input.split(',') if x.strip()]))
        if 0.0 not in h2o_list: h2o_list = [0.0] + h2o_list
        st.caption(f"Will calculate at: {h2o_list} wt% H₂O")
    except ValueError:
        st.error("Invalid H₂O input. Use numbers separated by commas.")
        st.stop()

    if st.button("▶️ Run hydrous modelling", type="primary"):

        model = load_model()
        row0 = df_h.iloc[0]
        sname = row0['Sample']
        wt_orig = np.array([row0[o] for o in OXIDES], dtype=float)
        wt_orig[12] = 0.0  # ensure anhydrous
        wt_dry, _ = redistribute_iron(wt_orig)
        wt_dry = normalize_to_100(wt_dry)

        results = []
        progress = st.progress(0, text="Calculating ANN points...")

        for i, h2o in enumerate(h2o_list):
            try:
                Tg, m = get_Tg_m(wt_dry, h2o, model)
                x_mol = wt_to_mol_h2o(h2o, wt_dry)
                results.append({'h2o_wt': h2o, 'h2o_mol': x_mol, 'Tg': Tg, 'm': m})
            except Exception as e:
                st.warning(f"H₂O = {h2o} wt% failed: {e}")
            progress.progress((i+1)/len(h2o_list), text=f"Calculating H₂O = {h2o} wt%...")
        progress.empty()

        if len(results) < 2:
            st.error("Not enough points calculated. Check your composition.")
            st.stop()

        Tg_d = results[0]['Tg']
        m_d  = results[0]['m']
        x_mol_arr = np.array([r['h2o_mol'] for r in results])
        Tg_arr    = np.array([r['Tg']     for r in results])
        m_arr     = np.array([r['m']      for r in results])

        # Fit Tg(x_H2O) — Eq. 9-10
        with st.spinner("Fitting Tg(H₂O) with Eq. 9-10..."):
            (b_fit, c_fit, d_fit), tg_rmse = fit_tg(x_mol_arr, Tg_arr, Tg_d)

        # Fit m poly
        poly1 = np.polyfit(x_mol_arr, m_arr, 1)
        poly2 = np.polyfit(x_mol_arr, m_arr, 2)
        rmse1 = np.sqrt(np.mean((np.polyval(poly1,x_mol_arr)-m_arr)**2))
        rmse2 = np.sqrt(np.mean((np.polyval(poly2,x_mol_arr)-m_arr)**2))
        m_poly = poly2 if (rmse1-rmse2)>0.05 else poly1
        poly_deg = 2 if (rmse1-rmse2)>0.05 else 1

        # Smooth curves
        x_smooth   = np.linspace(0, max(x_mol_arr)*1.05, 200)
        Tg_smooth  = np.array([tg_model(x, b_fit, c_fit, d_fit, Tg_d) for x in x_smooth])
        m_eq12_sm  = np.array([m_from_tg(Tg, Tg_d, m_d) for Tg in Tg_smooth])
        m_poly_sm  = np.polyval(m_poly, x_smooth)

        # ── Figure: 5 panels ─────────────────────────────────────────────────
        colors_visc = cm.plasma(np.linspace(0.1, 0.9, len(results)))
        fig2, axes = plt.subplots(1, 5, figsize=(26, 5))
        fig2.suptitle(f"Anhydrous & Hydrous Modelling — {sname}", fontsize=13, fontweight='bold')

        # Panel 1: Tg vs H2O
        ax1 = axes[0]
        ax1.scatter(x_mol_arr, Tg_arr-273.15, color='steelblue', s=60, zorder=5, label='ANN')
        ax1.plot(x_smooth, Tg_smooth-273.15, 'steelblue', linewidth=2,
                 label='Eq.9-10 fit\nb={:.3f}\nc={:.3f}\nd={:.3f}'.format(b_fit,c_fit,d_fit))
        ax1.set_xlabel('H$_2$O (mol%)', fontsize=11)
        ax1.set_ylabel('Tg (°C)', fontsize=11)
        ax1.set_title('Glass transition temperature', fontsize=11)
        ax1.legend(fontsize=7); ax1.grid(True, linestyle='--', alpha=0.4)

        # Panel 2: m vs H2O
        ax2 = axes[1]
        ax2.scatter(x_mol_arr, m_arr, color='tomato', s=60, zorder=5, label='ANN calculated')
        ax2.plot(x_smooth, m_eq12_sm, 'tomato', linewidth=2, linestyle='--', label='Eq. 12 (from Tg fit)')
        ax2.plot(x_smooth, m_poly_sm, 'darkorange', linewidth=2, label='Poly deg-{} (ANN)'.format(poly_deg))
        ax2.axhline(m_d, color='steelblue', linewidth=2, linestyle=':', label='m constant = {:.2f}'.format(m_d))
        ax2.set_xlabel('H$_2$O (mol%)', fontsize=11)
        ax2.set_ylabel('Fragility index m', fontsize=11)
        ax2.set_title('Fragility index — three models', fontsize=11)
        ax2.legend(fontsize=7); ax2.grid(True, linestyle='--', alpha=0.4)

        # Helper
        def visc_panel(ax, m_func, title):
            for i, r in enumerate(results):
                Tg_f = tg_model(r['h2o_mol'], b_fit, c_fit, d_fit, Tg_d)
                m_v  = m_func(r['h2o_mol'])
                try: T_max = brentq(myega_eq, Tg_f, 5000.0, args=(Tg_f, m_v))
                except: T_max = 3000.0
                T_arr2 = np.arange(Tg_f, T_max+50, 25)
                ax.plot(T_arr2-273.15, myega_eq(T_arr2, Tg_f, m_v),
                        color=colors_visc[i], linewidth=2,
                        label='{:.1f} wt%'.format(r['h2o_wt']))
                ax.scatter([Tg_f-273.15], [myega_eq(Tg_f, Tg_f, m_v)],
                           color=colors_visc[i], marker='D', s=40, zorder=5)
            ax.set_xlabel('Temperature (°C)', fontsize=11)
            ax.set_ylabel('log$_{10}$(η / Pa·s)', fontsize=11)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.4)

        visc_panel(axes[2], lambda x: m_d,
                   'm constant = {:.2f}'.format(m_d))
        visc_panel(axes[3],
                   lambda x: m_from_tg(tg_model(x, b_fit, c_fit, d_fit, Tg_d), Tg_d, m_d),
                   'm from Eq. 12')
        visc_panel(axes[4],
                   lambda x: np.polyval(m_poly, x),
                   'm poly deg-{} (ANN fit)'.format(poly_deg))

        plt.tight_layout()

        # ── Excel ─────────────────────────────────────────────────────────────
        wb2 = Workbook()

        # Sheet 1: Parameters
        ws_p = wb2.active; ws_p.title = 'Parameters'
        thin2=Side(style='thin',color='AAAAAA')
        brd2=Border(left=thin2,right=thin2,top=thin2,bottom=thin2)
        ctr2=Alignment(horizontal='center',vertical='center')
        def hdr2(cell, color='1B5E20'):
            cell.font=Font(name='Arial',bold=True,color='FFFFFF',size=10)
            cell.fill=PatternFill('solid',start_color=color)
            cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
            cell.border=brd2
        def dat2(cell, alt=False):
            cell.alignment=ctr2; cell.border=brd2
            if alt: cell.fill=PatternFill('solid',start_color='D6E4F0')
            if isinstance(cell.value,float): cell.number_format='0.000'

        param_data = [
            ('Sample',       sname,              'Sample name'),
            ('A',            A_FIXED,            'log visc at infinite T (Zheng et al. 2011)'),
            ('Tg_d (K)',     round(Tg_d,2),      'Anhydrous Tg'),
            ('Tg_d (C)',     round(Tg_d-273.15,2),'Anhydrous Tg'),
            ('m_dry',        round(m_d,3),        'Anhydrous fragility index'),
            ('b (Eq.10)',    round(b_fit,5),      'Tg(H2O) fit param b'),
            ('c (Eq.10)',    round(c_fit,5),      'Tg(H2O) fit param c'),
            ('d (Eq.10)',    round(d_fit,5),      'Tg(H2O) fit param d'),
            ('Tg fit RMSE (K)', round(tg_rmse,3), 'RMSE of Tg(H2O) fit'),
            ('poly_coeff_0', round(m_poly[0],6),  'Poly m — leading coeff'),
            ('poly_coeff_1', round(m_poly[1],6),  'Poly m — second coeff'),
            ('poly_intercept',round(m_poly[-1],6),'Poly m — intercept'),
        ]
        for c,h in enumerate(['Parameter','Value','Description'],1):
            cell=ws_p.cell(row=1,column=c,value=h)
            hdr2(cell,'1B5E20')
            ws_p.column_dimensions[get_column_letter(c)].width=max(len(h),8)+4
        ws_p.row_dimensions[1].height=25
        for r,(p,v,d) in enumerate(param_data,2):
            for c,val in enumerate([p,v,d],1):
                cell=ws_p.cell(row=r,column=c,value=val)
                dat2(cell,alt=(r%2==0))

        # Sheet 2: Tg and m vs H2O
        ws_tgm = wb2.create_sheet('Tg_m_vs_H2O')
        h2o_headers=['H2O (wt%)','H2O (mol%)','Tg_ANN (K)','Tg_ANN (C)',
                     'Tg_fit (K)','Tg_fit (C)','m_ANN','m_Eq12','m_poly','m_constant']
        for c,h in enumerate(h2o_headers,1):
            cell=ws_tgm.cell(row=1,column=c,value=h)
            hdr2(cell,'4A235A')
            ws_tgm.column_dimensions[get_column_letter(c)].width=max(len(h),8)+3
        ws_tgm.row_dimensions[1].height=25
        for r,res_r in enumerate(results,2):
            Tg_f   = tg_model(res_r['h2o_mol'], b_fit, c_fit, d_fit, Tg_d)
            m_eq12 = m_from_tg(Tg_f, Tg_d, m_d)
            m_p    = float(np.polyval(m_poly, res_r['h2o_mol']))
            vals   = [round(res_r['h2o_wt'],2), round(res_r['h2o_mol'],3),
                      round(res_r['Tg'],2),      round(res_r['Tg']-273.15,2),
                      round(Tg_f,2),             round(Tg_f-273.15,2),
                      round(res_r['m'],3),        round(m_eq12,3),
                      round(m_p,3),              round(m_d,3)]
            for c,val in enumerate(vals,1):
                cell=ws_tgm.cell(row=r,column=c,value=val)
                dat2(cell,alt=(r%2==0))
        ws_tgm.freeze_panes='A2'

        # Sheet 3: Chemistry check
        ws_chem = wb2.create_sheet('Chemistry_Check')
        chem_headers = (['Field'] + OXIDES +
                        ['SUM_input', 'SUM_normalised', 'Fe_treatment'] +
                        [o+'_mol%' for o in OXIDES])
        for c, h in enumerate(chem_headers, 1):
            cell = ws_chem.cell(row=1, column=c, value=h)
            hdr2(cell, '1B5E20')
            ws_chem.column_dimensions[get_column_letter(c)].width = max(len(h), 6) + 2
        ws_chem.row_dimensions[1].height = 28

        # Row 1: Input composition (as entered by user)
        wt_input_row = np.array([row0[o] for o in OXIDES], dtype=float)
        wt_input_row[12] = 0.0  # anhydrous
        sum_input = wt_input_row.sum()

        # Row 2: After iron redistribution
        wt_fe_row, fe_flag_row = redistribute_iron(wt_input_row)

        # Row 3: Normalised to 100% (used for ANN)
        wt_norm_row = normalize_to_100(wt_fe_row)
        sum_norm = wt_norm_row.sum()

        # Mol% conversion
        _, _, mol_per_row = mol_conv(wt_norm_row)

        rows_chem = [
            ['Input (wt%, as entered)'] +
            [round(float(v), 4) for v in wt_input_row] +
            [round(sum_input, 4), '—', '—'] +
            ['—'] * len(OXIDES),

            ['After Fe redistribution (wt%)'] +
            [round(float(v), 4) for v in wt_fe_row] +
            [round(wt_fe_row.sum(), 4), '—', fe_flag_row] +
            ['—'] * len(OXIDES),

            ['Normalised to 100% — used for ANN (wt%)'] +
            [round(float(v), 4) for v in wt_norm_row] +
            [round(sum_input, 4), round(sum_norm, 4), fe_flag_row] +
            [round(float(v), 4) for v in mol_per_row],
        ]

        color_map = ['FFFFFF', 'FFF9C4', 'E3F2FD']
        for r, row_data in enumerate(rows_chem, 2):
            fill_c = PatternFill('solid', start_color=color_map[r-2])
            for c, val in enumerate(row_data, 1):
                cell = ws_chem.cell(row=r, column=c, value=val)
                cell.alignment = Alignment(horizontal='center', vertical='center')
                thin3 = Side(style='thin', color='AAAAAA')
                cell.border = Border(left=thin3, right=thin3, top=thin3, bottom=thin3)
                cell.fill = fill_c
                if isinstance(val, float): cell.number_format = '0.0000'

        # Add a legend below
        legend_row = 6
        ws_chem.cell(row=legend_row, column=1,
            value="Row colours: White = raw input | Yellow = after Fe redistribution | Blue = normalised (used for ANN + mol% conversion)"
        ).font = Font(name='Arial', italic=True, size=8, color='555555')

        ws_chem.freeze_panes = 'B2'

        # Sheets 4-6: viscosity curves
        tg_func = lambda x: tg_model(x, b_fit, c_fit, d_fit, Tg_d)
        make_visc_sheet_hydrous(wb2,'Visc_m_constant',
            results, lambda x: m_d, tg_func, '1F4E79')
        make_visc_sheet_hydrous(wb2,'Visc_m_Eq12',
            results, lambda x: m_from_tg(tg_func(x), Tg_d, m_d), tg_func, '7B1FA2')
        make_visc_sheet_hydrous(wb2,'Visc_m_poly',
            results, lambda x: float(np.polyval(m_poly,x)), tg_func, 'BF360C')

        buf_excel = io.BytesIO(); wb2.save(buf_excel); buf_excel.seek(0)
        buf_fig2  = io.BytesIO(); fig2.savefig(buf_fig2,format='png',dpi=200,bbox_inches='tight')
        buf_fig2.seek(0)

        st.session_state['hyd_done']     = True
        st.session_state['hyd_results']  = results
        st.session_state['hyd_fig']      = fig2
        st.session_state['hyd_buf_excel']= buf_excel
        st.session_state['hyd_buf_fig']  = buf_fig2
        st.session_state['hyd_meta']     = {
            'sname':sname, 'Tg_d':Tg_d, 'm_d':m_d,
            'b':b_fit, 'c':c_fit, 'd':d_fit,
            'tg_rmse':tg_rmse, 'poly_deg':poly_deg,
            'm_poly':m_poly.tolist(),
        }

    # ── Show results ──────────────────────────────────────────────────────────
    if st.session_state['hyd_done']:
        meta = st.session_state['hyd_meta']

        st.subheader("📈 Results")
        st.markdown(f"""
**Tg_dry** = {meta['Tg_d']-273.15:.1f} °C &nbsp;|&nbsp;
**m_dry** = {meta['m_d']:.2f} &nbsp;|&nbsp;
**Tg fit:** b={meta['b']:.4f}, c={meta['c']:.4f}, d={meta['d']:.4f} &nbsp;|&nbsp;
**RMSE** = {meta['tg_rmse']:.2f} K
        """)
        st.pyplot(st.session_state['hyd_fig'])

        st.subheader("🔬 Model descriptions")
        st.markdown("""
#### Panel 1 — Glass transition temperature Tg(H₂O)
Tg decreases with increasing water content because H₂O acts as a network modifier,
depolymerising the melt structure and reducing structural relaxation temperatures.
The fitted curve uses the **Schneider et al. (1997)** two-component model
(implemented in Langhammer et al. 2021, Eq. 9-10):

> Tg(x) = w₁·Tg_H₂O + w₂·Tg_d + c·w₁·w₂·(Tg_d − Tg_H₂O) + d·w₁·w₂²·(Tg_d − Tg_H₂O)

where w₁ and w₂ are mixing weights controlled by parameter b (Eq. 10),
Tg_H₂O = 136 K (−137 °C), and b, c, d are fitted on the ANN-calculated Tg values.

---

#### Panel 2 — Fragility index m(H₂O): three models
The fragility index m quantifies how rapidly viscosity changes near Tg
(m = 12 for a perfectly Arrhenian melt; larger m = more non-Arrhenian behaviour).
Three approaches are compared:

| Model | Description |
|---|---|
| **m constant** | m is fixed at the anhydrous value m_dry. Simplest assumption — water only affects Tg. |
| **m from Eq. 12** | m is derived analytically from the fitted Tg(H₂O) curve using Eq. 12 of Langhammer et al. (2021): m(x) = m_d + (12 − A) · ln(Tg(x) / Tg_d). Physically motivated — m follows from Tg with no additional free parameters. |
| **m polynomial** | m is fitted directly on the ANN-calculated m values using a polynomial (degree 1 or 2). Purely data-driven — captures whatever the ANN computes, without physical constraints. |

---

#### Panels 3-5 — Viscosity curves
MYEGA viscosity curves (Mauro et al. 2009, Eq. 7) calculated using the **fitted Tg(H₂O)**
and each of the three m models above. The ♦ marker indicates the Tg point on each curve.
Comparing the three panels shows the sensitivity of predicted viscosity to the choice of
fragility model — particularly important at low temperatures near Tg.

---

#### References
- Langhammer D., Di Genova D., Steinle-Neumann G. (2022). *Modeling viscosity of volcanic melts with ANN.* GGG, 23, e2022GC010673. https://doi.org/10.1029/2022GC010673
- Langhammer D., Di Genova D., Steinle-Neumann G. (2021). *Modeling the viscosity of anhydrous and hydrous volcanic melts.* GGG, 22, e2021GC009918. https://doi.org/10.1029/2021GC009918
- Mauro J.C. et al. (2009). *Viscosity of glass-forming liquids.* PNAS, 106, 19780–19784. https://doi.org/10.1073/pnas.0911705106
- Schneider H.A. et al. (1997). *The glass transition temperature of random copolymers.* Polymer, 38, 1323–1337. https://www.sciencedirect.com/science/article/pii/S0032386196006520
        """)

        st.subheader("📥 Download results")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download Excel (all models)",
                data=st.session_state['hyd_buf_excel'],
                file_name=f"hydrous_visc_{meta['sname']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            st.download_button("⬇️ Download Plot (PNG)",
                data=st.session_state['hyd_buf_fig'],
                file_name=f"hydrous_visc_{meta['sname']}.png",
                mime="image/png")
