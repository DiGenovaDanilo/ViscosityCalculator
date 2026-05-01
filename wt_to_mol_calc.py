import numpy as np




def mol_conv(wt):
    #atomic weights in au
    o_w = 15.999
    si_w = 28.085
    ti_w = 47.867
    al_w = 26.982
    fe_w = 55.845
    mn_w = 54.938
    mg_w = 24.305
    ca_w = 40.078
    na_w = 22.99
    k_w = 39.098
    p_w = 30.974
    cr_w = 51.996
    h_w = 1.007
    
    # Normalisation parameters except temperature
    avg_param = np.array([0.646068624,	0.006122257, 0.105031177, 0.015038062,	0.00071489,	0.056749591, 0.068780308, 0.048986216,	0.028347995, 0.000494008, 6.37117E-05, 0.008792248,	0.014476695, 0.218617062, 0.324060518])
    var_param = np.array([0.013966878, 7.54969E-05, 0.00186878, 0.000335448,	7.06524E-07, 0.00498134, 0.00372115, 0.001487268, 0.001029727, 1.62697E-06, 9.68478E-08, 0.000148228, 0.000994256, 0.010377959, 0.047453609])
    
    #calculating mol weight
    sio2 = si_w + 2*o_w
    tio2 = ti_w + 2*o_w
    al2o3 = 2*al_w + 3*o_w
    feo = fe_w + o_w
    mno = mn_w + o_w
    mgo = mg_w + o_w
    cao = ca_w + o_w
    na2o = 2*na_w + o_w
    k2o = 2*k_w + o_w
    p2o5 = 2*p_w + 5*o_w
    cr2o3 = 2*cr_w + 3*o_w
    fe2o3 = 2*fe_w + 3*o_w
    h2o = 2*h_w + o_w
    
    mol_weight = np.array([sio2, tio2, al2o3, feo, mno, mgo, cao, na2o, k2o, p2o5, cr2o3, fe2o3, h2o])
    
    #normalise wt% input to 100% excluding H2O
    wt_sum = np.sum(wt[:-1])
    wt_div = wt/wt_sum
    wt_norm_no_h2o = (100-wt[-1])*wt[:-1]/wt_sum
    #add H2O back to array
    wt_norm = np.append(wt_norm_no_h2o, wt[-1])
    
    #divide wt% by molar weight
    wt_div_mol = wt_norm/mol_weight
    
    #calculate mole fraction and mol%
    wt_div_mol_sum = np.sum(wt_div_mol)
    mol_frac = (wt_div_mol/wt_div_mol_sum)
    mol_per = mol_frac * 100
    
    # Normalisem mol fractions
    
    sm = mol_frac[3] + mol_frac[4] + mol_frac[5] + mol_frac[6] + mol_frac[7] + mol_frac[8]

    if mol_frac[8] + mol_frac[7] == 0:
        nak = 0
    else:
        nak = mol_frac[8]/(mol_frac[8] + mol_frac[7])
    mol_frac = np.append(mol_frac, [sm, nak])
    normalised = (mol_frac - avg_param)/np.sqrt(var_param)
    
    return normalised, mol_frac, mol_per

