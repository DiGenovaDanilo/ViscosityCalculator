# Changelog

## v1.2.1 (2026-10-23)

Bug-fix release of the version described in Di Genova (Computers & Geosciences, CAGEO-D-26-00946).
No change to the models, the ANN or any computed value, except where noted.

- **Module 2 (Anhydrous and Hydrous Modelling), `fit_tg`.** When the Tg(H2O) fit reached RMSE < 0.5 K, the
  early-termination branch returned two values while the caller expects three, so the module stopped with
  "ValueError: not enough values to unpack". The early termination now leaves the grid search and continues to the
  uncertainty estimate, as in the normal path. Results are unchanged whenever the error did not occur.
- **Module 3, Peridotite (Di Genova et al., 2023).** The reference composition was displayed as wt% but the values
  were the mol% of sample S44F6 (Table 1 of the paper). It is now given in wt% (SiO2 48.11, MgO 30.07, FeOtot 8.51).
  This affects only the displayed composition table: the viscosity model and the effective dry
  molar weight (51.01 g/mol) used for the wt%-mol% conversion of H2O are unchanged.
- Version string set to 1.2.1; the notice at the top of the page and the README now refer to v1.2.1 and to the Zenodo
  concept DOI (10.5281/zenodo.19945908), which always resolves to the latest version.
