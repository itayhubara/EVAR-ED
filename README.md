# Evar.pytorch project
Code was done by Itay Hubara in Pytorch framework 
## Abstract:
Objectives: We sought to develop a prediction score with data from the Vascular Quality Initiative (VQI) EVAR in efforts to
assist endovascular specialists in deciding whether or not a patient is appropriate for short-stay discharge. Background: Small
series describe short-stay discharge following elective EVAR. Our study aims to quantify characteristics associated with this
decision. Methods: The VQI EVAR and NSQIP datasets were queried. Patients who underwent elective EVAR recorded in VQI,
between 1/2010-5/2017 were split 2:1 into test and analytic cohorts via random number assignment. Cross-reference with the
Medicare claims database confirmed all-cause mortality data. Bootstrap sampling was employed in model. Deep learning algorithms independently evaluated each dataset as a sensitivity test. Results: Univariate outcomes, including 30-day survival, were
statistically worse in the DD group when compared to the SD group (all P < 0.05). A prediction score, SD-EVAR, derived from
the VQI EVAR dataset including pre- and intra-op variables that discriminate between SD and DD was externally validated in
NSQIP (Pearson correlation coefficient ¼ 0.79, P < 0.001); deep learning analysis concurred. This score suggests 66% of EVAR
patients may be appropriate for short-stay discharge. A free smart phone app calculating short-stay discharge potential is available
through QxMD Calculate https://qxcalc.app.link/vqidis. Conclusions: Selecting patients for short-stay discharge after EVAR is
possible without increasing harm. The majority of infrarenal AAA patients treated with EVAR in the United States fit a risk profile
consistent with short-stay discharge, representing a significant cost-savings potential to the healthcare system.
Full paper can be found [here](https://journals.sagepub.com/doi/abs/10.1177/1538574420954299)

## Running the script 
To run use:
```
python main_evar.py --model evar_model --save evar_exp 
```


