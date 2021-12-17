# Developing a Deep Survival Model for in-hospital patients diagnosed with ischemic heart disease

## Introduction and Objective
Ischemic heart disease (IHD), also referred to as coronary artery disease, is the leading cause of death in the world. One in four major IHD events are fatal and 40% are in-hospital deaths. Therefore, an early identification of patients with a high mortality risk is essential so that timely and appropriate treatments can be applied to improve the survival chance. We utilized a deep survival model to predict the time to mortality, which could provide greater amounts of information compared to the usual classification approach. We created a sequence based model to model after a time dependent dataset in MIMIC III v1.4.

We plan to design a model that uses a timeline of historical patient’s features leading up to the most recent event and predict a distribution describing the ranking of patient’s likelihood of death. A robust model is expected to predict a distribution that is weighted closer to the present for samples that are very close to experiencing an event and predict a much wider distribution for samples that are unlikely to experience an event any time soon. 

In the current study, we built a parametric survival function which assumes that the distribution of an individual survival time follows a Weibull distribution. The model is compared with the non-parametric function (i.e., the Kaplan-Meier estimator) which does not assume any feature association with the survival time and relies on the count of deaths and the number of patients alive. The Harrell's estimator of the concordance index is implemented for model comparison. No study has been done to date with respect to utilizing sequence-based survival models for the prediction of in-hospital mortality risk in IHD patients and our study would add value to this area of work.


## Library Installation

Packages required:

- numpy=1.21.1

- pandas=1.3.0

- scikit-learn=0.22.1

- torch=1.10.0

- scipy=1.5.0

- lifelines=0.26.4

- torch=1.10.0

## Implementation Details

## ETL
Data is extracted from Bigquery and transformed. It is then pivoted in python. Please view the folder ETL for scripts.

## Modelling
Data is further loaded into the appropriate shape for modelling. For each model, please refer to the model folder. We utilized NVIDIA Tesla K80 GPU to train the model.

Hyperparameters can be tuned by editing the parameters at the top of each script. Default hyperparameters are the hyperparameters for the best model for each artchitecture respectively.

## Folders description and running the code

Data: label.csv, patient_chartevents_final.csv and patients_data_final.csv are datasets downloaded using Bigquery. Train, test and validation sets are also provided.

ETL: Both SQL and python transformation scripts. These scripts are used to generate the files in the data folder. For transform.py, input: CSVs of labels, events, and cohort files from Bigquery in the data folder. Output: Train, test, validation split for labels and features

model: Each model is provided with a separate folder. Run each model separately to view the results. Input: Train, test, validation split for labels and features from the data folder. Output: Training, validation and test scores, loss and metric curves

## References

1. WHO. The top 10 causes of death. 2020; : https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death.

2. Ahmad FB, Anderson RN. The Leading Causes of Death in the US for 2020. JAMA 2021; 325: 1829.

3. Yang, L.;  Wang, L.;  Deng, Y.;  Sun, L.;  Lou, B.;  Yuan, Z.;  Wu, Y.;  Zhou, B.;  Liu, J.; She, J., Serum lipids profiling perturbances in patients with ischemic heart disease and ischemic cardiomyopathy. Lipids Health Dis 2020, 19 (1), 89.

4. Huang Y, Gulshan K, Nguyen T, Wu Y. Biomarkers of Cardiovascular Disease. Disease Markers 2017; 2017: 1–2.

5. Tanne, D.;  Yaari, S.; Goldbourt, U., High-density lipoprotein cholesterol and risk of ischemic stroke mortality. A 21-year follow-up of 8586 men from the Israeli Ischemic Heart Disease Study. Stroke 1997, 28 (1), 83-7.

6. Vogelzangs, N.;  Beekman, A. T.;  Milaneschi, Y.;  Bandinelli, S.;  Ferrucci, L.; Penninx, B. W., Urinary cortisol and six-year risk of all-cause and cardiovascular mortality. J Clin Endocrinol Metab 2010, 95 (11), 4959-64.

7. Chang, C. L.;  Robinson, S. C.;  Mills, G. D.;  Sullivan, G. D.;  Karalus, N. C.;  McLachlan, J. D.; Hancox, R. J., Biochemical markers of cardiac dysfunction predict mortality in acute exacerbations of COPD. Thorax 2011, 66 (9), 764-8.

8. Shaper AG, Pocock SJ, Walker M, Phillips AN, Whitehead TP, Macfarlane PW. Risk factors for ischaemic heart disease: the prospective phase of the British Regional Heart Study. Journal of Epidemiology & Community Health 1985; 39: 197–209.

9. Grey C, Jackson R, Schmidt M, et al. One in four major ischaemic heart disease events are fatal and 60% are pre-hospital deaths: a national data-linkage study (ANZACS-QI 8). Eur Heart J 2015; : ehv524.

10. Kaegi-Braun N, Mueller M, Schuetz P, Mueller B, Kutz A. Evaluation of Nutritional Support and In-Hospital Mortality in Patients With Malnutrition. JAMA Netw Open 2021; 4: e2033433.

11. McCullagh R, O’Connell E, O’Meara S, et al. Augmented exercise in hospital improves physical performance and reduces negative post hospitalization events: a randomized controlled trial. BMC Geriatr 2020; 20: 46.

12. Nicholson G, Gandra S, Halbert R, Richarriya A, Nordyke R. Patient-level costs of major cardiovascular conditions: a review of the international literature. CEOR 2016; Volume 8: 495–506.

13. Hou N., Li M., He L.,  et al. Predicting 30-days mortality for MIMIC-III patients with sepsis-3: a machine learning approach using XGboost. Journal of translational medicine 2020; 18(1), 1-14.

14. Kaplan, E. L.; Meier, P. Nonparametric estimation from incomplete observations. J. Amer. Statist. Assoc. 1958; 53 (282): 457–481. 

15. Breslow, N. E. Analysis of Survival Data under the Proportional Hazards Model. International Statistical Review / Revue Internationale de Statistique. 1975; 43 (1): 45–57.

16. Katzman JL, Shaham U, Cloninger A et al., DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology. 2018

17. Martinsson E. WTTE-RNN: Weibull time to event recurrent neural network. Master’s thesis, University of Gothenburg, Sweden. 2016
