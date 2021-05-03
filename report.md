
Report
======

Contents
========

* [Original Data](#original-data)
* [Filter Data](#filter-data)
* [Elimination Data](#elimination-data)
* [Slided Data *(windowsize=2)*](#slided-data-windowsize2)
* [Optimization Linear Regression *(windowsize=2)*](#optimization-linear-regression-windowsize2)
* [Best hyperparameters Linear Regression *(windowsize=2)*](#best-hyperparameters-linear-regression-windowsize2)
* [Metrics evaluation Linear Regression *(windowsize=2)*](#metrics-evaluation-linear-regression-windowsize2)
* [Slided Data *(windowsize=3)*](#slided-data-windowsize3)
* [Optimization Linear Regression *(windowsize=3)*](#optimization-linear-regression-windowsize3)
* [Best hyperparameters Linear Regression *(windowsize=3)*](#best-hyperparameters-linear-regression-windowsize3)
* [Metrics evaluation Linear Regression *(windowsize=3)*](#metrics-evaluation-linear-regression-windowsize3)
* [Slided Data *(windowsize=4)*](#slided-data-windowsize4)
* [Optimization Linear Regression *(windowsize=4)*](#optimization-linear-regression-windowsize4)
* [Best hyperparameters Linear Regression *(windowsize=4)*](#best-hyperparameters-linear-regression-windowsize4)
* [Metrics evaluation Linear Regression *(windowsize=4)*](#metrics-evaluation-linear-regression-windowsize4)
* [Distribution Data](#distribution-data)
* [PCA analisys](#pca-analisys)
* [ICA Analisys](#ica-analisys)
* [Correlation matrix](#correlation-matrix)
* [Validation curve Linear Regression *(fit_intercept, windowsize=2)*](#validation-curve-linear-regression-fit_intercept-windowsize2)
* [Validation curve Linear Regression *(normalize, windowsize=2)*](#validation-curve-linear-regression-normalize-windowsize2)
* [Prediction Linear Regression *(UCI, windowsize=2)*](#prediction-linear-regression-uci-windowsize2)
* [Validation curve Linear Regression *(fit_intercept, windowsize=3)*](#validation-curve-linear-regression-fit_intercept-windowsize3)
* [Validation curve Linear Regression *(normalize, windowsize=3)*](#validation-curve-linear-regression-normalize-windowsize3)
* [Prediction Linear Regression *(UCI, windowsize=3)*](#prediction-linear-regression-uci-windowsize3)
* [Validation curve Linear Regression *(fit_intercept, windowsize=4)*](#validation-curve-linear-regression-fit_intercept-windowsize4)
* [Validation curve Linear Regression *(normalize, windowsize=4)*](#validation-curve-linear-regression-normalize-windowsize4)
* [Prediction Linear Regression *(UCI, windowsize=4)*](#prediction-linear-regression-uci-windowsize4)
* [Windowsize comparison Linear Regression](#windowsize-comparison-linear-regression)
  
<div style="page-break-after: always;"></div>  
    
  
# Original Data


  

|Fecha|cod_ine|provincia|Casos|Fallecidos|Hospitalizados|UCI|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|2020-01-01|0|No consta|0|0|0|0|
|2020-01-01|1|Araba/Álava|0|0|0|0|
|2020-01-01|2|Albacete|0|0|0|0|
|2020-01-01|3|Alicante/Alacant|0|0|0|0|
|2020-01-01|4|Almería|0|0|0|0|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Filter Data


  

|Fecha|cod_ine|Casos|Fallecidos|Hospitalizados|UCI|
| :---: | :---: | :---: | :---: | :---: | :---: |
|2020-01-01|110|0|0|1|0|
|2020-01-02|110|0|0|0|0|
|2020-01-03|110|0|0|0|0|
|2020-01-04|110|0|0|0|0|
|2020-01-05|110|0|0|0|0|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Elimination Data


  

|Fecha|Casos|Fallecidos|Hospitalizados|UCI|
| :---: | :---: | :---: | :---: | :---: |
|2020-01-01|0|0|1|0|
|2020-01-02|0|0|0|0|
|2020-01-03|0|0|0|0|
|2020-01-04|0|0|0|0|
|2020-01-05|0|0|0|0|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Slided Data *(windowsize=2)*


  

|Fecha|Casos_t-1|Casos|Fallecidos_t-1|Fallecidos|Hospitalizados_t-1|Hospitalizados|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|2020-01-02|0|0|0|0|1|0|
|2020-01-03|0|0|0|0|0|0|
|2020-01-04|0|0|0|0|0|0|
|2020-01-05|0|0|0|0|0|0|
|2020-01-06|0|0|0|0|0|0|
<br>  

|UCI_t-1|UCI|
| :---: | :---: |
|0|0|
|0|0|
|0|0|
|0|0|
|0|0|
  
<div style="page-break-after: always;"></div>  
    
  
# Optimization Linear Regression *(windowsize=2)*


  

|mean_fit_time|std_fit_time|mean_score_time|std_score_time|param_normalize|param_fit_intercept|params|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.0032|0.0004|0.0022|0.0004|True|True|{'normalize': 'True', 'fit_intercept': 'True'}|
|0.0032|0.0004|0.002|0.0|False|True|{'normalize': 'False', 'fit_intercept': 'True'}|
|0.003|0.0|0.0016|0.0005|True|False|{'normalize': 'True', 'fit_intercept': 'False'}|
|0.0028|0.0007|0.0024|0.0008|False|False|{'normalize': 'False', 'fit_intercept': 'False'}|
<br>  

|split0_test_score|split1_test_score|split2_test_score|split3_test_score|split4_test_score|mean_test_score|std_test_score|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.7807|0.8211|0.0649|0.4123|0.6399|0.5438|0.2789|
|0.7807|0.8211|0.0649|0.4123|0.6399|0.5438|0.2789|
|0.7807|0.8211|0.0649|0.4123|0.6399|0.5438|0.2789|
|0.7807|0.8211|0.0649|0.4123|0.6399|0.5438|0.2789|
<br>  

|rank_test_score|split0_train_score|split1_train_score|split2_train_score|split3_train_score|split4_train_score|mean_train_score|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|1|0.843|0.851|0.8429|0.8846|0.8512|0.8545|
|1|0.843|0.851|0.8429|0.8846|0.8512|0.8545|
|1|0.843|0.851|0.8429|0.8846|0.8512|0.8545|
|1|0.843|0.851|0.8429|0.8846|0.8512|0.8545|
<br>  

|std_train_score|
| :---: |
|0.0155|
|0.0155|
|0.0155|
|0.0155|
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters Linear Regression *(windowsize=2)*


  

|Model|normalize|fit_intercept|
| :---: | :---: | :---: |
|Linear Regression|True|True|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation Linear Regression *(windowsize=2)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|2.5634|14.2279|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Slided Data *(windowsize=3)*


  

|Fecha|Casos_t-2|Casos_t-1|Casos|Fallecidos_t-2|Fallecidos_t-1|Fallecidos|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|2020-01-03|0|0|0|0|0|0|
|2020-01-04|0|0|0|0|0|0|
|2020-01-05|0|0|0|0|0|0|
|2020-01-06|0|0|0|0|0|0|
|2020-01-07|0|0|0|0|0|0|
<br>  

|Hospitalizados_t-2|Hospitalizados_t-1|Hospitalizados|UCI_t-2|UCI_t-1|UCI|
| :---: | :---: | :---: | :---: | :---: | :---: |
|1|0|0|0|0|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|
  
<div style="page-break-after: always;"></div>  
    
  
# Optimization Linear Regression *(windowsize=3)*


  

|mean_fit_time|std_fit_time|mean_score_time|std_score_time|param_normalize|param_fit_intercept|params|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.0034|0.0005|0.002|0.0|True|True|{'normalize': 'True', 'fit_intercept': 'True'}|
|0.003|0.0006|0.002|0.0|False|True|{'normalize': 'False', 'fit_intercept': 'True'}|
|0.0048|0.0016|0.0034|0.0008|True|False|{'normalize': 'True', 'fit_intercept': 'False'}|
|0.0036|0.0008|0.0026|0.0008|False|False|{'normalize': 'False', 'fit_intercept': 'False'}|
<br>  

|split0_test_score|split1_test_score|split2_test_score|split3_test_score|split4_test_score|mean_test_score|std_test_score|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.7497|0.7528|0.0087|0.2808|0.617|0.4818|0.2925|
|0.7497|0.7528|0.0087|0.2808|0.617|0.4818|0.2925|
|0.7497|0.7528|0.0087|0.2808|0.617|0.4818|0.2925|
|0.7497|0.7528|0.0087|0.2808|0.617|0.4818|0.2925|
<br>  

|rank_test_score|split0_train_score|split1_train_score|split2_train_score|split3_train_score|split4_train_score|mean_train_score|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|1|0.8458|0.8542|0.8436|0.8897|0.8538|0.8574|
|1|0.8458|0.8542|0.8436|0.8897|0.8538|0.8574|
|1|0.8458|0.8542|0.8436|0.8897|0.8538|0.8574|
|1|0.8458|0.8542|0.8436|0.8897|0.8538|0.8574|
<br>  

|std_train_score|
| :---: |
|0.0167|
|0.0167|
|0.0167|
|0.0167|
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters Linear Regression *(windowsize=3)*


  

|Model|normalize|fit_intercept|
| :---: | :---: | :---: |
|Linear Regression|True|True|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation Linear Regression *(windowsize=3)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|2.5579|14.2907|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Slided Data *(windowsize=4)*


  

|Fecha|Casos_t-3|Casos_t-2|Casos_t-1|Casos|Fallecidos_t-3|Fallecidos_t-2|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|2020-01-04|0|0|0|0|0|0|
|2020-01-05|0|0|0|0|0|0|
|2020-01-06|0|0|0|0|0|0|
|2020-01-07|0|0|0|0|0|0|
|2020-01-08|0|0|0|0|0|0|
<br>  

|Fallecidos_t-1|Fallecidos|Hospitalizados_t-3|Hospitalizados_t-2|Hospitalizados_t-1|Hospitalizados|UCI_t-3|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0|1|0|0|0|0|
|0|0|0|0|0|0|0|
|0|0|0|0|0|0|0|
|0|0|0|0|0|0|0|
|0|0|0|0|0|1|0|
<br>  

|UCI_t-2|UCI_t-1|UCI|
| :---: | :---: | :---: |
|0|0|0|
|0|0|0|
|0|0|0|
|0|0|0|
|0|0|0|
  
<div style="page-break-after: always;"></div>  
    
  
# Optimization Linear Regression *(windowsize=4)*


  

|mean_fit_time|std_fit_time|mean_score_time|std_score_time|param_normalize|param_fit_intercept|params|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.0034|0.0005|0.002|0.0|True|True|{'normalize': 'True', 'fit_intercept': 'True'}|
|0.0032|0.0004|0.0018|0.0004|False|True|{'normalize': 'False', 'fit_intercept': 'True'}|
|0.0032|0.0004|0.002|0.0|True|False|{'normalize': 'True', 'fit_intercept': 'False'}|
|0.003|0.0|0.002|0.0|False|False|{'normalize': 'False', 'fit_intercept': 'False'}|
<br>  

|split0_test_score|split1_test_score|split2_test_score|split3_test_score|split4_test_score|mean_test_score|std_test_score|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.7277|0.7939|0.0643|0.2496|0.593|0.4857|0.2823|
|0.7277|0.7939|0.0643|0.2496|0.593|0.4857|0.2823|
|0.7277|0.7939|0.0643|0.2496|0.593|0.4857|0.2823|
|0.7277|0.7939|0.0643|0.2496|0.593|0.4857|0.2823|
<br>  

|rank_test_score|split0_train_score|split1_train_score|split2_train_score|split3_train_score|split4_train_score|mean_train_score|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|1|0.8531|0.8572|0.8497|0.8956|0.8608|0.8633|
|1|0.8531|0.8572|0.8497|0.8956|0.8608|0.8633|
|1|0.8531|0.8572|0.8497|0.8956|0.8608|0.8633|
|1|0.8531|0.8572|0.8497|0.8956|0.8608|0.8633|
<br>  

|std_train_score|
| :---: |
|0.0166|
|0.0166|
|0.0166|
|0.0166|
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters Linear Regression *(windowsize=4)*


  

|Model|normalize|fit_intercept|
| :---: | :---: | :---: |
|Linear Regression|True|True|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation Linear Regression *(windowsize=4)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|2.4524|12.3391|
  
  
<div style="page-break-after: always;"></div>  
  
# Distribution Data


  
<img src=".\img\1.Distribution Data.jpg" />  
<img src=".\img\2.Distribution Data(1).jpg" />  
<img src=".\img\3.Distribution Data(2).jpg" />  
<img src=".\img\4.Distribution Data(3).jpg" />  
<div style="page-break-after: always;"></div>  
  
# PCA analisys


  
<img src=".\img\5.PCA analisys.jpg" />  
<div style="page-break-after: always;"></div>  
  
# ICA Analisys


  
<img src=".\img\6.ICA Analisys.jpg" />  
<img src=".\img\7.ICA Analisys(1).jpg" />  
<div style="page-break-after: always;"></div>  
  
# Correlation matrix


  
<img src=".\img\8.Correlation matrix.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Validation curve Linear Regression *(fit_intercept, windowsize=2)*


  
<img src=".\img\9.Validation curve Linear Regression-fit_intercept, windowsize=2.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Validation curve Linear Regression *(normalize, windowsize=2)*


  
<img src=".\img\10.Validation curve Linear Regression-normalize, windowsize=2.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Prediction Linear Regression *(UCI, windowsize=2)*


  
<img src=".\img\11.Prediction Linear Regression-UCI, windowsize=2.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Validation curve Linear Regression *(fit_intercept, windowsize=3)*


  
<img src=".\img\12.Validation curve Linear Regression-fit_intercept, windowsize=3.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Validation curve Linear Regression *(normalize, windowsize=3)*


  
<img src=".\img\13.Validation curve Linear Regression-normalize, windowsize=3.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Prediction Linear Regression *(UCI, windowsize=3)*


  
<img src=".\img\14.Prediction Linear Regression-UCI, windowsize=3.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Validation curve Linear Regression *(fit_intercept, windowsize=4)*


  
<img src=".\img\15.Validation curve Linear Regression-fit_intercept, windowsize=4.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Validation curve Linear Regression *(normalize, windowsize=4)*


  
<img src=".\img\16.Validation curve Linear Regression-normalize, windowsize=4.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Prediction Linear Regression *(UCI, windowsize=4)*


  
<img src=".\img\17.Prediction Linear Regression-UCI, windowsize=4.jpg" />  
<div style="page-break-after: always;"></div>  
  
# Windowsize comparison Linear Regression


  
<img src=".\img\19.Windowsize comparison Linear Regression.jpg" />