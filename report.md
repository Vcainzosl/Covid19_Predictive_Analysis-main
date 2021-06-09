
Report
======

Contents
========

* [Original Data](#original-data)
* [Best hyperparameters *(windowsize_1, Casos_t+1)*](#best-hyperparameters-windowsize_1-casos_t1)
* [Metrics evaluation *(windowsize_1, Casos_t+1)*](#metrics-evaluation-windowsize_1-casos_t1)
* [Best hyperparameters *(windowsize_1, Casos_t+7)*](#best-hyperparameters-windowsize_1-casos_t7)
* [Metrics evaluation *(windowsize_1, Casos_t+7)*](#metrics-evaluation-windowsize_1-casos_t7)
* [Best hyperparameters *(windowsize_1, Casos_t+14)*](#best-hyperparameters-windowsize_1-casos_t14)
* [Metrics evaluation *(windowsize_1, Casos_t+14)*](#metrics-evaluation-windowsize_1-casos_t14)
* [Best hyperparameters *(windowsize_7, Casos_t+1)*](#best-hyperparameters-windowsize_7-casos_t1)
* [Metrics evaluation *(windowsize_7, Casos_t+1)*](#metrics-evaluation-windowsize_7-casos_t1)
* [Best hyperparameters *(windowsize_7, Casos_t+7)*](#best-hyperparameters-windowsize_7-casos_t7)
* [Metrics evaluation *(windowsize_7, Casos_t+7)*](#metrics-evaluation-windowsize_7-casos_t7)
* [Best hyperparameters *(windowsize_7, Casos_t+14)*](#best-hyperparameters-windowsize_7-casos_t14)
* [Metrics evaluation *(windowsize_7, Casos_t+14)*](#metrics-evaluation-windowsize_7-casos_t14)
* [Best hyperparameters *(windowsize_14, Casos_t+1)*](#best-hyperparameters-windowsize_14-casos_t1)
* [Metrics evaluation *(windowsize_14, Casos_t+1)*](#metrics-evaluation-windowsize_14-casos_t1)
* [Best hyperparameters *(windowsize_14, Casos_t+7)*](#best-hyperparameters-windowsize_14-casos_t7)
* [Metrics evaluation *(windowsize_14, Casos_t+7)*](#metrics-evaluation-windowsize_14-casos_t7)
* [Best hyperparameters *(windowsize_14, Casos_t+14)*](#best-hyperparameters-windowsize_14-casos_t14)
* [Metrics evaluation *(windowsize_14, Casos_t+14)*](#metrics-evaluation-windowsize_14-casos_t14)
* [Distribution Data](#distribution-data)
* [PCA analisys](#pca-analisys)
* [Validation curve, windowsize_1, Casos_t+1](#validation-curve-windowsize_1-casos_t1)
  
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
    
  
# Best hyperparameters *(windowsize_1, Casos_t+1)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|True|False|-|-|-|-|
|SVR|-|-|0.0001|sigmoid|-|-|
|DNN|-|-|-|-|7.0|[16, 11, 13, 16, 10, 18, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_1, Casos_t+1)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|30.9221|1522.1434|
|SVR|41.3421|2741.6554|
|DNN|30.7089|1499.0334|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_1, Casos_t+7)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|True|False|-|-|-|-|
|SVR|-|-|0.0008|poly|-|-|
|DNN|-|-|-|-|6.0|[10, 14, 12, 14, 14, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'relu', 'relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_1, Casos_t+7)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|44.3529|2982.0234|
|SVR|41.5371|2809.3107|
|DNN|37.795|2076.9354|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_1, Casos_t+14)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|True|True|-|-|-|-|
|SVR|-|-|0.0339|poly|-|-|
|DNN|-|-|-|-|3.0|[10, 12, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_1, Casos_t+14)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|72.0406|7859.3332|
|SVR|38.695|2375.9279|
|DNN|76.8269|7701.0046|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_7, Casos_t+1)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|True|False|-|-|-|-|
|SVR|-|-|0.0002|sigmoid|-|-|
|DNN|-|-|-|-|5.0|[18, 18, 15, 19, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_7, Casos_t+1)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|28.1061|1218.2048|
|SVR|41.5867|2815.6744|
|DNN|36.1511|2176.217|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_7, Casos_t+7)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|False|True|-|-|-|-|
|SVR|-|-|0.1598|rbf|-|-|
|DNN|-|-|-|-|6.0|[16, 19, 15, 11, 10, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'relu', 'relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_7, Casos_t+7)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|42.0842|2746.1027|
|SVR|43.0746|3057.1872|
|DNN|41.5956|3154.5466|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_7, Casos_t+14)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|True|True|-|-|-|-|
|SVR|-|-|0.2126|sigmoid|-|-|
|DNN|-|-|-|-|7.0|[18, 16, 20, 12, 13, 20, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_7, Casos_t+14)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|58.487|5226.9936|
|SVR|52.1441|4292.549|
|DNN|49.2168|4364.1806|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_14, Casos_t+1)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|False|True|-|-|-|-|
|SVR|-|-|0.021|sigmoid|-|-|
|DNN|-|-|-|-|7.0|[12, 16, 16, 10, 12, 20, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_14, Casos_t+1)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|27.2153|1142.8419|
|SVR|44.3037|3196.3455|
|DNN|44.1906|2742.4465|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_14, Casos_t+7)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|False|False|-|-|-|-|
|SVR|-|-|0.0004|sigmoid|-|-|
|DNN|-|-|-|-|3.0|[18, 18, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_14, Casos_t+7)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|50.3871|3721.3793|
|SVR|51.7307|4270.215|
|DNN|46.921|3897.4505|
  
  
<div style="page-break-after: always;"></div>  
    
  
# Best hyperparameters *(windowsize_14, Casos_t+14)*


  

|Model|normalize|fit_intercept|C|kernel|layers|neurons|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Linear Regression|False|False|-|-|-|-|
|SVR|-|-|1.268|poly|-|-|
|DNN|-|-|-|-|4.0|[19, 14, 13, 1]|
<br>  

|activation|optimizer|lr|
| :---: | :---: | :---: |
|-|-|-|
|-|-|-|
|['relu', 'relu', 'relu', 'linear']|Adam|0.001|
  
<div style="page-break-after: always;"></div>  
    
  
# Metrics evaluation *(windowsize_14, Casos_t+14)*


  

|Model|MAE|MSE|
| :---: | :---: | :---: |
|Linear Regression|78.4029|10082.8063|
|SVR|39.5656|2533.599|
|DNN|46.5453|4057.9108|
  
  
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
  
# Validation curve, windowsize_1, Casos_t+1


  
<img src=".\img\6.Validation curve, windowsize_1, Casos_t+1.jpg" />