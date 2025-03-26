# ELEN0062 - Machine Learning Project 3: Competition

**Authors:**  
- Alexia Donati (S200742)  
- Lei Yang (S201670)

---

## Overview

This project addresses predicting taxi trip destinations using various machine learning models. The challenge involves data cleaning, feature engineering, and model evaluation based on the haversine distance metric.

---

## Data Pre-processing

- **Cleaning:**  
  - Removed trips with missing data, empty trajectories, and duplicate trip entries.
  - Dropped constant or redundant features (e.g., `DAY_TYPE` and standardized `MISSING_DATA`).
  
- **Feature Engineering:**  
  - Converted `CALL_TYPE` to numerical values.
  - Replaced null values in `ORIGIN_CALL` and `ORIGIN_STAND` with 0.
  - Extracted day of week and hour from `TIMESTAMP`.
  - Final features: `CALL_TYPE`, `ORIGIN_CALL`, `ORIGIN_STAND`, `TAXI_ID`, `POLYLINE`, `WEEKDAY`, `HOUR` (with `END_Long` and `END_Lat` as targets).

---

## Modeling Approach

### Non-Sequential Models
- **Decision Tree, KNN, Ridge, Lasso, Neural Network, and Stacking Ensemble:**  
  - Tuned hyperparameters via 10-fold cross-validation.
  - Achieved similar haversine scores around 3.6–3.7 km (with stacking performing slightly worse).

### Sequential Model
- **LSTM (Recurrent Neural Network):**  
  - Used the first 5 coordinates of the `POLYLINE` (padded and scaled).
  - Trained with dropout, dynamic learning rate, and early stopping.
  - Achieved superior performance with a haversine distance around 2.78 km (Public) and 3.16 km (Private).

---

## Summary of Results

| **Model**            | **Public Score** | **Private Score** |
|----------------------|------------------|-------------------|
| Non-Sequential (avg)| ~3.6–3.7 km      | ~3.63 km          |
| **LSTM (Sequential)**| **2.78 km**      | **3.16 km**       |

---

## Running the Project

1. **Pre-processing:**  
   - Clean the data, handle missing values, and extract temporal features.
2. **Model Training:**  
   - Split data (90% train, 10% test) and use 10-fold cross-validation for hyperparameter tuning.
3. **Evaluation:**  
   - Compare model performance using the haversine distance metric.
4. **Sequential Approach:**  
   - Prepare and scale sequential data, then train the LSTM network with dropout and learning rate scheduling.

---

This README provides a concise overview of the project's methodology and outcomes.
