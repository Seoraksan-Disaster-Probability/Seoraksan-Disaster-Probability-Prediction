# Seoraksan National Park: Visitor and Accident Prediction Models

This project aims to develop machine learning models to predict the number of visitors to Seoraksan National Park and to assess the probability of accidents occurring on a given day. By leveraging historical data on visitor numbers, accident records, and weather conditions, we strive to provide insights that can aid in park management, resource allocation, and visitor safety.

## Project Goals

1.  **Visitor Count Prediction (Regression):** Develop a model to accurately forecast the daily number of visitors to Seoraksan National Park.
2.  **Accident Probability Prediction (Classification):** Develop a model to predict the likelihood of one or more accidents occurring on a specific day, utilizing weather data and the *predicted visitor counts* as key features.

## Datasets Used

The project utilizes several datasets, which are preprocessed and merged:

1.  **Accident Data:** Historical records of accidents within Seoraksan National Park (e.g., from the National Fire Agency or Korea National Park Service), including dates, number of rescued individuals, causes, and outcomes. Filtered for the Sokcho region relevant to Seoraksan.
2.  **Weather Data:** Daily weather observations for the Seoraksan area (e.g., from the Korea Meteorological Administration), including temperature (min, max, avg), wind speed (avg, max), humidity, precipitation, etc. Specific weather station data for 'Sokcho' is prioritized.
3.  **Visitor Data:** Daily visitor statistics for Seoraksan National Park, specifically focusing on the 'Seorak-dong' district, and potentially total national park visitor counts for broader context.

*Initial data preprocessing 통합 involves merging these datasets based on date, handling missing values, and standardizing column names (Korean to English).*

## Methodology

The project is divided into two main modeling tasks:

### 1. Visitor Count Prediction Model (Regression)

*   **Objective:** Predict `Total_Visitor_Count`.
*   **Data Preprocessing:**
    *   Loading and cleaning raw visitor, weather, and potentially accident data.
    *   **Log Transformation:** Applied to the target variable (`Total_Visitor_Count`) to stabilize variance and handle right-skewed distribution.
    *   **Outlier Treatment:** IQR-based capping/flooring applied to the log-transformed target variable to mitigate the impact of extreme values.
*   **Feature Engineering:**
    *   Date-based features: `is_weekend`, `month_sin`, `month_cos`, `month_10` (binary for October).
    *   Holiday features: `is_final_long_holiday` (identifies long weekends including bridge holidays).
    *   Lagged features: `Total_Visitor_Count_LagN` (e.g., N=1, 7, 14, 30).
    *   Rolling mean features: `Total_Visitor_Count_RollN_Mean` (e.g., N=7, 14, 30).
    *   Weather-derived features: `consecutive_rain_3days`, `consecutive_freeze_2days`.
    *   Time-based features: `Hour_Of_Max_Temp`, `Hour_Of_Min_Temp`.
    *   Interaction terms: `Temp_Humidity_Interaction`.
    *   Past accident information: `rescue_event_yesterday`.
*   **Model Selection & Training:**
    *   Primary model: **LightGBM Regressor (`LGBMRegressor`)**.
    *   Hyperparameter Tuning: `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation.
    *   Evaluation Metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R2 Score (on the original scale after inverse log transformation).
*   **Key Findings (Visitor Model):**
    *   The model achieved an R2 score of approximately **0.69 - 0.77** (depending on the test period and specific preprocessing steps like outlier treatment) on the original scale for the test set, indicating a good ability to explain visitor count variance.
    *   MAE was in the range of ~680-1050 visitors, depending on the test period and preprocessing.
    *   Log transformation of the target variable and appropriate feature engineering (especially lag and holiday features) were crucial for performance.

### 2. Accident Probability Prediction Model (Classification)

*   **Objective:** Predict `Rescue_Event` (binary: 1 if an accident occurred, 0 otherwise).
*   **Data Preprocessing:**
    *   Utilizes the preprocessed and merged dataset.
    *   The **predicted visitor count** from the regression model is a key input feature.
*   **Feature Engineering:**
    *   Similar date-based, holiday, weather-derived, time-based, and interaction features as the visitor model.
    *   **Crucially includes `Predicted_Total_Visitor_Count` as a feature.**
    *   Lagged features based on historical actual visitor counts and rescue events.
*   **Model Selection & Training:**
    *   Primary model: **RandomForest Classifier (`RandomForestClassifier`)**.
    *   Handling Class Imbalance:
        *   **SMOTE (Synthetic Minority Over-sampling Technique)** applied to the training data.
        *   `class_weight='balanced'` option in RandomForestClassifier (when SMOTE is not used).
    *   Hyperparameter Tuning: `RandomizedSearchCV` with k-fold cross-validation (e.g., 5 folds).
    *   Evaluation Metrics: Precision, Recall, F1-Score, ROC AUC, and Confusion Matrix.
*   **Key Findings (Accident Model):**
    *   The model achieved a **ROC AUC of approximately 0.72 - 0.78** on the test set, indicating a fair ability to discriminate between days with and without accidents.
    *   **Threshold Adjustment was critical:**
        *   At the default threshold of 0.5, Recall for accident events (class 1) was very low (often 0.00).
        *   By adjusting the threshold (e.g., to ~0.10 - 0.30 based on PR curve optimization for Recall or F1), a significantly better balance was achieved.
        *   **Recall-focused (e.g., Recall >= 0.70-0.75):** Achieved Recall жертвуя (sacrificing) Precision. For example, with a threshold of ~0.13-0.15, Recall reached ~0.70-0.78, while Precision was around ~0.35-0.39. This means successfully identifying a large portion of actual accident days, but with a higher number of false alarms.
        *   **Precision-focused (e.g., Precision >= 0.65):** Resulted in very low Recall, making it less practical for this problem.
    *   The inclusion of predicted visitor counts and robust feature engineering are important for this model.

## How to Use This Project

1.  **Data Preparation:**
    *   Ensure your raw data files (accident, weather, visitor) are placed in the respective directories defined in the configuration section of the scripts (e.g., `data/accident/`, `data/weather/`).
    *   Run the data preprocessing and merging script (`preprocess_and_merge_data.py` or similar, based on the final version of your data pipeline script) to generate the `preprocessed_data.csv` file.
2.  **Train Visitor Prediction Model:**
    *   Execute the regression model training script (e.g., `train_visitor_regression_refactored.py`).
    *   This will train the `LGBMRegressor`, perform hyperparameter tuning, evaluate it, and save the trained model package (e.g., `visitor_lgbm_regressor_best_model.pkl`) along with evaluation results and visualizations.
3.  **Train Accident Prediction Model:**
    *   Execute the classification model training script (e.g., `train_rescue_classification_final.py`).
    *   This script uses the `preprocessed_data.csv` (which includes actual visitor counts for training).
    *   It will train the `RandomForestClassifier`, handle class imbalance (e.g., with SMOTE), perform hyperparameter tuning, evaluate it, and save the trained model package (e.g., `rescue_model_package.pkl`) with an optimal threshold.
4.  **Integrated Prediction (Predicting Future Accident Probability):**
    *   Execute the integrated prediction script (e.g., `total.py` or `predict_rescue_with_visitor_model.py`).
    *   This script will:
        *   Load the trained visitor prediction model.
        *   Take new input data (e.g., weather forecasts for future dates).
        *   Perform feature engineering identical to visitor model training (handling missing actual visitor counts for lags appropriately for future dates).
        *   Predict visitor counts for the future dates.
        *   Load the trained accident prediction model.
        *   Perform feature engineering identical to accident model training, incorporating the *predicted visitor counts*.
        *   Predict the probability of accidents for the future dates using the chosen/saved optimal threshold.
        *   Output the final predictions.

## Future Work

*   Explore more advanced time-series models for visitor prediction (e.g., SARIMA, Prophet).
*   Incorporate more granular data if available (e.g., specific trail usage, real-time weather alerts).
*   Refine feature engineering for the accident prediction model to improve precision while maintaining high recall.
*   Develop a more sophisticated method for handling missing future values for lag features in the prediction pipeline.
*   Deploy the models as a web service or dashboard for practical use.

## Dependencies
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   joblib
*   holidays
*   lightgbm
*   xgboost (if used)
*   imbalanced-learn

(You can generate a `requirements.txt` file using `pip freeze > requirements.txt`)
