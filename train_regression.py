import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
import seaborn as sns
import lightgbm as lgb
import holidays
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# --- 0. Configuration ---
INPUT_DATA_FILE = "data/preprocessed_data.csv"
DATE_COLUMN = 'Date'
TARGET_VISITOR_COL = 'Total_Visitor_Count'
ORIGINAL_RESCUE_COUNT_COL = 'Total_Rescued_Count'
ORIGINAL_RESCUE_EVENT_COL_FOR_FE = 'Rescue_Event'

TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9, "Split ratios must sum to 1."

TOP_N_FEATURES_VISITOR = 20
APPLY_LOG_TRANSFORM_VISITOR_TARGET = True
APPLY_OUTLIER_CAPPING_VISITOR_TARGET = True
OUTLIER_CAPPING_METHOD = 'iqr'
IQR_MULTIPLIER = 1.5
PERCENTILE_LOWER = 1
PERCENTILE_UPPER = 99

N_ITER_RANDOM_SEARCH_VISITOR = 50
RANDOM_SEED = 42
CV_N_SPLITS_VISITOR = 3
SCORER_VISITOR = 'r2'
EARLY_STOPPING_ROUNDS_LGBM = 50

CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_RESULTS_DIR = "./regression/visitor_model_training_results" # 디렉토리명 변경
RUN_SPECIFIC_DIR = os.path.join(BASE_RESULTS_DIR, CURRENT_TIME_STR)
VISUALIZATION_DIR = os.path.join(RUN_SPECIFIC_DIR, "visualizations")
os.makedirs(RUN_SPECIFIC_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
print(f"All results will be saved in: {RUN_SPECIFIC_DIR}")


# --- 실험 조건 정의 ---
EXPERIMENT_CONDITIONS = [
    # LightGBM
    {"name": "LGBM_Scaled_OutlierCapped", "model_type": "lgbm", "apply_scaling": True, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": N_ITER_RANDOM_SEARCH_VISITOR},
    {"name": "LGBM_NoScale_OutlierCapped", "model_type": "lgbm", "apply_scaling": False, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": N_ITER_RANDOM_SEARCH_VISITOR},
    {"name": "LGBM_Scaled_NoOutlier", "model_type": "lgbm", "apply_scaling": True, "apply_outlier_capping": False, "n_iter_search": N_ITER_RANDOM_SEARCH_VISITOR},
    {"name": "LGBM_Default", "model_type": "lgbm", "apply_scaling": True, "apply_outlier_capping": False, "n_iter_search": N_ITER_RANDOM_SEARCH_VISITOR}, # A common default

    # RandomForestRegressor
    {"name": "RF_Scaled_OutlierCapped", "model_type": "rf", "apply_scaling": True, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": 30}, # RF can be slower, fewer iterations
    {"name": "RF_NoScale_NoOutlier", "model_type": "rf", "apply_scaling": False, "apply_outlier_capping": False, "n_iter_search": 30},

    # XGBoostRegressor
    {"name": "XGB_Scaled_OutlierCapped", "model_type": "xgb", "apply_scaling": True, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": N_ITER_RANDOM_SEARCH_VISITOR},
    {"name": "XGB_NoScale_NoOutlier", "model_type": "xgb", "apply_scaling": False, "apply_outlier_capping": False, "n_iter_search": N_ITER_RANDOM_SEARCH_VISITOR}
]

# 로그 변환은 모든 실험에서 기본 적용한다고 가정
APPLY_LOG_TRANSFORM_VISITOR_TARGET = True

# --- 1. Data Handling and Initial Preprocessing ---
def load_data(file_path, date_col):
    print(f"\n--- 1. Loading Data from {file_path} ---")
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
        df.sort_values(by=date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Data loaded and sorted.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}"); exit()
    except Exception as e:
        print(f"Error loading data: {e}"); exit()

def initial_preprocess_visitor(df, target_col, rescue_count_col, rescue_event_col, apply_log_transform):
    print(f"\n--- 2. Initial Preprocessing (Log Transform: {apply_log_transform}) ---")
    df_p = df.copy()
    if target_col not in df_p.columns:
        print(f"Error: Target column '{target_col}' not found."); exit()
    df_p[target_col] = pd.to_numeric(df_p[target_col], errors='coerce').fillna(0).astype(float)
    if apply_log_transform:
        df_p[target_col] = np.log1p(df_p[target_col])
        print(f"Applied np.log1p to target column '{target_col}'.")

    if rescue_count_col in df_p.columns:
        df_p[rescue_count_col] = pd.to_numeric(df_p[rescue_count_col], errors='coerce').fillna(0).astype(int)
    if rescue_event_col in df_p.columns:
        df_p[rescue_event_col] = pd.to_numeric(df_p[rescue_event_col], errors='coerce').fillna(0).astype(int)
    elif rescue_count_col in df_p.columns:
        df_p[rescue_event_col] = (df_p[rescue_count_col] > 0).astype(int)

    cols_to_drop_list_like = ['Accident_Cause_List', 'Accident_Outcome_List']
    df_p.drop(columns=[col for col in cols_to_drop_list_like if col in df_p.columns], inplace=True, errors='ignore')
    print("Initial preprocessing complete.")
    return df_p

def cap_outliers(df, column_name, method='iqr', iqr_multiplier=1.5, lower_p=1, upper_p=99):
    print(f"\n--- 2a. Applying Outlier Capping on '{column_name}' using {method} method ---")
    df_c = df.copy()
    if method == 'iqr':
        Q1 = df_c[column_name].quantile(0.25)
        Q3 = df_c[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
    elif method == 'percentile':
        lower_bound = df_c[column_name].quantile(lower_p / 100)
        upper_bound = df_c[column_name].quantile(upper_p / 100)
    else:
        print(f"  Warning: Unknown outlier capping method '{method}'. No capping applied.")
        return df_c

    capped_low_count = (df_c[column_name] < lower_bound).sum()
    capped_high_count = (df_c[column_name] > upper_bound).sum()
    df_c[column_name] = np.clip(df_c[column_name], lower_bound, upper_bound)
    print(f"  Capping on '{column_name}': {capped_low_count} low, {capped_high_count} high values adjusted.")
    return df_c

# --- 3. Visualization Functions ---
def plot_distribution(df, column, title_suffix, save_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=50)
    plt.title(f'Distribution of {column} ({title_suffix})')
    plt.xlabel(column); plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, f'{column}_distribution_{title_suffix.lower().replace(" ", "_")}.png')); plt.close()

def plot_boxplot(df, column, title_suffix, save_dir):
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Box Plot of {column} ({title_suffix})')
    plt.ylabel(column)
    plt.savefig(os.path.join(save_dir, f'{column}_boxplot_{title_suffix.lower().replace(" ", "_")}.png')); plt.close()

def plot_time_series(df, date_column, value_column, title_suffix, save_dir):
    if date_column in df.columns and value_column in df.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df[date_column], df[value_column], marker='.', linestyle='-')
        plt.title(f'Time Series of {value_column} ({title_suffix})')
        plt.xlabel(date_column); plt.ylabel(value_column); plt.grid(True)
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{value_column}_timeseries_{title_suffix.lower().replace(" ", "_")}.png')); plt.close()

# --- 4. Feature Engineering Function (for Visitor Model) ---
def engineer_features_visitor(df_input, date_col, actual_visitor_col_for_lag, actual_rescue_event_col_for_lag):
    print("\n--- 4. Engineering Features for Visitor Model ---")
    df_eng = df_input.copy()
    df_eng.reset_index(drop=True, inplace=True) # Ensure consistent index for shift operations

    # Date-based features
    df_eng['temp_day_of_week'] = df_eng[date_col].dt.dayofweek
    df_eng['temp_month'] = df_eng[date_col].dt.month
    df_eng['is_weekend'] = (df_eng['temp_day_of_week'] >= 5).astype(int)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['temp_month'] / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['temp_month'] / 12)
    month_onehot = pd.get_dummies(df_eng['temp_month'], prefix='month', dtype=int)
    df_eng['month_10'] = month_onehot['month_10'] if 'month_10' in month_onehot.columns else 0

    # Holiday features
    min_year = df_eng[date_col].dt.year.min(); max_year = df_eng[date_col].dt.year.max()
    if pd.isna(min_year) or pd.isna(max_year): 
        current_year = df_eng[date_col].dt.year.iloc[0] if not df_eng.empty else datetime.now().year
        kr_holidays = holidays.KR(years=current_year)
    else: 
        kr_holidays = holidays.KR(years=range(min_year, max_year + 1))
    df_eng['is_official_holiday'] = df_eng[date_col].apply(lambda date: date in kr_holidays)
    df_eng['is_day_off_official'] = (df_eng['is_weekend'] | df_eng['is_official_holiday']).astype(int)
    df_eng['day_off_group'] = (df_eng['is_day_off_official'].diff() != 0).astype(int).cumsum() # diff() is enough
    df_eng['consecutive_official_days_off'] = df_eng.groupby('day_off_group')['is_day_off_official'].transform('sum')
    long_holiday_threshold = 3
    df_eng['is_base_long_holiday'] = ((df_eng['is_day_off_official'] == 1) & (df_eng['consecutive_official_days_off'] >= long_holiday_threshold)).astype(int)
    df_eng['prev_day_is_off'] = df_eng['is_day_off_official'].shift(1).fillna(0).astype(int)
    df_eng['next_day_is_off'] = df_eng['is_day_off_official'].shift(-1).fillna(0).astype(int)
    df_eng['is_bridge_day_candidate'] = ((df_eng['is_day_off_official'] == 0) & (df_eng['prev_day_is_off'] == 1) & (df_eng['next_day_is_off'] == 1)).astype(int)
    df_eng['is_extended_holiday'] = df_eng['is_base_long_holiday'].copy() # Start with base
    df_eng.loc[df_eng['is_bridge_day_candidate'].values == 1, 'is_extended_holiday'] = 1
    df_eng.loc[df_eng['is_bridge_day_candidate'].shift(-1).fillna(False).values, 'is_extended_holiday'] = 1
    df_eng.loc[df_eng['is_bridge_day_candidate'].shift(1).fillna(False).values, 'is_extended_holiday'] = 1
    df_eng['final_holiday_group'] = (df_eng['is_extended_holiday'].diff() != 0).astype(int).cumsum()
    df_eng['consecutive_extended_days_off'] = df_eng.groupby('final_holiday_group')['is_extended_holiday'].transform('sum')
    df_eng['is_final_long_holiday'] = ((df_eng['is_extended_holiday'] == 1) & (df_eng['consecutive_extended_days_off'] >= long_holiday_threshold)).astype(int)

    # Lagged/Rolling features
    lag_values = [1, 7, 14, 30]
    rolling_windows = [7, 14, 30]
    lag_base_col = actual_visitor_col_for_lag if actual_visitor_col_for_lag and actual_visitor_col_for_lag in df_eng.columns else None
    if lag_base_col:
        for lag in lag_values: 
            df_eng[f'{lag_base_col}_Lag{lag}'] = df_eng[lag_base_col].shift(lag).fillna(0)
        for window in rolling_windows: 
            df_eng[f'{lag_base_col}_Roll{window}_Mean'] = df_eng[lag_base_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
    else:
        for lag in lag_values: 
            df_eng[f'{TARGET_VISITOR_COL}_Lag{lag}'] = 0
        for window in rolling_windows: 
            df_eng[f'{TARGET_VISITOR_COL}_Roll{window}_Mean'] = 0
        if actual_visitor_col_for_lag: 
            print(f"  Warning: Lag base col '{actual_visitor_col_for_lag}' not found. Lags/Rolls filled with 0.")

    # Past rescue event
    rescue_lag_base_col = actual_rescue_event_col_for_lag if actual_rescue_event_col_for_lag and actual_rescue_event_col_for_lag in df_eng.columns else None
    if rescue_lag_base_col:
        df_eng['rescue_event_yesterday'] = df_eng[rescue_lag_base_col].shift(1).fillna(0)
    else: 
        df_eng['rescue_event_yesterday'] = 0
        if actual_rescue_event_col_for_lag: print(f"  Warning: Rescue lag base col '{actual_rescue_event_col_for_lag}' not found. 'rescue_event_yesterday' filled with 0.")

    # Weather-derived features
    weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'), 
                     'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    for condition, (col, window, new_col_name) in weather_rules.items():
        if col in df_eng.columns:
            is_condition = (df_eng[col] > 0 if condition == 'rain' else df_eng[col] < 0).astype(int)
            df_eng[new_col_name] = (is_condition.rolling(window=window, min_periods=window).sum() == window).astype(int).fillna(0)
            
    # Time transformation
    time_cols_map = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    for orig_col, new_col in time_cols_map.items():
        if orig_col in df_eng.columns:
            try: 
                df_eng[new_col] = pd.to_datetime(df_eng[orig_col], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
            except: 
                df_eng[new_col] = -1
            
    # Interaction term
    temp_col, hum_col = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    if temp_col in df_eng.columns and hum_col in df_eng.columns:
        # Impute NaNs before multiplication, using overall mean (or train mean if available)
        mean_temp = df_eng[temp_col].mean(); mean_hum = df_eng[hum_col].mean()
        df_eng['Temp_Humidity_Interaction'] = df_eng[temp_col].fillna(mean_temp) * df_eng[hum_col].fillna(mean_hum)
        
    # Cleanup temporary columns
    temp_cols_to_drop = [
        'temp_day_of_week', 'temp_month', 'is_official_holiday', 'is_day_off_official', 
        'day_off_group', 'consecutive_official_days_off', 'is_base_long_holiday', 
        'prev_day_is_off', 'next_day_is_off', 'is_bridge_day_candidate', 
        'is_extended_holiday', 'final_holiday_group', 'consecutive_extended_days_off',
        'TimeOfMaxTempC', 'TimeOfMinTempC' # Original time columns if new ones are created
    ]
    df_eng.drop(columns=[col for col in temp_cols_to_drop if col in df_eng.columns], inplace=True, errors='ignore')
    print("Feature engineering for visitor model complete.")
    return df_eng

# --- 5. Data Splitting, Feature Selection & Scaling (for Training Visitor Model) ---
def prepare_and_split_data_for_visitor_training(df_engineered, target_col_name, date_col_name, 
                                                train_r, val_r, test_r, top_n, 
                                                apply_scaling): # <--- apply_scaling 인자 추가
    print("\n--- 5. Preparing and Splitting Data for Visitor Model Training ---")
    
    exclude_from_X = [date_col_name, target_col_name, ORIGINAL_RESCUE_COUNT_COL, ORIGINAL_RESCUE_EVENT_COL_FOR_FE,
                      'Accident_Cause_List', 'Accident_Outcome_List']
    print("--- Debug: exclude_from_X (inside prepare_and_split_data_for_visitor_training) ---")
    for item_idx, item_val in enumerate(exclude_from_X):
        print(f"Item {item_idx}: Value='{item_val}', Type={type(item_val)}")
    print("--- End Debug ---")
    potential_X_cols = [col for col in df_engineered.columns if col not in exclude_from_X]
    if not potential_X_cols: print("Error: No potential features for X."); exit()

    X_full = df_engineered[potential_X_cols].copy()
    y_full = df_engineered[target_col_name].copy()
    dates_full = df_engineered[date_col_name].copy()

    n_total = len(X_full)
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)

    X_train_raw = X_full.iloc[:n_train]
    y_train = y_full.iloc[:n_train]
    dates_train = dates_full.iloc[:n_train] # Defined here

    X_val_raw = X_full.iloc[n_train : n_train + n_val]
    y_val = y_full.iloc[n_train : n_train + n_val]
    dates_val = dates_full.iloc[n_train : n_train + n_val]

    X_test_raw = X_full.iloc[n_train + n_val:]
    y_test = y_full.iloc[n_train + n_val:]
    dates_test = dates_full.iloc[n_train + n_val:]
    print(f"Data split: Train {len(X_train_raw)}, Validation {len(X_val_raw)}, Test {len(X_test_raw)}")

    # Feature importance and selection (on training data)
    X_train_for_imp = X_train_raw.copy()
    imputer_values_train = {} # Store means/modes for selected features
    for col in X_train_for_imp.columns:
        if X_train_for_imp[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train_for_imp[col]):
                val_to_fill = X_train_for_imp[col].mean()
            else:
                val_to_fill = X_train_for_imp[col].mode()[0] if not X_train_for_imp[col].mode().empty else "Unknown"
            X_train_for_imp[col].fillna(val_to_fill, inplace=True)
            imputer_values_train[col] = val_to_fill # Store for all potential features initially

    rf_imp = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    X_numeric_for_imp = X_train_for_imp.select_dtypes(include=np.number)
    if X_numeric_for_imp.empty: print("Error: No numeric features for importance calculation."); exit()
    rf_imp.fit(X_numeric_for_imp, y_train)
    
    importances_df = pd.DataFrame({'Feature': X_numeric_for_imp.columns, 'Importance': rf_imp.feature_importances_})
    importances_df = importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    selected_features = importances_df.head(min(top_n, len(importances_df)))['Feature'].tolist()
    print(f"Top {len(selected_features)} features selected: {selected_features}")

    # Filter imputer_values for selected_features only
    imputer_means_selected_train = {feat: imputer_values_train[feat] for feat in selected_features if feat in imputer_values_train}

    X_train_sel = X_train_raw[selected_features].copy()
    X_val_sel = X_val_raw[selected_features].copy()
    X_test_sel = X_test_raw[selected_features].copy()

    X_train_filled = X_train_sel.fillna(imputer_means_selected_train)
    X_val_filled = X_val_sel.fillna(imputer_means_selected_train)
    X_test_filled = X_test_sel.fillna(imputer_means_selected_train)
    
    scaler_obj = None
    cols_to_scale = []
    
    if apply_scaling: # <--- 조건부 스케일링 시작
        print("  Applying scaling...")
        scaler_obj = MinMaxScaler()
        # 스케일링 대상 컬럼 식별 (이전 로직과 동일)
        binary_like_cols_visitor = [col for col in selected_features if X_train_filled[col].nunique(dropna=False) <= 2]
        cols_to_scale = [col for col in selected_features 
                         if col not in binary_like_cols_visitor and \
                            pd.api.types.is_numeric_dtype(X_train_filled[col])]
        
        if cols_to_scale:
            X_train_filled[cols_to_scale] = scaler_obj.fit_transform(X_train_filled[cols_to_scale])
            X_val_filled[cols_to_scale] = scaler_obj.transform(X_val_filled[cols_to_scale])
            X_test_filled[cols_to_scale] = scaler_obj.transform(X_test_filled[cols_to_scale])
            print(f"  Features scaled: {cols_to_scale}")
        else:
            print("  No features to scale.")
            scaler_obj = None # 스케일링 안 했으면 스케일러는 None
    else: # apply_scaling이 False인 경우
        print("  Scaling not applied as per experiment condition.")
        # scaler_obj는 이미 None으로 초기화됨
        # X_train_filled 등은 스케일링 안 된 상태로 유지

    return X_train_filled, X_val_filled, X_test_filled, \
           y_train, y_val, y_test, \
           dates_train, dates_val, dates_test, \
           selected_features, scaler_obj, imputer_means_selected_train, cols_to_scale

# --- 6. Model Training and Tuning (for Visitor Model) ---
def train_and_tune_visitor_model(X_train, y_train, X_val, y_val, param_dist, 
                                 n_iter, scorer, early_stopping_rounds, seed):
    print(f"\n--- 6. Training and Tuning Visitor Model (LGBMRegressor) ---")
    estimator = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1)
    fit_params = {}
    if X_val is not None and y_val is not None and not X_val.empty:
        fit_params['callbacks'] = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)]
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['eval_metric'] = 'rmse' 
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS_VISITOR)
    random_search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring=scorer, 
                                     cv=tscv, random_state=seed, n_jobs=-1, verbose=1)
    if fit_params: random_search.fit(X_train, y_train, **fit_params)
    else: random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV Score ({scorer}): {random_search.best_score_:.4f}")
    tuned_model = random_search.best_estimator_
    if hasattr(tuned_model, 'best_iteration_'): print(f"  Early stopping at iteration: {tuned_model.best_iteration_}")
    val_metrics = None
    if X_val is not None and y_val is not None and not X_val.empty:
        preds_val = tuned_model.predict(X_val)
        val_metrics = {'MAE_model_scale': mean_absolute_error(y_val, preds_val),
                       'RMSE_model_scale': np.sqrt(mean_squared_error(y_val, preds_val)),
                       'R2_model_scale': r2_score(y_val, preds_val)}
        print(f"Validation metrics (model scale): {val_metrics}")
    return tuned_model, random_search.best_params_, val_metrics

# --- 7. Model Evaluation and Saving (for Visitor Model) ---
def evaluate_and_save_visitor_model(model, X_val, y_val, X_test, y_test, 
                                    scaler_obj, imputer_means_obj, cols_scaled_list, 
                                    selected_features_list, best_params, 
                                    save_dir, input_file, ratios_str, 
                                    val_metrics_tuning, timestamp, 
                                    dates_val_vis, dates_test_vis, 
                                    apply_log_transform_for_eval):

    model_prefix = "visitor_lgbm_regressor" 
    print(f"\n--- 7. Evaluating and Saving {model_prefix.upper()} ---")
    y_val_eval = np.expm1(y_val) if apply_log_transform_for_eval else y_val
    y_test_eval = np.expm1(y_test) if apply_log_transform_for_eval else y_test
    preds_val_raw = model.predict(X_val); preds_test_raw = model.predict(X_test)
    preds_val_eval = np.expm1(preds_val_raw) if apply_log_transform_for_eval else preds_val_raw
    preds_test_eval = np.expm1(preds_test_raw) if apply_log_transform_for_eval else preds_test_raw
    val_r2_model_scale = r2_score(y_val, preds_val_raw)
    test_r2_model_scale = r2_score(y_test, preds_test_raw)
    val_mae_orig = mean_absolute_error(y_val_eval, preds_val_eval)
    val_rmse_orig = np.sqrt(mean_squared_error(y_val_eval, preds_val_eval))
    test_mae_orig = mean_absolute_error(y_test_eval, preds_test_eval)
    test_rmse_orig = np.sqrt(mean_squared_error(y_test_eval, preds_test_eval))
    print(f"Final Validation (Original Scale): MAE={val_mae_orig:.2f}, RMSE={val_rmse_orig:.2f}")
    print(f"Final Validation (Model Scale): R2={val_r2_model_scale:.4f}")
    print(f"Test Performance (Original Scale): MAE={test_mae_orig:.2f}, RMSE={test_rmse_orig:.2f}")
    print(f"Test Performance (Model Scale): R2={test_r2_model_scale:.4f}")
    if dates_val_vis is not None:
        plt.figure(figsize=(15,6))
        plt.plot(dates_val_vis, y_val_eval, label='Actual (Val)', marker='.')
        plt.plot(dates_val_vis, preds_val_eval, label='Predicted (Val)', marker='.')
        plt.title(f'Actual vs Predicted ({model_prefix} - Validation)'); plt.legend(); plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_prefix}_val_plot.png')); plt.close()
    if dates_test_vis is not None:
        plt.figure(figsize=(15,6))
        plt.plot(dates_test_vis, y_test_eval, label='Actual (Test)', marker='.')
        plt.plot(dates_test_vis, preds_test_eval, label='Predicted (Test)', marker='.')
        plt.title(f'Actual vs Predicted ({model_prefix} - Test)')
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_prefix}_test_plot.png'))
        plt.close()
    log_path = os.path.join(save_dir, f'{model_prefix}_evaluation_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp}\nInput File: {input_file}\nSplit Ratios: {ratios_str}\n")
        f.write(f"Best Hyperparameters: {best_params}\n")
        if val_metrics_tuning: f.write(f"Tuning Validation Metrics (model scale): {val_metrics_tuning}\n")
        f.write(f"Final Validation (Original Scale): MAE={val_mae_orig:.2f}, RMSE={val_rmse_orig:.2f}\n")
        f.write(f"Final Validation (Model Scale): R2={val_r2_model_scale:.4f}\n")
        f.write(f"Test Performance (Original Scale): MAE={test_mae_orig:.2f}, RMSE={test_rmse_orig:.2f}\n")
        f.write(f"Test Performance (Model Scale): R2={test_r2_model_scale:.4f}\n")
        f.write(f"Selected Features ({len(selected_features_list)}):\n" + "\n".join(selected_features_list))
    model_pkg_path = os.path.join(save_dir, f"{model_prefix}_best_model.pkl")
    imputer_means_to_save = imputer_means_obj.to_dict() if isinstance(imputer_means_obj, pd.Series) else imputer_means_obj
    model_package = {'model': model, 'scaler': scaler_obj, 'imputer_means': imputer_means_to_save,
                     'cols_scaled_at_fit': cols_scaled_list if cols_scaled_list else [], 
                     'features': selected_features_list, 'best_hyperparameters': best_params,
                     'model_type': type(model).__name__, 'training_timestamp': timestamp,
                     'test_metrics_original_scale': {'mae': test_mae_orig, 'rmse': test_rmse_orig},
                     'test_metrics_model_scale': {'r2': test_r2_model_scale}}
    joblib.dump(model_package, model_pkg_path)
    print(f"Visitor model package saved to: {model_pkg_path}")

# --- Main Execution ---
if __name__ == '__main__':
    all_experiment_results = [] # 모든 실험 결과를 저장할 리스트
    for condition_idx, exp_condition in enumerate(EXPERIMENT_CONDITIONS):
        print(f"\n\n========== Experiment {condition_idx + 1}: {exp_condition['name']} ==========")
        
        # --- 디렉토리 설정 (각 실험별 하위 디렉토리 생성) ---
        exp_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_base_dir = os.path.join("./visitor_model_experiments", exp_condition['name'] + "_" + exp_time_str)
        exp_viz_dir = os.path.join(exp_base_dir, "visualizations")
        os.makedirs(exp_base_dir, exist_ok=True)
        os.makedirs(exp_viz_dir, exist_ok=True)
        print(f"Experiment results will be saved in: {exp_base_dir}")

        # --- 데이터 로드 ---
        df_raw = load_data(INPUT_DATA_FILE, DATE_COLUMN)

        # --- 초기 전처리 (로그 변환) ---
        df_processed = initial_preprocess_visitor(
            df_raw, 
            TARGET_VISITOR_COL, 
            ORIGINAL_RESCUE_COUNT_COL, 
            ORIGINAL_RESCUE_EVENT_COL_FOR_FE,
            apply_log_transform=APPLY_LOG_TRANSFORM_VISITOR_TARGET 
        )
        
        # --- 이상치 처리 (조건에 따라) ---
        df_processed_for_fe = df_processed.copy()
        if exp_condition.get("apply_outlier_capping", False): # 기본값 False
            df_processed_for_fe = cap_outliers(
                df_processed.copy(), 
                TARGET_VISITOR_COL, 
                method=exp_condition.get("outlier_method", "iqr"), # 기본값 iqr
                iqr_multiplier=IQR_MULTIPLIER, # 전역 설정값 사용 또는 조건별 설정
                lower_p=PERCENTILE_LOWER,
                upper_p=PERCENTILE_UPPER
            )
            plot_distribution(df_processed_for_fe, TARGET_VISITOR_COL, f"Capped ({exp_condition.get('outlier_method', 'iqr')})", exp_viz_dir)
            plot_boxplot(df_processed_for_fe, TARGET_VISITOR_COL, f"Capped ({exp_condition.get('outlier_method', 'iqr')})", exp_viz_dir)
        
        # --- 피처 엔지니어링 ---
        df_engineered = engineer_features_visitor(
            df_processed_for_fe, 
            DATE_COLUMN, 
            TARGET_VISITOR_COL, 
            ORIGINAL_RESCUE_EVENT_COL_FOR_FE
        )
    
        # --- 데이터 분할, 특성 선택, 스케일링 ---
        # prepare_and_split_data_for_visitor_training 함수는 스케일링 적용 여부를 내부적으로 결정함
        apply_scaling = exp_condition.get("apply_scaling")
        
        X_train_prep, X_val_prep, X_test_prep, \
        y_train, y_val, y_test, \
        _, dates_val_vis, dates_test_vis, \
        selected_features, scaler_obj, imputer_means, cols_scaled = prepare_and_split_data_for_visitor_training(
            df_engineered, TARGET_VISITOR_COL, DATE_COLUMN, 
            TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, TOP_N_FEATURES_VISITOR,
            apply_scaling # 함수에 스케일링 적용 여부 전달
        )
        
        X_train, X_val, X_test = X_train_prep, X_val_prep, X_test_prep
        if not exp_condition.get("apply_scaling", True) and scaler_obj is not None and cols_scaled:
            print("  Skipping scaling as per experiment condition. Using unscaled (but imputed and selected) data.")
            current_scaler_to_save = None if not exp_condition.get("apply_scaling", True) else scaler_obj
            current_cols_scaled = [] if not exp_condition.get("apply_scaling", True) else cols_scaled
        else:
            current_scaler_to_save = scaler_obj
            current_cols_scaled = cols_scaled


        # --- 모델 학습 및 튜닝 ---
        model_to_train_type = exp_condition.get("model_type", "lgbm") # 기본 lgbm
        
        if model_to_train_type == "lgbm":
            param_dist = { # LightGBM용 파라미터 분포
                'n_estimators': sp_randint(400, 2000), 'learning_rate': sp_uniform(0.005, 0.095),
                'max_depth': sp_randint(5, 15), 'num_leaves': sp_randint(15, 100),
                'min_child_samples': sp_randint(10, 101), 'subsample': sp_uniform(0.6, 0.4),
                'colsample_bytree': sp_uniform(0.4, 0.6), 'reg_alpha': sp_uniform(0, 5),
                'reg_lambda': sp_uniform(0, 5)
            }
            trained_model, best_params, val_metrics = train_and_tune_visitor_model(
                X_train, y_train, X_val, y_val, 
                param_dist, 
                N_ITER_RANDOM_SEARCH_VISITOR, 
                SCORER_VISITOR,
                EARLY_STOPPING_ROUNDS_LGBM,
                RANDOM_SEED
            )
        elif model_to_train_type == "rf":
            estimator = RandomForestRegressor(random_state=seed, n_jobs=-1)
            param_dist = {
                'n_estimators': sp_randint(100, 800),
                'max_depth': [None] + list(sp_randint(5, 25).rvs(5)), # None and a few random depths
                'min_samples_split': sp_randint(2, 20),
                'min_samples_leaf': sp_randint(1, 20),
                'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9] # Common values for max_features
            }
            # RandomForestRegressor doesn't have direct early stopping in fit_params for RandomizedSearchCV
            # Early stopping can be approximated by limiting n_estimators or through more complex callback setups (not standard)

        elif model_to_train_type == "xgb":
            estimator = xgb.XGBRegressor(random_state=seed, n_jobs=-1, objective='reg:squarederror', eval_metric='rmse')
            param_dist = {
                'n_estimators': sp_randint(100, 1000),
                'learning_rate': sp_uniform(0.01, 0.2),
                'max_depth': sp_randint(3, 10),
                'min_child_weight': sp_randint(1, 10),
                'subsample': sp_uniform(0.6, 0.4), # loc=0.6, scale=0.4 -> [0.6, 1.0]
                'colsample_bytree': sp_uniform(0.6, 0.4), # loc=0.6, scale=0.4 -> [0.6, 1.0]
                'gamma': sp_uniform(0, 0.5),
                'reg_alpha': sp_uniform(0, 2), # XGBoost calls it alpha
                'reg_lambda': sp_uniform(0, 2)  # XGBoost calls it lambda
            }
            if X_val is not None and y_val is not None and not X_val.empty:
                fit_params['early_stopping_rounds'] = early_stopping_rounds
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = False # Suppress XGBoost training verbosity during search
            else:
                print(f"Unsupported model type: {model_to_train_type}. Skipping experiment.")
                continue

        # --- 모델 평가 및 저장 ---
        evaluate_and_save_visitor_model(
            trained_model, X_val, y_val, X_test, y_test, 
            current_scaler_to_save, # 조건부 스케일러
            imputer_means, 
            current_cols_scaled,    # 조건부 스케일된 컬럼 목록
            selected_features, best_params, 
            exp_base_dir, # 각 실험별 디렉토리에 저장
            INPUT_DATA_FILE,
            f"{TRAIN_RATIO*100:.0f}%/{VALIDATION_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}%",
            val_metrics, 
            exp_time_str, # 실험 시작 시간
            dates_val_vis, dates_test_vis,
            apply_log_transform_for_eval=APPLY_LOG_TRANSFORM_VISITOR_TARGET
        )
        
        # 실험 결과 수집        
        saved_model_filename = f"visitor_lgbm_regressor_best_model.pkl" # evaluate_and_save_visitor_model에서 사용한 파일명 규칙
        saved_model_pkg_path = os.path.join(exp_base_dir, saved_model_filename) 
        
        if os.path.exists(saved_model_pkg_path):
            loaded_pkg_for_results = joblib.load(saved_model_pkg_path)
            test_mae = loaded_pkg_for_results.get('test_metrics_original_scale', {}).get('mae', np.nan)
            test_rmse = loaded_pkg_for_results.get('test_metrics_original_scale', {}).get('rmse', np.nan)
            test_r2 = loaded_pkg_for_results.get('test_metrics_model_scale', {}).get('r2', np.nan)
            
            all_experiment_results.append({
                "Experiment_Name": exp_condition['name'],
                "Test_MAE_Orig": test_mae,
                "Test_RMSE_Orig": test_rmse,
                "Test_R2_Model_Scale": test_r2,
                "Best_Params": best_params # train_and_tune_visitor_model에서 반환된 값
            })
        else:
            print(f"Warning: Could not find saved model package at '{saved_model_pkg_path}' to retrieve test metrics.")
            # 패키지를 찾지 못하면 빈 값으로 추가하거나, 해당 실험 결과를 기록하지 않을 수 있음
            all_experiment_results.append({
                "Experiment_Name": exp_condition['name'],
                "Test_MAE_Orig": np.nan,
                "Test_RMSE_Orig": np.nan,
                "Test_R2_Model_Scale": np.nan,
                "Best_Params": best_params
            })

    # --- 모든 실험 결과 비교 ---
    if all_experiment_results:
        results_summary_df = pd.DataFrame(all_experiment_results)
        print("\n\n========== All Experiment Results Summary ==========")
        print(results_summary_df.to_string()) # 모든 행과 열을 보기 위해 to_string() 사용
        
        # 요약 결과 저장
        summary_file_path = os.path.join(BASE_RESULTS_DIR, f"experiment_summary_{CURRENT_TIME_STR}.csv")
        results_summary_df.to_csv(summary_file_path, index=False, encoding='utf-8-sig')
        print(f"\nExperiment summary saved to: {summary_file_path}")

        # 간단한 시각화 (예: Test R2 비교)
        if 'Test_R2_Model_Scale' in results_summary_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Experiment_Name', y='Test_R2_Model_Scale', data=results_summary_df.sort_values('Test_R2_Model_Scale', ascending=False))
            plt.title('Comparison of Test R2 Scores by Experiment')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_RESULTS_DIR, f"experiment_r2_comparison_{CURRENT_TIME_STR}.png")); plt.close()
            print("Experiment R2 comparison plot saved.")
                
        print("\n========== All Experiments Complete ==========")