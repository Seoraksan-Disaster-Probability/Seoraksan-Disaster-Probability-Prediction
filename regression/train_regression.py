import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # For feature importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit # GridSearchCV는 현재 사용 안 함
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
# File Paths
INPUT_DATA_FILE = "/home/imes-server2/sunmin/termP/data/merged_data_with_seorakdong_visitors.csv"

# Column Names
DATE_COLUMN = 'Date'
TARGET_VISITOR_COL = 'Total_Visitor_Count' # 탐방객 모델의 타겟이자, FE시 Lag 생성 기준
ORIGINAL_RESCUE_COUNT_COL = 'Total_Rescued_Count'  # 원본 데이터의 구조 인원 컬럼 (타겟 생성 또는 FE에 사용)
ORIGINAL_RESCUE_EVENT_COL_FOR_FE = 'Rescue_Event' # FE 시 사용되는 원본 조난 발생 유무 컬럼명

# Data Splitting Ratios
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9, "Split ratios must sum to 1."

# Feature Selection
TOP_N_FEATURES_VISITOR = 20 # 탐방객 모델에서 사용할 상위 피처 개수

# Preprocessing Flags
APPLY_LOG_TRANSFORM_VISITOR_TARGET = True # 탐방객 모델 타겟 로그 변환 여부
APPLY_OUTLIER_CAPPING_VISITOR_TARGET = True # 탐방객 모델 타겟 이상치 처리(Capping) 여부
OUTLIER_CAPPING_METHOD = 'iqr' # 'iqr' 또는 'percentile'
IQR_MULTIPLIER = 1.5
PERCENTILE_LOWER = 1
PERCENTILE_UPPER = 99

# Model Training Parameters
N_ITER_RANDOM_SEARCH_VISITOR = 50 # RandomizedSearchCV 반복 횟수
RANDOM_SEED = 42 # 결과 재현을 위한 시드값
CV_N_SPLITS_VISITOR = 3 # TimeSeriesSplit 폴드 수
SCORER_VISITOR = 'r2' # RandomizedSearchCV 평가 지표
EARLY_STOPPING_ROUNDS_LGBM = 50

# Directory Setup
CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_RESULTS_DIR = "./visitor_model_training_results_refactored" # 결과 저장 기본 디렉토리명 변경
RUN_SPECIFIC_DIR = os.path.join(BASE_RESULTS_DIR, CURRENT_TIME_STR)
VISUALIZATION_DIR = os.path.join(RUN_SPECIFIC_DIR, "visualizations")
os.makedirs(RUN_SPECIFIC_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
print(f"All results will be saved in: {RUN_SPECIFIC_DIR}")


# --- 1. Data Handling and Initial Preprocessing ---
def load_data(file_path, date_col):
    # ... (함수 내용 동일) ...
    print(f"\n--- 1. Loading Data from {file_path} ---")
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
        df.sort_values(by=date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Data loaded and sorted.")
        return df
    except FileNotFoundError: print(f"Error: File not found at {file_path}"); exit()
    except Exception as e: print(f"Error loading data: {e}"); exit()

def initial_preprocess_visitor(df, target_col, rescue_count_col, rescue_event_col, apply_log_transform):
    # ... (함수 내용 동일, TARGET_VISITOR_COL 대신 target_col 사용) ...
    print(f"\n--- 2. Initial Preprocessing for Visitor Model (Log Transform: {apply_log_transform}) ---")
    df_p = df.copy()
    if target_col not in df_p.columns: print(f"Error: Target column '{target_col}' not found."); exit()
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
    print("Initial preprocessing for visitor model complete.")
    return df_p

# --- 2a. Outlier Capping ---
def cap_outliers(df, column_name, method='iqr', iqr_multiplier=1.5, lower_p=1, upper_p=99):
    print(f"\n--- 2a. Applying Outlier Capping on '{column_name}' using {method} method ---")
    df_c = df.copy()
    if method == 'iqr':
        Q1 = df_c[column_name].quantile(0.25); Q3 = df_c[column_name].quantile(0.75)
        IQR_value = Q3 - Q1
        lower_b = Q1 - iqr_multiplier * IQR_value; upper_b = Q3 + iqr_multiplier * IQR_value
        capped_low = (df_c[column_name] < lower_b).sum()
        capped_high = (df_c[column_name] > upper_b).sum()
        df_c[column_name] = np.where(df_c[column_name] < lower_b, lower_b, df_c[column_name])
        df_c[column_name] = np.where(df_c[column_name] > upper_b, upper_b, df_c[column_name])
        print(f"  IQR Capping: {capped_low} low, {capped_high} high values capped.")
    elif method == 'percentile':
        lower_b = df_c[column_name].quantile(lower_p / 100)
        upper_b = df_c[column_name].quantile(upper_p / 100)
        capped_low = (df_c[column_name] < lower_b).sum()
        capped_high = (df_c[column_name] > upper_b).sum()
        df_c[column_name] = np.where(df_c[column_name] < lower_b, lower_b, df_c[column_name])
        df_c[column_name] = np.where(df_c[column_name] > upper_b, upper_b, df_c[column_name])
        print(f"  Percentile Capping ({lower_p}%-{upper_p}%): {capped_low} low, {capped_high} high values capped.")
    else:
        print(f"  Warning: Unknown outlier capping method '{method}'. No capping applied.")
    return df_c

# --- 3. Visualization Functions ---
# (plot_target_distribution, plot_target_boxplot, plot_target_timeseries - 이전과 동일)
def plot_target_distribution(df, target_col, stage_name, save_dir):
    plt.figure(figsize=(10, 6)); sns.histplot(df[target_col], kde=True, bins=50)
    plt.title(f'Distribution of {target_col} ({stage_name})')
    plt.savefig(os.path.join(save_dir, f'{target_col}_dist_{stage_name}.png')); plt.close()

def plot_target_boxplot(df, target_col, stage_name, save_dir):
    plt.figure(figsize=(8, 6)); sns.boxplot(y=df[target_col])
    plt.title(f'Box Plot of {target_col} ({stage_name})')
    plt.savefig(os.path.join(save_dir, f'{target_col}_boxplot_{stage_name}.png')); plt.close()

def plot_target_timeseries(df, date_col, target_col, stage_name, save_dir):
    if date_col in df.columns:
        plt.figure(figsize=(15, 6)); plt.plot(df[date_col], df[target_col], marker='.', linestyle='-')
        plt.title(f'Time Series of {target_col} ({stage_name})')
        plt.xlabel(date_col); plt.ylabel(target_col); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{target_col}_timeseries_{stage_name}.png')); plt.close()

# --- 4. Feature Engineering Function (for Visitor Model) ---
def engineer_features_visitor(df_input, date_col, 
                              actual_visitor_col_for_lag, 
                              actual_rescue_event_col_for_lag):
    # ... (함수 내용 이전과 동일, 단 TARGET_VISITOR_COL 대신 actual_visitor_col_for_lag 사용 주의) ...
    # Lag/Rolling 생성 시 TARGET_VISITOR_COL (전역 변수) 대신 actual_visitor_col_for_lag 사용
    print("\n--- 4. Engineering Features for Visitor Model ---")
    df_eng = df_input.copy()
    df_eng['temp_day_of_week'] = df_eng[date_col].dt.dayofweek
    df_eng['temp_month'] = df_eng[date_col].dt.month
    df_eng['is_weekend'] = (df_eng['temp_day_of_week'] >= 5).astype(int)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['temp_month'] / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['temp_month'] / 12)
    month_onehot = pd.get_dummies(df_eng['temp_month'], prefix='month', dtype=int)
    df_eng['month_10'] = month_onehot['month_10'] if 'month_10' in month_onehot.columns else 0
    min_year = df_eng[date_col].dt.year.min(); max_year = df_eng[date_col].dt.year.max()
    if pd.isna(min_year) or pd.isna(max_year): current_year = df_eng[date_col].dt.year.iloc[0]; kr_holidays = holidays.KR(years=current_year)
    else: kr_holidays = holidays.KR(years=range(min_year, max_year + 1))
    df_eng['is_official_holiday'] = df_eng[date_col].apply(lambda date: date in kr_holidays)
    df_eng['is_day_off_official'] = (df_eng['is_weekend'] | df_eng['is_official_holiday']).astype(int)
    df_eng['day_off_group'] = (df_eng['is_day_off_official'].diff(1) != 0).astype(int).cumsum()
    df_eng['consecutive_official_days_off'] = df_eng.groupby('day_off_group')['is_day_off_official'].transform('sum')
    long_holiday_threshold = 3
    df_eng['is_base_long_holiday'] = ((df_eng['is_day_off_official'] == 1) & (df_eng['consecutive_official_days_off'] >= long_holiday_threshold)).astype(int)
    df_eng['prev_day_is_off'] = df_eng['is_day_off_official'].shift(1).fillna(0).astype(int)
    df_eng['next_day_is_off'] = df_eng['is_day_off_official'].shift(-1).fillna(0).astype(int)
    df_eng['is_bridge_day_candidate'] = ((df_eng['is_day_off_official'] == 0) & (df_eng['prev_day_is_off'] == 1) & (df_eng['next_day_is_off'] == 1)).astype(int)
    df_eng['is_extended_holiday'] = df_eng['is_base_long_holiday']
    df_eng.loc[df_eng['is_bridge_day_candidate'] == 1, 'is_extended_holiday'] = 1
    df_eng.loc[df_eng['is_bridge_day_candidate'].shift(-1).fillna(False), 'is_extended_holiday'] = 1
    df_eng.loc[df_eng['is_bridge_day_candidate'].shift(1).fillna(False), 'is_extended_holiday'] = 1
    df_eng['final_holiday_group'] = (df_eng['is_extended_holiday'].diff(1) != 0).astype(int).cumsum()
    df_eng['consecutive_extended_days_off'] = df_eng.groupby('final_holiday_group')['is_extended_holiday'].transform('sum')
    df_eng['is_final_long_holiday'] = ((df_eng['is_extended_holiday'] == 1) & (df_eng['consecutive_extended_days_off'] >= long_holiday_threshold)).astype(int)
    lag_values = [1, 7, 14, 30]; rolling_windows = [7, 14, 30]
    # Use actual_visitor_col_for_lag for creating lag/rolling features
    if actual_visitor_col_for_lag and actual_visitor_col_for_lag in df_eng.columns:
        for lag in lag_values: df_eng[f'{actual_visitor_col_for_lag}_Lag{lag}'] = df_eng[actual_visitor_col_for_lag].shift(lag).fillna(0)
        for window in rolling_windows: df_eng[f'{actual_visitor_col_for_lag}_Roll{window}_Mean'] = df_eng[actual_visitor_col_for_lag].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
    else: # Fallback if the specific column for lag isn't present (e.g., future prediction)
        for lag in lag_values: df_eng[f'{TARGET_VISITOR_COL}_Lag{lag}'] = 0 # Use global TARGET_VISITOR_COL for consistency in naming
        for window in rolling_windows: df_eng[f'{TARGET_VISITOR_COL}_Roll{window}_Mean'] = 0
    if actual_rescue_event_col_for_lag and actual_rescue_event_col_for_lag in df_eng.columns:
        df_eng['rescue_event_yesterday'] = df_eng[actual_rescue_event_col_for_lag].shift(1).fillna(0)
    else: df_eng['rescue_event_yesterday'] = 0
    weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'), 'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    for condition, (col, window, new_col_name) in weather_rules.items():
        if col in df_eng.columns:
            is_condition = (df_eng[col] > 0 if condition == 'rain' else df_eng[col] < 0).astype(int)
            df_eng[new_col_name] = (is_condition.rolling(window=window, min_periods=window).sum() == window).astype(int).fillna(0)
    time_cols_map = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    for orig_col, new_col in time_cols_map.items():
        if orig_col in df_eng.columns:
            try: df_eng[new_col] = pd.to_datetime(df_eng[orig_col], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
            except: df_eng[new_col] = -1
    temp_col, hum_col = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    if temp_col in df_eng.columns and hum_col in df_eng.columns:
        df_eng['Temp_Humidity_Interaction'] = df_eng[temp_col].fillna(df_eng[temp_col].mean()) * df_eng[hum_col].fillna(df_eng[hum_col].mean())
    temp_cols_to_drop = ['temp_day_of_week', 'temp_month', 'is_official_holiday', 'is_day_off_official', 'day_off_group', 
                         'consecutive_official_days_off', 'is_base_long_holiday', 'prev_day_is_off', 'next_day_is_off', 
                         'is_bridge_day_candidate', 'is_extended_holiday', 'final_holiday_group', 
                         'consecutive_extended_days_off', 'TimeOfMaxTempC', 'TimeOfMinTempC']
    df_eng.drop(columns=[col for col in temp_cols_to_drop if col in df_eng.columns], inplace=True, errors='ignore')
    print("Feature engineering for visitor model complete.")
    return df_eng
# prepare_and_split_data_for_visitor_training 함수 수정
def prepare_and_split_data_for_visitor_training(df_engineered_input, 
                                                target_col_name_str, 
                                                date_col_name_str,   
                                                train_r, val_r, test_r, top_n):
    print("\n--- 5. Preparing and Splitting Data for Visitor Model Training ---")
    
    explicit_exclude_from_X = [
        date_col_name_str, target_col_name_str, ORIGINAL_RESCUE_COUNT_COL, 
        ORIGINAL_RESCUE_EVENT_COL_FOR_FE, 'Accident_Cause_List', 'Accident_Outcome_List'
    ]
    potential_features_for_X = [col for col in df_engineered_input.columns if col not in explicit_exclude_from_X]

    if not potential_features_for_X:
        print("오류: X를 구성할 피처가 없습니다. explicit_exclude_from_X 목록을 확인하세요."); exit()

    try:
        X_full = df_engineered_input[potential_features_for_X].copy()
        y_full = df_engineered_input[target_col_name_str].copy()
        dates_full = df_engineered_input[date_col_name_str].copy() # dates_full 정의
    except KeyError as e:
        # ... (에러 처리) ...
        exit()

    # 데이터 분할
    n_total = len(X_full)
    n_train = int(n_total * train_r); n_val = int(n_total * val_r)
    
    X_train_raw, y_train, dates_train = X_full.iloc[:n_train], y_full.iloc[:n_train], dates_full.iloc[:n_train] # <--- dates_train 할당
    X_val_raw, y_val, dates_val = X_full.iloc[n_train:n_train+n_val], y_full.iloc[n_train:n_train+n_val], dates_full.iloc[n_train:n_train+n_val]
    X_test_raw, y_test, dates_test = X_full.iloc[n_train+n_val:], y_full.iloc[n_train+n_val:], dates_full.iloc[n_train+n_val:]
    print(f"Data split: Train {len(X_train_raw)}, Validation {len(X_val_raw)}, Test {len(X_test_raw)}")

    # ... (이하 특성 중요도 계산, 결측치 처리, 스케일링 로직 동일) ...
    # (이전 답변의 나머지 함수 내용 복사)
    X_train_for_imp = X_train_raw.copy()
    imputer_means_train_full_X = {} 
    for col in X_train_for_imp.columns:
        if X_train_for_imp[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train_for_imp[col]):
                mean_val = X_train_for_imp[col].mean()
                X_train_for_imp[col].fillna(mean_val, inplace=True)
                imputer_means_train_full_X[col] = mean_val
            else: 
                mode_val = X_train_for_imp[col].mode()[0] if not X_train_for_imp[col].mode().empty else "Unknown"
                X_train_for_imp[col].fillna(mode_val, inplace=True)
                imputer_means_train_full_X[col] = mode_val
    rf_imp = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    X_numeric_for_imp = X_train_for_imp.select_dtypes(include=np.number)
    if X_numeric_for_imp.empty: print("오류: 특성 중요도 계산을 위한 숫자형 피처가 없습니다."); exit()
    rf_imp.fit(X_numeric_for_imp, y_train)
    importances = pd.DataFrame({'Feature': X_numeric_for_imp.columns, 'Importance': rf_imp.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    selected_features = importances.head(min(top_n, len(importances)))['Feature'].tolist()
    print(f"Top {len(selected_features)} features selected: {selected_features}")
    X_train_sel = X_train_raw[selected_features].copy(); X_val_sel = X_val_raw[selected_features].copy(); X_test_sel = X_test_raw[selected_features].copy()
    imputer_means_selected_train = {k: v for k, v in imputer_means_train_full_X.items() if k in selected_features}
    X_train_filled = X_train_sel.fillna(imputer_means_selected_train); X_val_filled = X_val_sel.fillna(imputer_means_selected_train); X_test_filled = X_test_sel.fillna(imputer_means_selected_train)
    scaler = MinMaxScaler()
    binary_like_cols_visitor = [col for col in selected_features if X_train_filled[col].nunique(dropna=False) <= 2]
    cols_to_scale = [col for col in selected_features if col not in binary_like_cols_visitor and pd.api.types.is_numeric_dtype(X_train_filled[col])]
    if cols_to_scale:
        X_train_filled[cols_to_scale] = scaler.fit_transform(X_train_filled[cols_to_scale])
        X_val_filled[cols_to_scale] = scaler.transform(X_val_filled[cols_to_scale])
        X_test_filled[cols_to_scale] = scaler.transform(X_test_filled[cols_to_scale])
        print(f"Features scaled: {cols_to_scale}")
    else: scaler = None; cols_to_scale = []
    # --- 수정된 반환값 ---
    return X_train_filled, X_val_filled, X_test_filled, \
           y_train, y_val, y_test, \
           dates_train, dates_val, dates_test, \
           selected_features, scaler, imputer_means_selected_train, cols_to_scale
           
           
# --- 6. Model Training and Tuning (for Visitor Model) ---
def train_and_tune_visitor_model(X_train, y_train, X_val, y_val, param_dist, n_iter, scorer, early_stopping_rounds, seed):
    # ... (함수 내용 이전과 동일, early_stopping_rounds 인자 사용) ...
    print(f"\n--- 6. Training and Tuning Visitor Model (LGBMRegressor) ---")
    estimator = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1)
    fit_params = {}
    if X_val is not None and y_val is not None and not X_val.empty:
        fit_params['callbacks'] = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)]
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['eval_metric'] = 'rmse' 
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS_VISITOR) # 전역 변수 사용
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
    # ... (함수 내용 이전과 동일, model_prefix="visitor_lgbm_regressor" 사용) ...
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
        plt.figure(figsize=(15,6)); plt.plot(dates_val_vis, y_val_eval, label='Actual (Val)', marker='.'); 
        plt.plot(dates_val_vis, preds_val_eval, label='Predicted (Val)', marker='.'); plt.title(f'Actual vs Predicted ({model_prefix} - Validation)'); plt.legend(); plt.grid();
        plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{model_prefix}_val_plot.png')); plt.close()
    if dates_test_vis is not None:
        plt.figure(figsize=(15,6)); plt.plot(dates_test_vis, y_test_eval, label='Actual (Test)', marker='.');
        plt.plot(dates_test_vis, preds_test_eval, label='Predicted (Test)', marker='.'); plt.title(f'Actual vs Predicted ({model_prefix} - Test)'); plt.legend(); plt.grid();
        plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{model_prefix}_test_plot.png')); plt.close()
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
    df_raw = load_data(INPUT_DATA_FILE, DATE_COLUMN)

    # Visualizations for raw target
    plot_target_distribution(df_raw, TARGET_VISITOR_COL, "raw", VISUALIZATION_DIR)
    plot_target_boxplot(df_raw, TARGET_VISITOR_COL, "raw", VISUALIZATION_DIR)
    
    # Initial preprocessing (log transform)
    df_processed = initial_preprocess_visitor(
        df_raw, 
        TARGET_VISITOR_COL, 
        ORIGINAL_RESCUE_COUNT_COL, 
        ORIGINAL_RESCUE_EVENT_COL_FOR_FE,
        apply_log_transform=APPLY_LOG_TRANSFORM_VISITOR_TARGET
    )
    
    if APPLY_LOG_TRANSFORM_VISITOR_TARGET:
        plot_target_distribution(df_processed, TARGET_VISITOR_COL, "log_transformed", VISUALIZATION_DIR)
        plot_target_boxplot(df_processed, TARGET_VISITOR_COL, "log_transformed", VISUALIZATION_DIR)
    
    plot_target_timeseries(df_processed, DATE_COLUMN, TARGET_VISITOR_COL, 
                           "processed_log_transformed" if APPLY_LOG_TRANSFORM_VISITOR_TARGET else "processed", 
                           VISUALIZATION_DIR)

    # Outlier capping (if enabled)
    if APPLY_OUTLIER_CAPPING_VISITOR_TARGET:
        df_processed_final = cap_outliers( # Renamed function
            df_processed.copy(), 
            TARGET_VISITOR_COL, 
            method=OUTLIER_CAPPING_METHOD,
            iqr_multiplier=IQR_MULTIPLIER,
            lower_p=PERCENTILE_LOWER,
            upper_p=PERCENTILE_UPPER
        )
        plot_target_distribution(df_processed_final, TARGET_VISITOR_COL, f"capped_{OUTLIER_CAPPING_METHOD}", VISUALIZATION_DIR)
        plot_target_boxplot(df_processed_final, TARGET_VISITOR_COL, f"capped_{OUTLIER_CAPPING_METHOD}", VISUALIZATION_DIR)
    else:
        df_processed_final = df_processed.copy()


    # Feature Engineering
    df_engineered = engineer_features_visitor(
        df_processed_final, # Use capped data if outlier capping was applied
        DATE_COLUMN, 
        TARGET_VISITOR_COL, # For Lag/Rolling from (log-transformed, possibly capped) target
        ORIGINAL_RESCUE_EVENT_COL_FOR_FE
    )
    
# prepare_and_split_data_for_visitor_training 함수 호출 시
    # df_engineered 전체와 함께 문자열 컬럼명 전달
    X_train, X_val, X_test, y_train, y_val, y_test, \
    _, dates_val_vis, dates_test_vis, \
    selected_features, scaler, imputer_means, cols_scaled = prepare_and_split_data_for_visitor_training(
        df_engineered,
        TARGET_VISITOR_COL,
        DATE_COLUMN,
        TRAIN_RATIO, 
        VALIDATION_RATIO, 
        TEST_RATIO, 
        TOP_N_FEATURES_VISITOR
    )
    
    # Define parameter distribution for RandomizedSearchCV
    lgbm_param_dist = {
        'n_estimators': sp_randint(400, 2000), 
        'learning_rate': sp_uniform(0.005, 0.095),
        'max_depth': sp_randint(5, 15), 
        'num_leaves': sp_randint(15, 100),
        'min_child_samples': sp_randint(10, 101), 
        'subsample': sp_uniform(0.6, 0.4),
        'colsample_bytree': sp_uniform(0.4, 0.6), # Corrected range
        'reg_alpha': sp_uniform(0, 5),
        'reg_lambda': sp_uniform(0, 5)
    }

    print("\n\n========== LightGBM Visitor Model Training ==========")
    lgbm_model, best_params, val_metrics = train_and_tune_visitor_model(
        X_train, y_train, X_val, y_val, 
        lgbm_param_dist, 
        N_ITER_RANDOM_SEARCH_VISITOR, 
        SCORER_VISITOR,
        EARLY_STOPPING_ROUNDS_LGBM, # Pass the global config
        RANDOM_SEED # Pass the global config
    )
    
    evaluate_and_save_visitor_model(
        lgbm_model, X_val, y_val, X_test, y_test, scaler, imputer_means, cols_scaled,
        selected_features, best_params, RUN_SPECIFIC_DIR, INPUT_DATA_FILE,
        f"{TRAIN_RATIO*100:.0f}%/{VALIDATION_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}%",
        val_metrics, CURRENT_TIME_STR, dates_val_vis, dates_test_vis,
        apply_log_transform_for_eval=APPLY_LOG_TRANSFORM_VISITOR_TARGET
    )
    
    print("\n========== Visitor Model Training Pipeline Complete ==========")