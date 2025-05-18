import os
import joblib
import holidays
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import ( # 분류 평가지표
    mean_absolute_error, 
    r2_score,
    classification_report, 
    roc_auc_score, 
    accuracy_score, 
    f1_score,
    precision_recall_curve,
    confusion_matrix
) 

# --- 0. Configuration ---
# File Paths
INPUT_DATA_FILE = "data/preprocessed_data.csv"
VISITOR_MODEL_PKG_PATH = "regression/visitor_lgbm_regressor_best_model.pkl"
RESCUE_MODEL_PKG_PATH = "classification/randomforest_best_model.pkl" # 파일명 확인: randomforest_best_mode.pkl -> randomforest_best_model.pkl

# Column Names
DATE_COLUMN = 'Date'
# Visitor Model Related
TARGET_VISITOR_COL = 'Total_Visitor_Count' # 탐방객 모델의 타겟이자, FE시 Lag 생성 기준
# Rescue Model Related
# Common Original Columns (used in FE for one or both models)
ORIGINAL_RESCUE_COUNT_COL = 'Total_Rescued_Count'
ORIGINAL_RESCUE_EVENT_COL_FOR_FE = 'Rescue_Event' # FE 시 사용되는 원본 조난 발생 유무 컬럼명

# Prediction Output Column Name
PREDICTED_VISITOR_COL_NAME = 'Predicted_Total_Visitor_Count'

# Preprocessing Flags
APPLY_LOG_TRANSFORM_VISITOR_TARGET = True # 탐방객 모델 타겟 로그 변환 여부

# Preprocessing Flags
APPLY_LOG_TRANSFORM_VISITOR_TARGET = True
APPLY_OUTLIER_CAPPING_VISITOR_TARGET = True  # <--- 이 변수를 추가 또는 주석 해제하세요!
OUTLIER_CAPPING_METHOD = 'iqr'             # 이상치 처리 방법: 'iqr' 또는 'percentile'
IQR_MULTIPLIER = 1.5                       # IQR 방식 사용 시 multiplier
PERCENTILE_LOWER = 1                       # 백분위수 방식 사용 시 하위 퍼센트
PERCENTILE_UPPER = 99                      # 백분위수 방식 사용 시 상위 퍼센트

# test prediction range
TEST_PREDICTION_RANGE = 365 # 데이터의 마지막으로 부터 해당일 만큼 테스트로 예측

# --- Directory Setup ---
CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_RESULTS_DIR = "./integrated_prediction_results"
RUN_SPECIFIC_DIR = os.path.join(BASE_RESULTS_DIR, CURRENT_TIME_STR)
os.makedirs(RUN_SPECIFIC_DIR, exist_ok=True)
print(f"All results will be saved in: {RUN_SPECIFIC_DIR}")

# --- 1. Utility Functions ---
def load_data(file_path, date_col):
    print(f"\n--- Loading data from {file_path} ---")
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

def load_model_package(pkg_path):
    print(f"Loading model package: {pkg_path}")
    try:
        model_package = joblib.load(pkg_path)
        print(f"  Model ({model_package.get('model_type', 'N/A')}) loaded.")
        return model_package
    except FileNotFoundError:
        print(f"Error: Model package not found at {pkg_path}"); exit()
    except Exception as e:
        print(f"Error loading model package: {e}"); exit()

def get_last_n_days_data(df, n_days=14):
    if len(df) < n_days:
        print(f"Warning: Full data has less than {n_days} days. Returning all available data.")
        return df.copy()
    return df.tail(n_days).copy()

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

# --- 2. Preprocessing and Feature Engineering Functions ---
def initial_preprocess_for_visitor(df, target_col, rescue_count_col, rescue_event_col, apply_log_transform):
    print(f"\n--- Initial Preprocessing for Visitor Model (Log Transform: {apply_log_transform}) ---")
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

def engineer_features_for_visitor_model(df_input, date_col, 
                                        actual_visitor_col_for_lag, # 학습/과거 예측 시 실제 타겟값, 미래 예측 시 None
                                        actual_rescue_event_col_for_lag): # 학습/과거 예측 시 실제 조난값, 미래 예측 시 None
    print("  Feature Engineering for Visitor Model...")
    df_eng = df_input.copy()
    df_eng.reset_index(drop=True, inplace=True) # <--- 인덱스 리셋 추가!

    # --- PASTE THE EXACT FEATURE ENGINEERING LOGIC FROM VISITOR MODEL TRAINING SCRIPT HERE ---
    # Example (ensure this matches your training script's engineer_features function):
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
    # 징검다리 휴일 및 그로 인해 연결되는 휴일/주말을 is_extended_holiday에 포함
    # 징검다리 휴일(평일) 자체를 연휴로 표시
    # .values를 사용하여 불리언 값 배열을 전달
    df_eng.loc[df_eng['is_bridge_day_candidate'].values == 1, 'is_extended_holiday'] = 1
    
    # 징검다리 휴일의 전날(휴일/주말)과 다음날(휴일/주말)도 연휴 기간으로 확실히 포함
    # .values를 사용하여 불리언 값 배열을 전달
    condition_prev_day_bridge = df_eng['is_bridge_day_candidate'].shift(-1).fillna(False).values
    df_eng.loc[condition_prev_day_bridge, 'is_extended_holiday'] = 1
    
    condition_next_day_bridge = df_eng['is_bridge_day_candidate'].shift(1).fillna(False).values
    df_eng.loc[condition_next_day_bridge, 'is_extended_holiday'] = 1
    
    df_eng['final_holiday_group'] = (df_eng['is_extended_holiday'].diff(1) != 0).astype(int).cumsum()
    df_eng['consecutive_extended_days_off'] = df_eng.groupby('final_holiday_group')['is_extended_holiday'].transform('sum')
    df_eng['is_final_long_holiday'] = ((df_eng['is_extended_holiday'] == 1) & (df_eng['consecutive_extended_days_off'] >= long_holiday_threshold)).astype(int)

    lag_values = [1, 7, 14, 30]; rolling_windows = [7, 14, 30]
    if actual_visitor_col_for_lag and actual_visitor_col_for_lag in df_eng.columns:
        for lag in lag_values: df_eng[f'{actual_visitor_col_for_lag}_Lag{lag}'] = df_eng[actual_visitor_col_for_lag].shift(lag).fillna(0)
        for window in rolling_windows: df_eng[f'{actual_visitor_col_for_lag}_Roll{window}_Mean'] = df_eng[actual_visitor_col_for_lag].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
    else:
        for lag in lag_values: df_eng[f'{TARGET_VISITOR_COL}_Lag{lag}'] = 0 # Use global TARGET_VISITOR_COL for consistency
        for window in rolling_windows: df_eng[f'{TARGET_VISITOR_COL}_Roll{window}_Mean'] = 0
        
    if actual_rescue_event_col_for_lag and actual_rescue_event_col_for_lag in df_eng.columns:
        df_eng['rescue_event_yesterday'] = df_eng[actual_rescue_event_col_for_lag].shift(1).fillna(0)
    else: 
        df_eng['rescue_event_yesterday'] = 0
        
    weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'), 'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    for condition, (col, window, new_col_name) in weather_rules.items():
        if col in df_eng.columns:
            is_condition = (df_eng[col] > 0 if condition == 'rain' else df_eng[col] < 0).astype(int)
            df_eng[new_col_name] = (is_condition.rolling(window=window, min_periods=window).sum() == window).astype(int).fillna(0)
            
    time_cols_map = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    for orig_col, new_col in time_cols_map.items():
        if orig_col in df_eng.columns:
            try: 
                df_eng[new_col] = pd.to_datetime(df_eng[orig_col], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
            except: 
                df_eng[new_col] = -1
                
    temp_col, hum_col = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    if temp_col in df_eng.columns and hum_col in df_eng.columns:
        df_eng['Temp_Humidity_Interaction'] = df_eng[temp_col].fillna(df_eng[temp_col].mean()) * df_eng[hum_col].fillna(df_eng[hum_col].mean())
    temp_cols_to_drop = ['temp_day_of_week', 'temp_month', 'is_official_holiday', 'is_day_off_official', 'day_off_group', 
                         'consecutive_official_days_off', 'is_base_long_holiday', 'prev_day_is_off', 'next_day_is_off', 
                         'is_bridge_day_candidate', 'is_extended_holiday', 'final_holiday_group', 
                         'consecutive_extended_days_off', 'TimeOfMaxTempC', 'TimeOfMinTempC']
    
    df_eng.drop(columns=[col for col in temp_cols_to_drop if col in df_eng.columns], inplace=True, errors='ignore')
    
    # --- END OF PASTED LOGIC ---
    print("  Feature engineering for visitor model complete.")
    return df_eng

def preprocess_for_prediction(df_engineered, model_pkg, model_name_log="Model"):
    print(f"  Preprocessing for {model_name_log} Prediction...")
    selected_features = model_pkg.get('features')
    scaler = model_pkg.get('scaler')
    imputer_means = model_pkg.get('imputer_means', {})
    cols_scaled_at_fit = model_pkg.get('cols_scaled_at_fit')

    if selected_features is None: print(f"    Error: Feature list not found in {model_name_log} package."); return None

    missing_cols = [col for col in selected_features if col not in df_engineered.columns]
    if missing_cols:
        print(f"    Warning: Required features missing in engineered data for {model_name_log}: {missing_cols}. Filling with 0.")
        for col in missing_cols: df_engineered[col] = 0
            
    try:
        X_selected = df_engineered[selected_features].copy()
    except KeyError as e:
        print(f"    Error selecting features for {model_name_log}: {e}"); return None

    for col in X_selected.columns:
        if X_selected[col].isnull().any():
            fill_value = imputer_means.get(col, X_selected[col].mean()) # Fallback to current mean if not in imputer_means
            X_selected[col].fillna(fill_value, inplace=True)
            
    if scaler:
        if cols_scaled_at_fit is not None:
            transform_target_cols = [col for col in cols_scaled_at_fit if col in X_selected.columns]
            if len(transform_target_cols) != len(cols_scaled_at_fit):
                print(f"    Warning: Mismatch in columns to scale for {model_name_log}. Expected {len(cols_scaled_at_fit)}, found {len(transform_target_cols)} in current data.")
            if transform_target_cols:
                try:
                    X_selected[transform_target_cols] = scaler.transform(X_selected[transform_target_cols])
                except ValueError as e: print(f"    Error scaling data for {model_name_log}: {e}"); return None
            else: print(f"    No columns to scale for {model_name_log} based on 'cols_scaled_at_fit'.")
        else: print(f"    Warning: 'cols_scaled_at_fit' not found in {model_name_log} package. Skipping scaling or using heuristic.")
            # Fallback or error if cols_scaled_at_fit is crucial and missing
    else: print(f"    No scaler found in {model_name_log} package. Skipping scaling.")
            
    print(f"  Preprocessing for {model_name_log} prediction complete.")
    return X_selected

def engineer_features_for_rescue_model(df_input_with_pred_visitor, date_col, 
                                       predicted_visitor_col, # This is now a feature
                                       actual_rescue_event_col_for_lag=None): # For rescue_event_yesterday
    print("  Feature Engineering for Rescue Model...")
    df_eng = df_input_with_pred_visitor.copy()
    if date_col in df_eng.columns:
        df_eng['is_weekend'] = (df_eng[date_col].dt.dayofweek >= 5).astype(int)
        df_eng['month_num'] = df_eng[date_col].dt.month
        df_eng = pd.get_dummies(df_eng, columns=['month_num'], prefix='month', drop_first=False)
    if TARGET_VISITOR_COL and TARGET_VISITOR_COL in df_eng.columns:
        for lag in [1, 7, 14, 30]: df_eng[f'{TARGET_VISITOR_COL}_Lag{lag}'] = df_eng[TARGET_VISITOR_COL].shift(lag).fillna(0)
        for window in [7, 14, 30]: df_eng[f'{TARGET_VISITOR_COL}_Roll{window}_Mean'] = df_eng[TARGET_VISITOR_COL].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
    else:
        for lag in [1, 7, 14, 30]: df_eng[f'{TARGET_VISITOR_COL}_Lag{lag}'] = 0
        for window in [7, 14, 30]: df_eng[f'{TARGET_VISITOR_COL}_Roll{window}_Mean'] = 0
    if ORIGINAL_RESCUE_EVENT_COL_FOR_FE and ORIGINAL_RESCUE_EVENT_COL_FOR_FE in df_eng.columns:
        df_eng['rescue_event_yesterday'] = df_eng[ORIGINAL_RESCUE_EVENT_COL_FOR_FE].shift(1).fillna(0)
    else: df_eng['rescue_event_yesterday'] = 0
    weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'), 'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    for condition, (col, window, new_col) in weather_rules.items():
        if col in df_eng.columns:
            is_cond = (df_eng[col] > 0 if condition == 'rain' else df_eng[col] < 0).astype(int)
            df_eng[new_col] = (is_cond.rolling(window=window, min_periods=window).sum() == window).astype(int).fillna(0)
    if ORIGINAL_RESCUE_EVENT_COL_FOR_FE and ORIGINAL_RESCUE_EVENT_COL_FOR_FE in df_eng.columns and date_col in df_eng.columns:
        try:
            df_eng['year_week_iso_temp'] = df_eng[date_col].dt.isocalendar().year.astype(str) + '-' + df_eng[date_col].dt.isocalendar().week.astype(str).str.zfill(2)
            weekly_rescue_agg = df_eng.groupby('year_week_iso_temp')[ORIGINAL_RESCUE_EVENT_COL_FOR_FE].transform('max')
            df_eng['last_week_date_temp'] = df_eng[date_col] - pd.to_timedelta(7, unit='D')
            df_eng['last_year_week_iso_temp'] = df_eng['last_week_date_temp'].dt.isocalendar().year.astype(str) + '-' + df_eng['last_week_date_temp'].dt.isocalendar().week.astype(str).str.zfill(2)
            week_map = pd.Series(weekly_rescue_agg.values, index=df_eng['year_week_iso_temp']).drop_duplicates().to_dict()
            df_eng['rescue_in_last_week'] = df_eng['last_year_week_iso_temp'].map(week_map).fillna(0).astype(int)
            df_eng.drop(columns=['year_week_iso_temp', 'last_week_date_temp', 'last_year_week_iso_temp'], inplace=True, errors='ignore')
        except: 
            df_eng['rescue_in_last_week'] = 0
    else: 
        df_eng['rescue_in_last_week'] = 0
        
    time_cols = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    
    for orig, new in time_cols.items():
        if orig in df_eng.columns:
            try: 
                df_eng[new] = pd.to_datetime(df_eng[orig], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
            except: 
                df_eng[new] = -1
                
            df_eng.drop(columns=[orig], inplace=True, errors='ignore')
            
    temp_c, hum_c = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    if temp_c in df_eng.columns and hum_c in df_eng.columns:
        df_eng['Temp_Humidity_Interaction'] = df_eng[temp_c].fillna(df_eng[temp_c].mean()) * df_eng[hum_c].fillna(df_eng[hum_c].mean())
    
    if date_col in df_eng.columns and 'is_weekend' in df_eng.columns: # Ensure is_weekend exists
        min_year_h, max_year_h = df_eng[date_col].dt.year.min(), df_eng[date_col].dt.year.max()
        if pd.isna(min_year_h) or pd.isna(max_year_h): 
            current_year_h = df_eng[date_col].dt.year.iloc[0]; kr_holidays_r = holidays.KR(years=current_year_h)
        else: 
            kr_holidays_r = holidays.KR(years=range(min_year_h, max_year_h + 1))
            
        df_eng['is_official_holiday'] = df_eng[date_col].apply(lambda date: date in kr_holidays_r)
        df_eng['is_day_off_official'] = (df_eng['is_weekend'] | df_eng['is_official_holiday']).astype(int)
        
        long_holiday_threshold = 3
        df_eng['is_base_long_holiday'] = ((df_eng['is_day_off_official'] == 1) & (df_eng.groupby((df_eng['is_day_off_official'].diff(1) != 0).astype(int).cumsum())['is_day_off_official'].transform('sum') >= long_holiday_threshold)).astype(int)
        df_eng['prev_day_is_off'] = df_eng['is_day_off_official'].shift(1).fillna(0).astype(int)
        df_eng['next_day_is_off'] = df_eng['is_day_off_official'].shift(-1).fillna(0).astype(int)
        df_eng['is_bridge_day_candidate'] = ((df_eng['is_day_off_official'] == 0) & (df_eng['prev_day_is_off'] == 1) & (df_eng['next_day_is_off'] == 1)).astype(int)
        df_eng['is_extended_holiday'] = df_eng['is_base_long_holiday']
        
        df_eng.loc[df_eng['is_bridge_day_candidate'].values == 1, 'is_extended_holiday'] = 1 # Use .values for boolean indexing
        df_eng.loc[df_eng['is_bridge_day_candidate'].shift(-1).fillna(False).values, 'is_extended_holiday'] = 1
        df_eng.loc[df_eng['is_bridge_day_candidate'].shift(1).fillna(False).values, 'is_extended_holiday'] = 1
        
        df_eng['final_holiday_group'] = (df_eng['is_extended_holiday'].diff(1) != 0).astype(int).cumsum()
        df_eng['consecutive_extended_days_off'] = df_eng.groupby('final_holiday_group')['is_extended_holiday'].transform('sum')
        df_eng['is_final_long_holiday_rescue'] = ((df_eng['is_extended_holiday'] == 1) & (df_eng['consecutive_extended_days_off'] >= long_holiday_threshold)).astype(int)
        
        df_eng.drop(columns=['is_official_holiday', 'is_day_off_official', 'day_off_group', 'consecutive_official_days_off', 'is_base_long_holiday', 'prev_day_is_off', 'next_day_is_off', 'is_bridge_day_candidate', 'is_extended_holiday', 'final_holiday_group', 'consecutive_extended_days_off'], inplace=True, errors='ignore')
    print("Feature engineering for rescue model complete.")
    return df_eng

# --- Main Execution ---
if __name__ == '__main__':
    # --- 데이터 로드 ---
    df_raw_full = load_data(INPUT_DATA_FILE, DATE_COLUMN)
    if df_raw_full is None or df_raw_full.empty:
        print("데이터 로드 실패. 프로그램을 종료합니다."); exit()

    # --- 모델 패키지 로드 ---
    visitor_model_package = load_model_package(VISITOR_MODEL_PKG_PATH)
    rescue_model_package = load_model_package(RESCUE_MODEL_PKG_PATH)
    if visitor_model_package is None or rescue_model_package is None:
        print("하나 이상의 모델 패키지 로드 실패. 프로그램을 종료합니다."); exit()

    # --- 예측 대상 기간 설정 (예: 마지막 14일) ---
    print(f"\n--- {TEST_PREDICTION_RANGE}일간의 예측을 준비합니다 ---")

    # =======================================================
    # === 단계 1: 탐방객 수 예측 ===
    # =======================================================
    print("\n--- 단계 1: 탐방객 수 예측 시작 ---")

    # 1a. 탐방객 모델 FE를 위한 데이터 슬라이스 준비
    #     Lag/Rolling 피처 생성을 위해 예측 대상 기간 이전의 데이터도 포함해야 함.
    #     탐방객 모델 학습 시 사용된 최대 Lag 기간을 알아야 함 (예: 30일)
    max_lag_for_visitor_model = 30 # 예시, 실제 탐방객 모델의 최대 Lag 기간으로 설정
    required_data_len_for_visitor_fe = TEST_PREDICTION_RANGE + max_lag_for_visitor_model

    if len(df_raw_full) < required_data_len_for_visitor_fe:
        print(f"오류: 탐방객 모델 FE에 필요한 데이터가 부족합니다. (필요: {required_data_len_for_visitor_fe}일, 현재: {len(df_raw_full)}일)"); exit()
    
    df_slice_for_visitor_fe_raw = df_raw_full.tail(required_data_len_for_visitor_fe).copy()
    print(f"  탐방객 모델 FE용 원본 데이터 슬라이스 준비 완료 (최근 {required_data_len_for_visitor_fe}일).")

    # 1b. 초기 전처리 (로그 변환, 이상치 처리 등 - 탐방객 모델 타겟에 대해)
    df_processed_visitor_slice = initial_preprocess_for_visitor(
        df_slice_for_visitor_fe_raw, 
        TARGET_VISITOR_COL, 
        ORIGINAL_RESCUE_COUNT_COL, 
        ORIGINAL_RESCUE_EVENT_COL_FOR_FE, # 탐방객 FE에 사용될 조난 이벤트 컬럼
        apply_log_transform=APPLY_LOG_TRANSFORM_VISITOR_TARGET
    )
    if APPLY_OUTLIER_CAPPING_VISITOR_TARGET:
        df_processed_visitor_slice = cap_outliers(
            df_processed_visitor_slice, 
            TARGET_VISITOR_COL, 
            method=OUTLIER_CAPPING_METHOD,
            iqr_multiplier=IQR_MULTIPLIER,
            lower_p=PERCENTILE_LOWER,
            upper_p=PERCENTILE_UPPER
        )
    
    # 1c. 탐방객 모델용 피처 엔지니어링 (슬라이스된 데이터 전체에 적용)
    #     Lag/Rolling 생성 시 실제 TARGET_VISITOR_COL과 ORIGINAL_RESCUE_EVENT_COL_FOR_FE 사용
    df_engineered_visitor_slice = engineer_features_for_visitor_model(
        df_processed_visitor_slice, 
        DATE_COLUMN, 
        TARGET_VISITOR_COL, # 실제 타겟 컬럼명 전달 (Lag/Rolling 생성용)
        ORIGINAL_RESCUE_EVENT_COL_FOR_FE # 실제 조난 이벤트 컬럼 전달
    )

    # 1d. 엔지니어링된 데이터에서 실제 예측 대상 기간(마지막 14일)만 선택
    df_engineered_visitor_target_period = df_engineered_visitor_slice.tail(TEST_PREDICTION_RANGE).copy()
    if len(df_engineered_visitor_target_period) != TEST_PREDICTION_RANGE:
         print(f"오류: 탐방객 FE 후 예측 대상 기간 데이터 추출 실패."); exit()

    # 1e. 최종 전처리 (피처 선택, 스케일링 - 탐방객 모델용)
    X_to_predict_visitor = preprocess_for_prediction(
        df_engineered_visitor_target_period.copy(), # 예측 대상 기간의 엔지니어링된 데이터
        visitor_model_package,
        model_name_log="Visitor"
    )
    if X_to_predict_visitor is None: print("오류: 탐방객 수 예측을 위한 데이터 전처리 실패"); exit()

    # 1f. 탐방객 수 예측 수행
    predicted_visitor_counts_log_scale = visitor_model_package['model'].predict(X_to_predict_visitor)
    if APPLY_LOG_TRANSFORM_VISITOR_TARGET:
        predicted_visitor_counts_original_scale = np.expm1(predicted_visitor_counts_log_scale)
    else:
        predicted_visitor_counts_original_scale = predicted_visitor_counts_log_scale
    predicted_visitor_counts_original_scale = np.maximum(0, predicted_visitor_counts_original_scale)
    print("탐방객 수 예측 완료.")

    # 예측 결과 저장을 위한 DataFrame (날짜, 예측된 탐방객 수, 실제 탐방객 수)
    visitor_prediction_results_df = pd.DataFrame({
        DATE_COLUMN: df_engineered_visitor_target_period[DATE_COLUMN].values,
        PREDICTED_VISITOR_COL_NAME: predicted_visitor_counts_original_scale
        
    })
    if TARGET_VISITOR_COL in df_engineered_visitor_target_period.columns: # 실제값 비교용
        actual_visitors_log = df_engineered_visitor_target_period[TARGET_VISITOR_COL]
        actual_visitors_orig = np.expm1(actual_visitors_log) if APPLY_LOG_TRANSFORM_VISITOR_TARGET else actual_visitors_log
        visitor_prediction_results_df['Actual_Visitors'] = actual_visitors_orig.values
        
        print(f"\n--- 탐방객 수 예측 결과 (마지막 {TEST_PREDICTION_RANGE}일) ---")
        print(visitor_prediction_results_df)
        if 'Actual_Visitors' in visitor_prediction_results_df.columns:
            mae_vis = mean_absolute_error(visitor_prediction_results_df['Actual_Visitors'], visitor_prediction_results_df[PREDICTED_VISITOR_COL_NAME])
            r2_vis = r2_score(visitor_prediction_results_df['Actual_Visitors'], visitor_prediction_results_df[PREDICTED_VISITOR_COL_NAME])
            print(f"  MAE (원래 스케일): {mae_vis:.2f}, R2 (원래 스케일): {r2_vis:.4f}")


    # =======================================================
    # === 단계 2: 조난 확률 예측 ===
    # =======================================================
    print("\n\n--- 단계 2: 조난 확률 예측 시작 ---")

    # 2a. 조난 모델 FE를 위한 입력 데이터 준비
    #     예측 대상 기간(마지막 14일)의 '원본' 피처와 '예측된 탐방객 수'를 사용.
    #     Lag 피처 생성을 위해 조난 모델도 과거 데이터가 필요하다면 유사한 슬라이싱 필요.
    max_lag_for_rescue_model = 30 # 예시, 실제 조난 모델의 최대 Lag 기간으로 설정
    required_data_len_for_rescue_fe = TEST_PREDICTION_RANGE + max_lag_for_rescue_model

    if len(df_raw_full) < required_data_len_for_rescue_fe:
        print(f"오류: 조난 모델 FE에 필요한 데이터가 부족합니다. (필요: {required_data_len_for_rescue_fe}일, 현재: {len(df_raw_full)}일)"); exit()
    
    df_slice_for_rescue_fe_raw = df_raw_full.tail(required_data_len_for_rescue_fe).copy()
    print(f"  조난 모델 FE용 원본 데이터 슬라이스 준비 완료 (최근 {required_data_len_for_rescue_fe}일).")

    # 예측된 탐방객 수를 이 슬라이스된 원본 데이터에 병합 (마지막 14일에만 값이 있음)
    df_input_for_rescue_fe = pd.merge(
        df_slice_for_rescue_fe_raw,
        visitor_prediction_results_df[[DATE_COLUMN, PREDICTED_VISITOR_COL_NAME]], # 날짜와 예측된 탐방객 수만
        on=DATE_COLUMN,
        how='left' 
    )
    df_input_for_rescue_fe[PREDICTED_VISITOR_COL_NAME].fillna(df_input_for_rescue_fe[TARGET_VISITOR_COL], inplace=True)


    print(f"  조난 예측용 입력 데이터 준비 완료 (슬라이스 + 예측 탐방객). Shape: {df_input_for_rescue_fe.shape}")

    # 2b. 조난 모델용 피처 엔지니어링
    df_rescue_engineered_slice = engineer_features_for_rescue_model( # 함수명 일치 필요
        df_input_for_rescue_fe.copy(),
        DATE_COLUMN,
        PREDICTED_VISITOR_COL_NAME, # 예측된 탐방객 수를 주요 탐방객 피처로 사용 가능
        ORIGINAL_RESCUE_EVENT_COL_FOR_FE # 실제 조난 이벤트 컬럼 (Lag 생성용)
    )
    if df_rescue_engineered_slice is None: print("오류: 조난 모델용 피처 엔지니어링 실패"); exit()

    # 2c. 엔지니어링된 데이터에서 실제 예측 대상 기간(마지막 14일)만 선택
    df_rescue_engineered_target_period = df_rescue_engineered_slice.tail(TEST_PREDICTION_RANGE).copy()
    if len(df_rescue_engineered_target_period) != TEST_PREDICTION_RANGE:
         print(f"오류: 조난 FE 후 예측 대상 기간 데이터 추출 실패."); exit()

    # 2d. 최종 전처리 (피처 선택, 스케일링 - 조난 모델용)
    X_to_predict_rescue = preprocess_for_prediction(
        df_rescue_engineered_target_period.copy(),
        rescue_model_package,
        model_name_log="Rescue"
    )
    if X_to_predict_rescue is None: 
        print("오류: 조난 모델용 데이터 전처리 실패"); exit()

    # 2e. 조난 발생 확률 예측
    predicted_rescue_probabilities = rescue_model_package['model'].predict_proba(X_to_predict_rescue)[:, 1]
    print("조난 발생 확률 예측 완료.")
        
    # --- 최종 결과 통합 및 저장 ---
    # 13. 최종 결과 DataFrame에 조난 확률 추가
    final_results_df = visitor_prediction_results_df.copy() 
    if len(final_results_df) == len(predicted_rescue_probabilities):
        final_results_df['Predicted_Rescue_Probability'] = predicted_rescue_probabilities
    else:
        print("경고: 예측된 조난 확률과 결과 DataFrame 길이 불일치. 'Date' 기준으로 병합 시도.")
        temp_rescue_proba_df = pd.DataFrame({
            DATE_COLUMN: df_rescue_engineered_target_period[DATE_COLUMN].values,
            'Predicted_Rescue_Probability': predicted_rescue_probabilities
        })
        final_results_df = pd.merge(final_results_df, temp_rescue_proba_df, on=DATE_COLUMN, how='left')


    # --- 실제 조난 발생 유무(0 또는 1) 추가 ---
    actual_rescue_events_source_df = get_last_n_days_data(df_raw_full, TEST_PREDICTION_RANGE)
    if ORIGINAL_RESCUE_COUNT_COL in actual_rescue_events_source_df.columns:
        actual_rescue_count_data = actual_rescue_events_source_df[[DATE_COLUMN, ORIGINAL_RESCUE_COUNT_COL]].copy()
        actual_rescue_count_data.rename(columns={ORIGINAL_RESCUE_COUNT_COL: 'Actual_Rescue_Count'}, inplace=True)
        actual_rescue_count_data['Actual_Rescue_Event'] = (pd.to_numeric(actual_rescue_count_data['Actual_Rescue_Count'], errors='coerce').fillna(0) > 0).astype(int)
        final_results_df = pd.merge(final_results_df, actual_rescue_count_data[[DATE_COLUMN, 'Actual_Rescue_Count', 'Actual_Rescue_Event']], on=DATE_COLUMN, how='left')
        if final_results_df['Actual_Rescue_Event'].isnull().any():
            final_results_df['Actual_Rescue_Event'].fillna(0, inplace=True)
        if final_results_df['Actual_Rescue_Count'].isnull().any():
            final_results_df['Actual_Rescue_Count'].fillna(0, inplace=True)
    else:
        final_results_df['Actual_Rescue_Event'] = np.nan
        final_results_df['Actual_Rescue_Count'] = np.nan

    print("\n--- 최종 예측 결과 (탐방객 수 + 조난 확률 + 실제 조난 정보) ---")
    # 전체 대신 일부만 출력하도록 수정 (예: 마지막 5개 또는 head(5))
    print(final_results_df.tail() if len(final_results_df) > 5 else final_results_df)


# (여기에 조난 예측 모델 성능 평가 코드 추가 - 이전 답변 참고)
    if 'Actual_Rescue_Event' in final_results_df.columns and not final_results_df['Actual_Rescue_Event'].isnull().all():
        print(f"\n--- 조난 예측 모델 성능 평가 (마지막 {TEST_PREDICTION_RANGE}일) ---")
        y_true_r = final_results_df['Actual_Rescue_Event'].astype(int) 
        y_pred_p_r = final_results_df['Predicted_Rescue_Probability'].fillna(0.5)
        
        # --- 기본 임계값 0.5로 평가 ---
        y_pred_b_r_default = (y_pred_p_r >= 0.5).astype(int)
        print("\nClassification Report (Threshold = 0.5):")
        print(classification_report(y_true_r, y_pred_b_r_default, zero_division=0))
        roc_auc_value = np.nan
        if len(np.unique(y_true_r)) > 1:
            roc_auc_value = roc_auc_score(y_true_r, y_pred_p_r)
            print(f"  ROC AUC: {roc_auc_value:.4f}")
        else:
            print(f"  실제 조난 이벤트가 한 종류({y_true_r.unique()})만 있어 ROC AUC를 계산할 수 없습니다. Accuracy: {accuracy_score(y_true_r, y_pred_b_r_default):.4f}")
        print(f"  F1 Score (Threshold = 0.5, for class 1): {f1_score(y_true_r, y_pred_b_r_default, pos_label=1, zero_division=0):.4f}")

        # --- Precision-Recall Curve 값들 가져오기 ---
        precision_values, recall_values, pr_thresholds = precision_recall_curve(y_true_r, y_pred_p_r)
        # thresholds는 precision/recall보다 길이가 1 짧음. precision/recall의 마지막 값은 각각 1.0, 0.0.
        # 계산의 편의를 위해 thresholds와 길이가 같은 precision[:-1], recall[:-1] 사용.
        valid_precisions = precision_values[:-1]
        valid_recalls = recall_values[:-1]

        # --- 시나리오 1: Recall 중심 임계값 (Recall >= 0.75~0.80, Precision 최대화) ---
        min_recall_target = 0.7 # 최소 Recall 목표 (0.75에서 0.80 사이로 조절 가능)
        
        candidate_indices_recall_focused = np.where(valid_recalls >= min_recall_target)[0]
        
        chosen_threshold_recall_focused = 0.5 # 기본값
        precision_at_chosen_recall_thresh = 0.0
        recall_at_chosen_recall_thresh = 0.0

        if len(candidate_indices_recall_focused) > 0:
            # 최소 Recall을 만족하는 인덱스들 중에서 Precision이 가장 높은 인덱스 찾기
            precisions_at_candidates = valid_precisions[candidate_indices_recall_focused]
            best_candidate_idx = candidate_indices_recall_focused[np.argmax(precisions_at_candidates)]
            
            chosen_threshold_recall_focused = pr_thresholds[best_candidate_idx]
            precision_at_chosen_recall_thresh = valid_precisions[best_candidate_idx]
            recall_at_chosen_recall_thresh = valid_recalls[best_candidate_idx]

            print(f"\n--- 시나리오 1: Recall 중심 (Recall >= {min_recall_target:.2f}, Precision 최대화) ---")
            print(f"  선택된 임계값: {chosen_threshold_recall_focused:.4f}")
            print(f"  해당 임계값에서의 Precision: {precision_at_chosen_recall_thresh:.4f}")
            print(f"  해당 임계값에서의 Recall: {recall_at_chosen_recall_thresh:.4f}")
            
            y_pred_recall_focused = (y_pred_p_r >= chosen_threshold_recall_focused).astype(int)
            print("\nClassification Report (Recall-focused Threshold):")
            print(classification_report(y_true_r, y_pred_recall_focused, zero_division=0))
            print(f"  F1 Score (Recall-focused, for class 1): {f1_score(y_true_r, y_pred_recall_focused, pos_label=1, zero_division=0):.4f}")
            cm_recall_focused = confusion_matrix(y_true_r, y_pred_recall_focused)
            print(f"  Confusion Matrix (Recall-focused):\n{cm_recall_focused}")
        else:
            print(f"\n경고: 최소 Recall 목표 ({min_recall_target:.2f})를 만족하는 임계값을 찾을 수 없습니다.")


        # --- 시나리오 2: Precision 중심 임계값 (Precision >= 0.75~0.80, Recall 최대화) ---
        min_precision_target_scenario2 = 0.65 # 최소 Precision 목표 (달성 어려울 수 있음, 필요시 낮춤)
        
        candidate_indices_precision_focused = np.where(valid_precisions >= min_precision_target_scenario2)[0]

        chosen_threshold_precision_focused = 0.5 # 기본값
        precision_at_chosen_precision_thresh = 0.0
        recall_at_chosen_precision_thresh = 0.0

        if len(candidate_indices_precision_focused) > 0:
            # 최소 Precision을 만족하는 인덱스들 중에서 Recall이 가장 높은 인덱스 찾기
            recalls_at_candidates_prec = valid_recalls[candidate_indices_precision_focused]
            best_candidate_idx_prec = candidate_indices_precision_focused[np.argmax(recalls_at_candidates_prec)]
            
            chosen_threshold_precision_focused = pr_thresholds[best_candidate_idx_prec]
            precision_at_chosen_precision_thresh = valid_precisions[best_candidate_idx_prec]
            recall_at_chosen_precision_thresh = valid_recalls[best_candidate_idx_prec]

            print(f"\n--- 시나리오 2: Precision 중심 (Precision >= {min_precision_target_scenario2:.2f}, Recall 최대화) ---")
            print(f"  선택된 임계값: {chosen_threshold_precision_focused:.4f}")
            print(f"  해당 임계값에서의 Precision: {precision_at_chosen_precision_thresh:.4f}")
            print(f"  해당 임계값에서의 Recall: {recall_at_chosen_precision_thresh:.4f}")

            y_pred_precision_focused = (y_pred_p_r >= chosen_threshold_precision_focused).astype(int)
            print("\nClassification Report (Precision-focused Threshold):")
            print(classification_report(y_true_r, y_pred_precision_focused, zero_division=0))
            print(f"  F1 Score (Precision-focused, for class 1): {f1_score(y_true_r, y_pred_precision_focused, pos_label=1, zero_division=0):.4f}")
            cm_precision_focused = confusion_matrix(y_true_r, y_pred_precision_focused)
            print(f"  Confusion Matrix (Precision-focused):\n{cm_precision_focused}")
        else:
            print(f"\n경고: 최소 Precision 목표 ({min_precision_target_scenario2:.2f})를 만족하는 임계값을 찾을 수 없습니다.")
            # 대안: Precision이 가장 높은 지점의 임계값 사용
            if len(valid_precisions) > 0:
                best_precision_idx_overall = np.argmax(valid_precisions)
                # thresholds는 precision/recall보다 1개 적으므로, best_precision_idx_overall이 thresholds 범위 내인지 확인
                if best_precision_idx_overall < len(pr_thresholds):
                    alt_threshold_prec_focused = pr_thresholds[best_precision_idx_overall]
                    alt_precision = valid_precisions[best_precision_idx_overall]
                    alt_recall = valid_recalls[best_precision_idx_overall]
                    print(f"  대안: 가장 높은 Precision 지점의 임계값 사용: {alt_threshold_prec_focused:.4f} (Precision: {alt_precision:.4f}, Recall: {alt_recall:.4f})")
                    # 이 임계값으로 평가해볼 수 있음
                else:
                    print(f"  대안 임계값 찾기 실패 (Precision 최대화).")
            else:
                print(f"  Precision-Recall 커브에서 유효한 값을 찾을 수 없습니다.")
    else:
        print("\n실제 조난 이벤트 데이터가 없어 조난 예측 모델 성능을 평가할 수 없습니다.")

    # --- 결과 저장을 위해 소수점 자릿수 조정 ---
    cols_to_round = {
        'Predicted_Total_Visitor_Count' : 4,
        'Actual_Visitors': 4, 
        'Predicted_Rescue_Probability': 4
    }
    df_to_save = final_results_df.copy()
    for col, decimals in cols_to_round.items():
        if col in df_to_save.columns:
            df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')
            df_to_save[col] = df_to_save[col].round(decimals)
    final_output_filename = os.path.join(RUN_SPECIFIC_DIR, "final_14days_predictions_with_rescue.csv")
    df_to_save.to_csv(final_output_filename, index=False, encoding='utf-8-sig')
    print(f"최종 예측 결과 저장 완료 (소수점 조정됨): {final_output_filename}")

    print("\n========== 전체 예측 파이프라인 실행 완료 ==========")