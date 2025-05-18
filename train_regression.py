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
import xgboost as xgb # XGBoost 추가

# --- 0. Configuration ---
INPUT_DATA_FILE = "data/preprocessed_merged_data.csv"
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
# APPLY_OUTLIER_CAPPING_VISITOR_TARGET 는 실험 조건에서 제어
OUTLIER_CAPPING_METHOD = 'iqr' # 기본값, 실험 조건에서 오버라이드 가능
IQR_MULTIPLIER = 1.5
PERCENTILE_LOWER = 1
PERCENTILE_UPPER = 99

N_ITER_RANDOM_SEARCH_DEFAULT = 50 # 기본 RandomSearch 반복 횟수
RANDOM_SEED = 42
CV_N_SPLITS_VISITOR = 3
SCORER_VISITOR = 'r2'
EARLY_STOPPING_ROUNDS_LGBM_XGB = 50 # LGBM 및 XGBoost용 조기 종료 라운드

CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
# BASE_RESULTS_DIR는 메인 실행부에서 각 실험의 상위 디렉토리로 사용
BASE_EXPERIMENT_DIR = "./visitor_model_experiments" # 모든 실험 결과의 최상위 디렉토리
os.makedirs(BASE_EXPERIMENT_DIR, exist_ok=True)
print(f"All experiment results will be saved under: {BASE_EXPERIMENT_DIR}")


# --- 실험 조건 정의 ---
EXPERIMENT_CONDITIONS = [
    # LightGBM
    {"name": "LGBM_Scaled_OutlierCapped", "model_type": "lgbm", "apply_scaling": True, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": N_ITER_RANDOM_SEARCH_DEFAULT},
    {"name": "LGBM_NoScale_OutlierCapped", "model_type": "lgbm", "apply_scaling": False, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": N_ITER_RANDOM_SEARCH_DEFAULT},
    {"name": "LGBM_Scaled_NoOutlier", "model_type": "lgbm", "apply_scaling": True, "apply_outlier_capping": False, "n_iter_search": N_ITER_RANDOM_SEARCH_DEFAULT},

    # RandomForestRegressor
    {"name": "RF_Scaled_OutlierCapped", "model_type": "rf", "apply_scaling": True, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": 30}, # RF는 튜닝이 오래 걸릴 수 있어 반복 횟수 줄임
    {"name": "RF_NoScale_NoOutlier", "model_type": "rf", "apply_scaling": False, "apply_outlier_capping": False, "n_iter_search": 30},
    {"name": "RF_Scaled_NoOutlier", "model_type": "rf", "apply_scaling": True, "apply_outlier_capping": False, "n_iter_search": 30},


    # XGBoostRegressor
    {"name": "XGB_Scaled_OutlierCapped", "model_type": "xgb", "apply_scaling": True, "apply_outlier_capping": True, "outlier_method": "iqr", "n_iter_search": N_ITER_RANDOM_SEARCH_DEFAULT},
    {"name": "XGB_NoScale_NoOutlier", "model_type": "xgb", "apply_scaling": False, "apply_outlier_capping": False, "n_iter_search": N_ITER_RANDOM_SEARCH_DEFAULT},
    {"name": "XGB_Scaled_NoOutlier", "model_type": "xgb", "apply_scaling": True, "apply_outlier_capping": False, "n_iter_search": N_ITER_RANDOM_SEARCH_DEFAULT},
]

# 로그 변환은 모든 실험에서 기본 적용한다고 가정
# APPLY_LOG_TRANSFORM_VISITOR_TARGET = True # 이미 전역 설정으로 존재

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
    elif rescue_count_col in df_p.columns: # 구조 이벤트 컬럼이 없고 구조 건수 컬럼만 있을 경우 생성
        df_p[rescue_event_col] = (df_p[rescue_count_col] > 0).astype(int)
        print(f"Created '{rescue_event_col}' from '{rescue_count_col}'.")


    cols_to_drop_list_like = ['Accident_Cause_List', 'Accident_Outcome_List'] # 리스트 형태의 컬럼 제거
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

    original_min = df_c[column_name].min()
    original_max = df_c[column_name].max()
    
    df_c[column_name] = np.clip(df_c[column_name], lower_bound, upper_bound)
    
    capped_low_count = (df_c[column_name] == lower_bound).sum() - (original_min == lower_bound and (df[column_name] == original_min).sum() == (df_c[column_name] == lower_bound).sum()) # 정확한 계수를 위해 조정
    capped_high_count = (df_c[column_name] == upper_bound).sum() - (original_max == upper_bound and (df[column_name] == original_max).sum() == (df_c[column_name] == upper_bound).sum()) # 정확한 계수를 위해 조정
    
    print(f"  Capping on '{column_name}': {capped_low_count} low values (potentially) adjusted to {lower_bound:.4f}, {capped_high_count} high values (potentially) adjusted to {upper_bound:.4f}.")
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
    if not pd.api.types.is_datetime64_any_dtype(df_eng[date_col]):
        df_eng[date_col] = pd.to_datetime(df_eng[date_col])

    df_eng.reset_index(drop=True, inplace=True)

    df_eng['temp_day_of_week'] = df_eng[date_col].dt.dayofweek
    df_eng['temp_month'] = df_eng[date_col].dt.month
    df_eng['is_weekend'] = (df_eng['temp_day_of_week'] >= 5).astype(int)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['temp_month'] / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['temp_month'] / 12)
    month_onehot = pd.get_dummies(df_eng['temp_month'], prefix='month', dtype=int)
    # 특정 월(예: 10월)이 중요할 수 있으므로 직접 추가 (get_dummies는 모든 월을 생성)
    for m in range(1, 13): # 모든 월에 대해 one-hot 변수를 만들고, 없는 경우 0으로 채움
      df_eng[f'month_{m}'] = month_onehot[f'month_{m}'] if f'month_{m}' in month_onehot.columns else 0


    min_year = df_eng[date_col].dt.year.min(); max_year = df_eng[date_col].dt.year.max()
    if pd.isna(min_year) or pd.isna(max_year):
        current_year = df_eng[date_col].dt.year.iloc[0] if not df_eng.empty else datetime.now().year
        kr_holidays_list = holidays.KR(years=current_year)
    else:
        kr_holidays_list = holidays.KR(years=range(min_year, max_year + 1))
    df_eng['is_official_holiday'] = df_eng[date_col].apply(lambda date: date in kr_holidays_list).astype(int)
    df_eng['is_day_off_official'] = (df_eng['is_weekend'] | df_eng['is_official_holiday']).astype(int)
    df_eng['day_off_group'] = (df_eng['is_day_off_official'].diff().fillna(0) != 0).astype(int).cumsum()
    df_eng['consecutive_official_days_off'] = df_eng.groupby('day_off_group')['is_day_off_official'].transform('sum') * df_eng['is_day_off_official'] # 휴일이 아닌 날은 0으로
    long_holiday_threshold = 3
    df_eng['is_base_long_holiday'] = ((df_eng['is_day_off_official'] == 1) & (df_eng['consecutive_official_days_off'] >= long_holiday_threshold)).astype(int)
    df_eng['prev_day_is_off'] = df_eng['is_day_off_official'].shift(1).fillna(0).astype(int)
    df_eng['next_day_is_off'] = df_eng['is_day_off_official'].shift(-1).fillna(0).astype(int)
    df_eng['is_bridge_day_candidate'] = ((df_eng['is_day_off_official'] == 0) & (df_eng['prev_day_is_off'] == 1) & (df_eng['next_day_is_off'] == 1)).astype(int)
    
    df_eng['is_extended_holiday'] = df_eng['is_base_long_holiday'].copy()
    bridge_indices = df_eng[df_eng['is_bridge_day_candidate'] == 1].index
    for idx in bridge_indices:
        df_eng.loc[idx, 'is_extended_holiday'] = 1 # 징검다리 자체
        if idx > 0: df_eng.loc[idx-1, 'is_extended_holiday'] = 1 # 징검다리 전날
        if idx < len(df_eng) - 1: df_eng.loc[idx+1, 'is_extended_holiday'] = 1 # 징검다리 다음날
    
    df_eng['final_holiday_group'] = (df_eng['is_extended_holiday'].diff().fillna(0) != 0).astype(int).cumsum()
    df_eng['consecutive_extended_days_off'] = df_eng.groupby('final_holiday_group')['is_extended_holiday'].transform('sum') * df_eng['is_extended_holiday'] # 휴일이 아닌 날은 0으로
    df_eng['is_final_long_holiday'] = ((df_eng['is_extended_holiday'] == 1) & (df_eng['consecutive_extended_days_off'] >= long_holiday_threshold)).astype(int)

    lag_values = [1, 7, 14, 30]
    rolling_windows = [7, 14, 30]
    lag_base_col = actual_visitor_col_for_lag if actual_visitor_col_for_lag and actual_visitor_col_for_lag in df_eng.columns else None
    if lag_base_col:
        for lag in lag_values:
            df_eng[f'{lag_base_col}_Lag{lag}'] = df_eng[lag_base_col].shift(lag).fillna(method='bfill').fillna(0) # bfill 후 0 채우기
        for window in rolling_windows:
            df_eng[f'{lag_base_col}_Roll{window}_Mean'] = df_eng[lag_base_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(method='bfill').fillna(0)
    else:
        default_lag_col_name = TARGET_VISITOR_COL # TARGET_VISITOR_COL을 기본값으로 사용
        for lag in lag_values:
            df_eng[f'{default_lag_col_name}_Lag{lag}'] = 0
        for window in rolling_windows:
            df_eng[f'{default_lag_col_name}_Roll{window}_Mean'] = 0
        if actual_visitor_col_for_lag:
            print(f"  Warning: Lag base col '{actual_visitor_col_for_lag}' not found. Lags/Rolls for '{default_lag_col_name}' filled with 0.")


    rescue_lag_base_col = actual_rescue_event_col_for_lag if actual_rescue_event_col_for_lag and actual_rescue_event_col_for_lag in df_eng.columns else None
    if rescue_lag_base_col:
        df_eng['rescue_event_yesterday'] = df_eng[rescue_lag_base_col].shift(1).fillna(0)
    else:
        df_eng['rescue_event_yesterday'] = 0
        if actual_rescue_event_col_for_lag: print(f"  Warning: Rescue lag base col '{actual_rescue_event_col_for_lag}' not found. 'rescue_event_yesterday' filled with 0.")

    weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'),
                     'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    for condition, (col, window, new_col_name) in weather_rules.items():
        if col in df_eng.columns:
            is_condition = (df_eng[col] > 0 if condition == 'rain' else df_eng[col] < 0).astype(int)
            df_eng[new_col_name] = (is_condition.rolling(window=window, min_periods=window).sum() == window).astype(int).fillna(0)
        else:
            df_eng[new_col_name] = 0 # 해당 컬럼이 없으면 0으로 채움
            print(f"  Warning: Weather column '{col}' for '{new_col_name}' not found. Filled with 0.")


    time_cols_map = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    for orig_col, new_col in time_cols_map.items():
        if orig_col in df_eng.columns:
            try:
                df_eng[new_col] = pd.to_datetime(df_eng[orig_col], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
            except Exception: # 더 일반적인 예외 처리
                df_eng[new_col] = -1
                print(f"  Warning: Could not parse time column '{orig_col}'. '{new_col}' filled with -1.")
        else:
            df_eng[new_col] = -1 # 해당 컬럼이 없으면 -1로 채움
            print(f"  Warning: Time column '{orig_col}' for '{new_col}' not found. Filled with -1.")


    temp_col, hum_col = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    if temp_col in df_eng.columns and hum_col in df_eng.columns:
        mean_temp = df_eng[temp_col].mean(); mean_hum = df_eng[hum_col].mean()
        df_eng['Temp_Humidity_Interaction'] = df_eng[temp_col].fillna(mean_temp) * df_eng[hum_col].fillna(mean_hum)
    else:
        df_eng['Temp_Humidity_Interaction'] = 0 # 컬럼 없으면 0
        print(f"  Warning: Temp ('{temp_col}') or Humidity ('{hum_col}') column not found for interaction. Filled with 0.")


    temp_cols_to_drop = [
        'temp_day_of_week', 'temp_month', 'is_official_holiday',
        'day_off_group', 'consecutive_official_days_off', 'is_base_long_holiday',
        'prev_day_is_off', 'next_day_is_off', 'is_bridge_day_candidate',
        'is_extended_holiday', 'final_holiday_group', 'consecutive_extended_days_off',
        'TimeOfMaxTempC', 'TimeOfMinTempC' # 원본 시간 컬럼은 변환 후 삭제
    ]
    df_eng.drop(columns=[col for col in temp_cols_to_drop if col in df_eng.columns], inplace=True, errors='ignore')
    print("Feature engineering for visitor model complete.")
    return df_eng
# --- 5. Data Splitting, Feature Selection & Scaling ---
def prepare_and_split_data_for_training(df_engineered, target_col_name, date_col_name,
                                           train_r, val_r, test_r, top_n,
                                           apply_scaling):
    print("\n--- 5. Preparing and Splitting Data for Model Training ---")

    exclude_from_X = [date_col_name, target_col_name, ORIGINAL_RESCUE_COUNT_COL, ORIGINAL_RESCUE_EVENT_COL_FOR_FE]
    potential_X_cols = [col for col in df_engineered.columns if col not in exclude_from_X and col != target_col_name]

    X_full = df_engineered[potential_X_cols].copy()
    y_full = df_engineered[target_col_name].copy()
    dates_full = df_engineered[date_col_name].copy()

    n_total = len(X_full)
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)

    X_train_raw = X_full.iloc[:n_train]
    y_train = y_full.iloc[:n_train]
    dates_train = dates_full.iloc[:n_train]

    X_val_raw = X_full.iloc[n_train : n_train + n_val]
    y_val = y_full.iloc[n_train : n_train + n_val]
    dates_val = dates_full.iloc[n_train : n_train + n_val]

    X_test_raw = X_full.iloc[n_train + n_val:]
    y_test = y_full.iloc[n_train + n_val:]
    dates_test = dates_full.iloc[n_train + n_val:]
    print(f"Data split: Train {len(X_train_raw)}, Validation {len(X_val_raw)}, Test {len(X_test_raw)}")

    X_train_for_imp = X_train_raw.copy()
    imputer_values_train = {}
    for col in X_train_for_imp.columns:
        # 먼저 NaN 처리
        if X_train_for_imp[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train_for_imp[col]):
                val_to_fill = X_train_for_imp[col].mean()
            else:
                val_to_fill = X_train_for_imp[col].mode()[0] if not X_train_for_imp[col].mode().empty else "Unknown"
            X_train_for_imp[col].fillna(val_to_fill, inplace=True)
            imputer_values_train[col] = val_to_fill

        # 그 다음 Inf 값 처리 (수치형 컬럼에 대해서만)
        if pd.api.types.is_numeric_dtype(X_train_for_imp[col]): #  <--- 수치형인지 먼저 확인
            if np.isinf(X_train_for_imp[col].values).any():
                print(f"  Warning: Inf values found in numeric column '{col}' of X_train_for_imp. Replacing with mean.")
                # 수치형 컬럼의 Inf는 평균으로 대체 (NaN은 이미 처리됨)
                # imputer_values_train에 해당 컬럼의 평균이 이미 저장되어 있을 수 있음
                # 만약 NaN이 없었고 Inf만 있었다면, 현재 데이터로 평균 계산
                mean_val = imputer_values_train.get(col, X_train_for_imp[col][~np.isinf(X_train_for_imp[col])].mean())
                X_train_for_imp[col].replace([np.inf, -np.inf], mean_val, inplace=True)
                if pd.isna(mean_val): # 평균 계산이 NaN이 되는 경우 (모든 값이 Inf였을 때)는 0으로 대체
                    X_train_for_imp[col].replace([np.inf, -np.inf], 0, inplace=True)
                    imputer_values_train[col] = 0
                else:
                    imputer_values_train[col] = mean_val # imputer 값 업데이트 또는 설정
        # 범주형/객체형 컬럼은 Inf 값을 가질 가능성이 매우 낮으므로, 여기서는 특별히 처리하지 않음.
        # 만약 문자열 형태의 'inf' 등이 있다면 다른 방식으로 처리 필요.

    # 특성 중요도 계산 및 선택 (결측치와 Inf가 처리된 X_train_for_imp 사용)
    rf_imp = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    # X_numeric_for_imp는 여기서 다시 만들어야 함 (위에서 X_train_for_imp가 변경되었으므로)
    X_numeric_for_imp = X_train_for_imp.select_dtypes(include=np.number).copy() # .copy() 추가

    # X_numeric_for_imp 내부에도 혹시 모를 Inf/NaN이 남아있다면 다시 처리 (방어적 코딩)
    for col_num in X_numeric_for_imp.columns:
        if X_numeric_for_imp[col_num].isnull().any() or np.isinf(X_numeric_for_imp[col_num].values).any():
            print(f"  Re-checking and cleaning NaNs/Infs in numeric column '{col_num}' for RF importance.")
            fill_val_num = X_numeric_for_imp[col_num][~(np.isinf(X_numeric_for_imp[col_num]) | X_numeric_for_imp[col_num].isnull())].mean()
            if pd.isna(fill_val_num): fill_val_num = 0 # 모든 값이 문제였다면 0으로
            X_numeric_for_imp[col_num].replace([np.inf, -np.inf], fill_val_num, inplace=True)
            X_numeric_for_imp[col_num].fillna(fill_val_num, inplace=True)


    if X_numeric_for_imp.empty:
        print("Error: No numeric features left after cleaning for importance calculation."); exit()

    if y_train.isnull().any() or (pd.api.types.is_numeric_dtype(y_train) and np.isinf(y_train.values).any()):
        print("  Warning: NaN/Inf values found in y_train. Replacing with mean.")
        if pd.api.types.is_numeric_dtype(y_train):
            y_train = y_train.replace([np.inf, -np.inf], np.nan)
        y_train = y_train.fillna(y_train.mean())
        if y_train.isnull().all(): # 모든 y_train이 NaN이어서 평균도 NaN인 극단적 경우
            print("  Error: y_train became all NaNs after attempting to fill. Exiting.")
            exit()


    rf_imp.fit(X_numeric_for_imp, y_train)

    importances_df = pd.DataFrame({'Feature': X_numeric_for_imp.columns, 'Importance': rf_imp.feature_importances_})
    importances_df = importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    selected_features = importances_df.head(min(top_n, len(importances_df)))['Feature'].tolist()
    print(f"Top {len(selected_features)} features selected: {selected_features}")

    # imputer_values_train은 selected_features에 대해서만 필요
    imputer_means_selected_train = {feat: imputer_values_train[feat]
                                    for feat in selected_features
                                    if feat in imputer_values_train}

    X_train_sel = X_train_raw[selected_features].copy()
    X_val_sel = X_val_raw[selected_features].copy()
    X_test_sel = X_test_raw[selected_features].copy()

    # 선택된 특성에 대해서 imputer_means_selected_train (또는 전체 imputer_values_train)을 사용하여 결측치 채우기
    # 이 단계에서는 selected_features에 대해서만 imputer_values_train에서 값을 가져와 사용
    X_train_filled = X_train_sel.copy()
    X_val_filled = X_val_sel.copy()
    X_test_filled = X_test_sel.copy()

    for df_part in [X_train_filled, X_val_filled, X_test_filled]:
        for col in selected_features:
            if col in df_part.columns:
                # NaN 처리
                if df_part[col].isnull().any():
                    fill_val_nan = imputer_means_selected_train.get(col)
                    # imputer_means_selected_train에 값이 없는 경우 (예: 모든 값이 NaN이 아니었던 수치형 컬럼)
                    if fill_val_nan is None and pd.api.types.is_numeric_dtype(df_part[col]):
                        fill_val_nan = X_train_for_imp[col].mean() # 원본 train set에서 다시 계산 (X_train_for_imp는 이미 처리됨)
                        if pd.isna(fill_val_nan): fill_val_nan = 0 # 그래도 NaN이면 0
                    elif fill_val_nan is None: # 범주형인데 imputer에 없는 경우
                        fill_val_nan = "Unknown"

                    df_part[col].fillna(fill_val_nan, inplace=True)

                # Inf 처리 (수치형 컬럼에 대해서만)
                if pd.api.types.is_numeric_dtype(df_part[col]):
                    if np.isinf(df_part[col].values).any():
                        fill_val_inf = imputer_means_selected_train.get(col)
                        if fill_val_inf is None or np.isinf(fill_val_inf): # imputer 값이 없거나 inf인 경우
                             # X_train_for_imp에서 해당 컬럼의 (inf 제외한) 평균값을 다시 가져오거나, 0으로 대체
                            clean_mean = X_train_for_imp[col][~np.isinf(X_train_for_imp[col])].mean()
                            fill_val_inf = clean_mean if not pd.isna(clean_mean) else 0

                        df_part[col].replace([np.inf, -np.inf], fill_val_inf, inplace=True)


    scaler_obj = None
    cols_to_scale = []

    if apply_scaling:
        print("  Applying scaling...")
        scaler_obj = MinMaxScaler()
        binary_like_cols = [col for col in selected_features
                            if col in X_train_filled.columns and X_train_filled[col].nunique(dropna=False) <= 2]
        cols_to_scale = [col for col in selected_features
                         if col in X_train_filled.columns and col not in binary_like_cols and \
                            pd.api.types.is_numeric_dtype(X_train_filled[col])]

        if cols_to_scale:
            # 스케일링 전에도 NaN/Inf가 없는지 최종 확인
            for df_part in [X_train_filled, X_val_filled, X_test_filled]:
                for col_s in cols_to_scale:
                    if df_part[col_s].isnull().any() or np.isinf(df_part[col_s].values).any():
                        print(f"  FATAL: NaN/Inf found in '{col_s}' just before scaling. This should not happen. Exiting.")
                        # 여기서 에러를 발생시키거나, 마지막으로 한번 더 처리할 수 있지만, 원인 파악이 중요.
                        # 간단히 평균으로 다시 채우는 코드를 넣을 수도 있음.
                        # 예: df_part[col_s].fillna(X_train_filled[col_s].mean(), inplace=True)
                        # 예: df_part[col_s].replace([np.inf, -np.inf], X_train_filled[col_s].mean(), inplace=True)
                        exit()


            X_train_filled[cols_to_scale] = scaler_obj.fit_transform(X_train_filled[cols_to_scale])
            if not X_val_filled.empty and cols_to_scale: # X_val_filled가 비어있지 않을 때만 transform
                X_val_filled[cols_to_scale] = scaler_obj.transform(X_val_filled[cols_to_scale])
            if not X_test_filled.empty and cols_to_scale: # X_test_filled가 비어있지 않을 때만 transform
                X_test_filled[cols_to_scale] = scaler_obj.transform(X_test_filled[cols_to_scale])
            print(f"  Features scaled: {cols_to_scale}")
        else:
            print("  No features to scale.")
            scaler_obj = None
    else:
        print("  Scaling not applied as per experiment condition.")

    return X_train_filled, X_val_filled, X_test_filled, \
           y_train, y_val, y_test, \
           dates_train, dates_val, dates_test, \
           selected_features, scaler_obj, imputer_means_selected_train, cols_to_scale
           
# --- 6. Model Training and Tuning ---
def train_and_tune_model(model_type, X_train, y_train, X_val, y_val,
                         n_iter_search, scorer, early_stopping_rounds, seed,
                         cv_n_splits):
    print(f"\n--- 6. Training and Tuning {model_type.upper()} Model ---")

    estimator = None
    param_dist = {}
    fit_params_for_search = {} # RandomizedSearchCV의 fit에 전달될 파라미터

    if model_type == "lgbm":
        estimator = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1)
        param_dist = {
            'n_estimators': sp_randint(400, 1500), 'learning_rate': sp_uniform(0.005, 0.095),
            'max_depth': sp_randint(5, 15), 'num_leaves': sp_randint(15, 100),
            'min_child_samples': sp_randint(10, 101), 'subsample': sp_uniform(0.6, 0.4),
            'colsample_bytree': sp_uniform(0.4, 0.6),
            'reg_alpha': sp_uniform(0, 5), 'reg_lambda': sp_uniform(0, 5)
        }
        if X_val is not None and y_val is not None and not X_val.empty:
            # LGBM은 callbacks을 사용
            fit_params_for_search['callbacks'] = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)]
            fit_params_for_search['eval_set'] = [(X_val, y_val)]
            fit_params_for_search['eval_metric'] = 'rmse'

    elif model_type == "rf":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=-1)
        param_dist = {
            'n_estimators': sp_randint(100, 800),
            'max_depth': [None] + list(sp_randint(5, 25).rvs(size=5, random_state=seed)),
            'min_samples_split': sp_randint(2, 20),
            'min_samples_leaf': sp_randint(1, 20),
            'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9]
        }
        # RF는 fit_params_for_search에 특별히 추가할 것 없음

    elif model_type == "xgb":
        # XGBoost의 경우, early_stopping_rounds는 estimator 생성자에 전달
        xgb_init_params = {
            'random_state': seed,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse' # 또는 ['rmse', 'mae'] 등 리스트로 여러개 가능
        }
        if X_val is not None and y_val is not None and not X_val.empty:
            xgb_init_params['early_stopping_rounds'] = early_stopping_rounds
            # RandomizedSearchCV의 fit 메소드에 전달될 fit_params에는 eval_set만 포함
            fit_params_for_search['eval_set'] = [(X_val, y_val)]
            fit_params_for_search['verbose'] = False # XGBoost 자체의 로그 출력 제어 (RandomizedSearchCV의 verbose와 별개)
        
        estimator = xgb.XGBRegressor(**xgb_init_params)

        param_dist = {
            'n_estimators': sp_randint(100, 1000),
            'learning_rate': sp_uniform(0.01, 0.2),
            'max_depth': sp_randint(3, 10),
            'min_child_weight': sp_randint(1, 10),
            'subsample': sp_uniform(0.6, 0.4),
            'colsample_bytree': sp_uniform(0.6, 0.4),
            'gamma': sp_uniform(0, 0.5),
            'reg_alpha': sp_uniform(0, 2), # XGBoost는 reg_alpha, reg_lambda 사용
            'reg_lambda': sp_uniform(0, 2)
        }

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    tscv = TimeSeriesSplit(n_splits=cv_n_splits)

    print(f"  Hyperparameter search space for {model_type.upper()}: {param_dist}")
    print(f"  Number of iterations for RandomizedSearchCV: {n_iter_search}")
    if fit_params_for_search:
        print(f"  Fit parameters for RandomizedSearchCV's fit: {fit_params_for_search.keys()}")


    random_search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter_search, scoring=scorer,
                                     cv=tscv, random_state=seed, n_jobs=-1, verbose=1)

    # fit_params_for_search가 비어있지 않은 경우에만 **로 전달
    if fit_params_for_search:
        random_search.fit(X_train, y_train, **fit_params_for_search)
    else:
        random_search.fit(X_train, y_train)


    print(f"Best parameters for {model_type.upper()}: {random_search.best_params_}")
    print(f"Best CV Score ({scorer}) for {model_type.upper()}: {random_search.best_score_:.4f}")

    tuned_model = random_search.best_estimator_

    # 조기 종료된 경우 best_iteration_ 정보 출력
    if model_type == "lgbm":
        if hasattr(tuned_model, 'best_iteration_') and tuned_model.best_iteration_ is not None:
            print(f"  LGBM Early stopping at iteration: {tuned_model.best_iteration_}")
        # LGBM은 best_estimator_가 이미 최적의 n_estimators로 학습된 상태
    elif model_type == "xgb":
        # XGBoost의 best_estimator_는 RandomizedSearch가 끝난 후 최적 파라미터로 다시 학습되는데,
        # 이때 early_stopping_rounds가 적용된 n_estimators를 가짐 (best_ntree_limit)
        # 또는 실제 학습된 트리의 수를 확인하려면 model.get_booster().best_ntree_limit (만약 조기종료 사용시)
        # 혹은 random_search.best_estimator_.get_params()['n_estimators']는 초기 설정값일 수 있음.
        # 실제 사용된 트리 수는 best_estimator_.n_estimators (scikit-learn wrapper) 또는
        # best_estimator_.get_booster().best_iteration (만약 early stopping이 적용되었다면)
        if hasattr(tuned_model, 'best_iteration') and tuned_model.best_iteration is not None : # XGBoost는 best_iteration (scikit-learn wrapper)
             print(f"  XGBoost (อาจจะ) early stopping around iteration: {tuned_model.best_iteration}") # best_iteration은 학습된 트리 수
        elif hasattr(tuned_model.get_booster(), 'best_ntree_limit') and tuned_model.get_booster().best_ntree_limit != tuned_model.get_params()['n_estimators']:
             print(f"  XGBoost used {tuned_model.get_booster().best_ntree_limit} trees (early stopping likely occurred).")


    val_metrics = None
    if X_val is not None and y_val is not None and not X_val.empty:
        preds_val = tuned_model.predict(X_val)
        val_metrics = {'MAE_model_scale': mean_absolute_error(y_val, preds_val),
                       'RMSE_model_scale': np.sqrt(mean_squared_error(y_val, preds_val)),
                       'R2_model_scale': r2_score(y_val, preds_val)}
        print(f"Validation metrics for {model_type.upper()} (model scale): {val_metrics}")

    return tuned_model, random_search.best_params_, val_metrics
# --- 7. Model Evaluation and Saving ---
def evaluate_and_save_model(model, model_type, X_val, y_val, X_test, y_test,
                                scaler_obj, imputer_means_obj, cols_scaled_list,
                                selected_features_list, best_params,
                                save_dir, visualization_subdir, input_file, ratios_str,
                                val_metrics_tuning, timestamp,
                                dates_val_series, dates_test_series,
                                apply_log_transform_for_eval):

    model_prefix = f"visitor_{model_type}_regressor"
    print(f"\n--- 7. Evaluating and Saving {model_prefix.upper()} ---")

    # 예측값 생성
    preds_val_raw = model.predict(X_val)
    preds_test_raw = model.predict(X_test)

    # 로그 변환된 타겟 되돌리기 (평가를 위해)
    y_val_eval = np.expm1(y_val) if apply_log_transform_for_eval else y_val.copy()
    y_test_eval = np.expm1(y_test) if apply_log_transform_for_eval else y_test.copy()
    preds_val_eval = np.expm1(preds_val_raw) if apply_log_transform_for_eval else preds_val_raw.copy()
    preds_test_eval = np.expm1(preds_test_raw) if apply_log_transform_for_eval else preds_test_raw.copy()
    
    # 음수 예측값 처리 (expm1 후에도 발생 가능성 낮지만, 안전장치)
    preds_val_eval[preds_val_eval < 0] = 0
    preds_test_eval[preds_test_eval < 0] = 0


    # 평가 지표 계산
    val_r2_model_scale = r2_score(y_val, preds_val_raw) # 모델 스케일 R2
    test_r2_model_scale = r2_score(y_test, preds_test_raw) # 모델 스케일 R2

    val_mae_orig = mean_absolute_error(y_val_eval, preds_val_eval)
    val_rmse_orig = np.sqrt(mean_squared_error(y_val_eval, preds_val_eval))
    test_mae_orig = mean_absolute_error(y_test_eval, preds_test_eval)
    test_rmse_orig = np.sqrt(mean_squared_error(y_test_eval, preds_test_eval))

    print(f"Final Validation (Original Scale): MAE={val_mae_orig:.2f}, RMSE={val_rmse_orig:.2f}")
    print(f"Final Validation (Model Scale): R2={val_r2_model_scale:.4f}")
    print(f"Test Performance (Original Scale): MAE={test_mae_orig:.2f}, RMSE={test_rmse_orig:.2f}")
    print(f"Test Performance (Model Scale): R2={test_r2_model_scale:.4f}")

    # 시각화 저장 경로 확인 및 생성
    os.makedirs(visualization_subdir, exist_ok=True)

    # 검증 데이터 시각화
    if dates_val_series is not None and not dates_val_series.empty:
        plt.figure(figsize=(15,6))
        plt.plot(dates_val_series, y_val_eval, label='Actual (Val)', marker='.', linestyle='-')
        plt.plot(dates_val_series, preds_val_eval, label='Predicted (Val)', marker='.', linestyle='--')
        plt.title(f'Actual vs Predicted ({model_prefix} - Validation Set)'); plt.legend(); plt.grid()
        plt.xlabel("Date"); plt.ylabel("Visitor Count")
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(visualization_subdir, f'{model_prefix}_val_predictions_vs_actual.png')); plt.close()

    # 테스트 데이터 시각화
    if dates_test_series is not None and not dates_test_series.empty:
        plt.figure(figsize=(15,6))
        plt.plot(dates_test_series, y_test_eval, label='Actual (Test)', marker='.', linestyle='-')
        plt.plot(dates_test_series, preds_test_eval, label='Predicted (Test)', marker='.', linestyle='--')
        plt.title(f'Actual vs Predicted ({model_prefix} - Test Set)'); plt.legend(); plt.grid()
        plt.xlabel("Date"); plt.ylabel("Visitor Count")
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(visualization_subdir, f'{model_prefix}_test_predictions_vs_actual.png')); plt.close()

    # 결과 로그 저장
    log_path = os.path.join(save_dir, f'{model_prefix}_evaluation_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp}\nInput File: {input_file}\nSplit Ratios: {ratios_str}\n")
        f.write(f"Model Type: {model_type.upper()}\n")
        f.write(f"Best Hyperparameters: {best_params}\n")
        if val_metrics_tuning: f.write(f"Tuning Validation Metrics (model scale): {val_metrics_tuning}\n")
        f.write(f"Final Validation (Original Scale): MAE={val_mae_orig:.2f}, RMSE={val_rmse_orig:.2f}\n")
        f.write(f"Final Validation (Model Scale): R2={val_r2_model_scale:.4f}\n")
        f.write(f"Test Performance (Original Scale): MAE={test_mae_orig:.2f}, RMSE={test_rmse_orig:.2f}\n")
        f.write(f"Test Performance (Model Scale): R2={test_r2_model_scale:.4f}\n")
        f.write(f"Selected Features ({len(selected_features_list)}):\n" + "\n".join(selected_features_list) + "\n")
        if cols_scaled_list:
            f.write(f"Scaled Features ({len(cols_scaled_list)}):\n" + "\n".join(cols_scaled_list) + "\n")
        else:
            f.write("No features were scaled.\n")

    # 모델 패키지 저장
    model_pkg_path = os.path.join(save_dir, f"{model_prefix}_best_model.pkl")
    imputer_means_to_save = imputer_means_obj.copy() if isinstance(imputer_means_obj, dict) else imputer_means_obj.to_dict() if isinstance(imputer_means_obj, pd.Series) else {}

    model_package = {'model': model,
                     'scaler': scaler_obj,
                     'imputer_means': imputer_means_to_save,
                     'cols_scaled_at_fit': cols_scaled_list if cols_scaled_list else [],
                     'features_selected_at_fit': selected_features_list,
                     'best_hyperparameters': best_params,
                     'model_type_trained': model_type, # 실제 학습된 모델 타입
                     'training_timestamp': timestamp,
                     'log_transformed_target': apply_log_transform_for_eval,
                     'test_metrics_original_scale': {'mae': test_mae_orig, 'rmse': test_rmse_orig},
                     'test_metrics_model_scale': {'r2': test_r2_model_scale,
                                                  'mae': mean_absolute_error(y_test, preds_test_raw), # 모델 스케일 MAE
                                                  'rmse': np.sqrt(mean_squared_error(y_test, preds_test_raw))}} # 모델 스케일 RMSE
    joblib.dump(model_package, model_pkg_path)
    print(f"Model package for {model_type.upper()} saved to: {model_pkg_path}")
    
    return { # 결과 요약을 위해 반환
        "Test_MAE_Orig": test_mae_orig,
        "Test_RMSE_Orig": test_rmse_orig,
        "Test_R2_Model_Scale": test_r2_model_scale
    }


# --- Main Execution ---
if __name__ == '__main__':
    all_experiment_results_summary = [] # 모든 실험 결과를 저장할 리스트

    # --- 데이터 로드 (한 번만 수행) ---
    df_raw_master = load_data(INPUT_DATA_FILE, DATE_COLUMN)
    if df_raw_master is None:
        exit()

    for condition_idx, exp_condition in enumerate(EXPERIMENT_CONDITIONS):
        current_exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n\n========== Experiment {condition_idx + 1}/{len(EXPERIMENT_CONDITIONS)}: {exp_condition['name']} (Timestamp: {current_exp_timestamp}) ==========")

        # --- 각 실험별 결과 저장 디렉토리 설정 ---
        # 실험 이름에 타임스탬프를 포함하여 각 실행이 고유한 디렉토리를 갖도록 함
        exp_specific_run_dir = os.path.join(BASE_EXPERIMENT_DIR, f"{exp_condition['name']}_{current_exp_timestamp}")
        exp_visualizations_dir = os.path.join(exp_specific_run_dir, "visualizations")
        os.makedirs(exp_specific_run_dir, exist_ok=True)
        os.makedirs(exp_visualizations_dir, exist_ok=True)
        print(f"Experiment results will be saved in: {exp_specific_run_dir}")

        # --- 초기 전처리 (로그 변환은 전역 설정 따름) ---
        df_processed = initial_preprocess_visitor(
            df_raw_master.copy(), # 원본 데이터의 복사본 사용
            TARGET_VISITOR_COL,
            ORIGINAL_RESCUE_COUNT_COL,
            ORIGINAL_RESCUE_EVENT_COL_FOR_FE,
            apply_log_transform=APPLY_LOG_TRANSFORM_VISITOR_TARGET
        )

        # --- 타겟 변수 분포 시각화 (전처리 후) ---
        plot_distribution(df_processed, TARGET_VISITOR_COL, "After Initial Preprocessing", exp_visualizations_dir)
        plot_boxplot(df_processed, TARGET_VISITOR_COL, "After Initial Preprocessing", exp_visualizations_dir)
        plot_time_series(df_processed, DATE_COLUMN, TARGET_VISITOR_COL, "After Initial Preprocessing", exp_visualizations_dir)


        # --- 이상치 처리 (실험 조건에 따라) ---
        df_for_feature_engineering = df_processed.copy()
        if exp_condition.get("apply_outlier_capping", False):
            df_for_feature_engineering = cap_outliers(
                df_for_feature_engineering, # 이미 로그 변환된 데이터에 적용
                TARGET_VISITOR_COL,
                method=exp_condition.get("outlier_method", OUTLIER_CAPPING_METHOD),
                iqr_multiplier=IQR_MULTIPLIER,
                lower_p=PERCENTILE_LOWER,
                upper_p=PERCENTILE_UPPER
            )
            # 이상치 처리 후 분포 시각화
            plot_distribution(df_for_feature_engineering, TARGET_VISITOR_COL, f"After Outlier Capping ({exp_condition.get('outlier_method', 'iqr')})", exp_visualizations_dir)
            plot_boxplot(df_for_feature_engineering, TARGET_VISITOR_COL, f"After Outlier Capping ({exp_condition.get('outlier_method', 'iqr')})", exp_visualizations_dir)
            plot_time_series(df_for_feature_engineering, DATE_COLUMN, TARGET_VISITOR_COL, f"After Outlier Capping", exp_visualizations_dir)


        # --- 피처 엔지니어링 ---
        # 피처 엔지니어링 시 lag/rolling을 위해 TARGET_VISITOR_COL을 전달 (이 컬럼은 로그변환/이상치처리 되었을 수 있음)
        df_engineered = engineer_features_visitor(
            df_for_feature_engineering,
            DATE_COLUMN,
            TARGET_VISITOR_COL, # 로그변환/이상치처리 된 타겟값을 lag/roll 생성에 사용
            ORIGINAL_RESCUE_EVENT_COL_FOR_FE # 원본 구조 이벤트 컬럼 이름
        )

        # --- 데이터 분할, 특성 선택, 스케일링 ---
        apply_scaling_for_exp = exp_condition.get("apply_scaling", False) # 실험 조건에서 스케일링 여부 가져오기

        X_train_prep, X_val_prep, X_test_prep, \
        y_train, y_val, y_test, \
        dates_train_series, dates_val_series, dates_test_series, \
        selected_features_list, scaler_object, imputer_means_dict, scaled_cols_list = prepare_and_split_data_for_training(
            df_engineered, TARGET_VISITOR_COL, DATE_COLUMN,
            TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, TOP_N_FEATURES_VISITOR,
            apply_scaling=apply_scaling_for_exp # 실험 조건에 따른 스케일링 적용
        )

        # --- 모델 학습 및 튜닝 ---
        model_type_for_exp = exp_condition.get("model_type", "lgbm") # 실험 조건에서 모델 타입 가져오기
        n_iter_for_exp_search = exp_condition.get("n_iter_search", N_ITER_RANDOM_SEARCH_DEFAULT)

        trained_model_instance, best_hyperparams, validation_metrics = train_and_tune_model(
            model_type=model_type_for_exp,
            X_train=X_train_prep,
            y_train=y_train,
            X_val=X_val_prep,
            y_val=y_val,
            n_iter_search=n_iter_for_exp_search,
            scorer=SCORER_VISITOR,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS_LGBM_XGB, # LGBM, XGB에만 해당
            seed=RANDOM_SEED,
            cv_n_splits=CV_N_SPLITS_VISITOR
        )

        # --- 모델 평가 및 저장 ---
        # evaluate_and_save_model 함수가 test 성능 지표를 반환하도록 수정됨
        current_exp_test_metrics = evaluate_and_save_model(
            model=trained_model_instance,
            model_type=model_type_for_exp,
            X_val=X_val_prep, y_val=y_val, X_test=X_test_prep, y_test=y_test,
            scaler_obj=scaler_object,
            imputer_means_obj=imputer_means_dict,
            cols_scaled_list=scaled_cols_list,
            selected_features_list=selected_features_list,
            best_params=best_hyperparams,
            save_dir=exp_specific_run_dir, # 현재 실험 결과 저장 경로
            visualization_subdir=exp_visualizations_dir, # 시각화 저장 경로
            input_file=INPUT_DATA_FILE,
            ratios_str=f"{TRAIN_RATIO*100:.0f}% Train, {VALIDATION_RATIO*100:.0f}% Val, {TEST_RATIO*100:.0f}% Test",
            val_metrics_tuning=validation_metrics,
            timestamp=current_exp_timestamp, # 현재 실험의 타임스탬프
            dates_val_series=dates_val_series, # 검증셋 날짜 시리즈
            dates_test_series=dates_test_series, # 테스트셋 날짜 시리즈
            apply_log_transform_for_eval=APPLY_LOG_TRANSFORM_VISITOR_TARGET # 로그 변환 여부 전달
        )

        # 실험 결과 요약에 추가
        all_experiment_results_summary.append({
            "Experiment_Name": exp_condition['name'],
            "Model_Type": model_type_for_exp,
            "Scaling_Applied": apply_scaling_for_exp,
            "Outlier_Capping_Applied": exp_condition.get("apply_outlier_capping", False),
            "Outlier_Method": exp_condition.get("outlier_method", "N/A") if exp_condition.get("apply_outlier_capping", False) else "N/A",
            "Test_MAE_Original_Scale": current_exp_test_metrics["Test_MAE_Orig"],
            "Test_RMSE_Original_Scale": current_exp_test_metrics["Test_RMSE_Orig"],
            "Test_R2_Model_Scale": current_exp_test_metrics["Test_R2_Model_Scale"],
            "Best_Hyperparameters": best_hyperparams,
            "Results_Directory": exp_specific_run_dir # 결과 저장 경로도 기록
        })
        print(f"Finished Experiment: {exp_condition['name']}")


    # --- 모든 실험 결과 비교 및 요약 ---
    if all_experiment_results_summary:
        results_summary_df = pd.DataFrame(all_experiment_results_summary)
        # 컬럼 순서 보기 좋게 정렬
        ordered_cols = [
            "Experiment_Name", "Model_Type", "Scaling_Applied", "Outlier_Capping_Applied", "Outlier_Method",
            "Test_MAE_Original_Scale", "Test_RMSE_Original_Scale", "Test_R2_Model_Scale",
            "Best_Hyperparameters", "Results_Directory"
        ]
        # 혹시 모를 누락된 컬럼이 있더라도 에러나지 않도록, 있는 컬럼만 선택
        final_ordered_cols = [col for col in ordered_cols if col in results_summary_df.columns]
        results_summary_df = results_summary_df[final_ordered_cols]


        print("\n\n========== All Experiment Results Summary ==========")
        print(results_summary_df.to_string())

        # 요약 결과 CSV 파일로 저장 (전체 실행에 대한 요약이므로 BASE_EXPERIMENT_DIR에 저장)
        # 파일명에 현재 전체 실행 시작 시간을 사용
        summary_csv_filename = f"all_experiments_summary_{CURRENT_TIME_STR}.csv"
        summary_csv_path = os.path.join(BASE_EXPERIMENT_DIR, summary_csv_filename)
        results_summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nExperiment summary saved to: {summary_csv_path}")

        # Test R2 Score 비교 시각화
        if 'Test_R2_Model_Scale' in results_summary_df.columns and not results_summary_df['Test_R2_Model_Scale'].isnull().all():
            plt.figure(figsize=(max(10, len(results_summary_df)*0.5), 7)) # 실험 개수에 따라 너비 조절
            sns.barplot(x='Experiment_Name', y='Test_R2_Model_Scale', data=results_summary_df.sort_values('Test_R2_Model_Scale', ascending=False))
            plt.title(f'Comparison of Test R2 Scores (Model Scale) by Experiment ({CURRENT_TIME_STR})', fontsize=15)
            plt.xlabel("Experiment Name", fontsize=12)
            plt.ylabel("Test R2 Score (Model Scale)", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            r2_comparison_plot_filename = f"experiments_r2_comparison_{CURRENT_TIME_STR}.png"
            r2_comparison_plot_path = os.path.join(BASE_EXPERIMENT_DIR, r2_comparison_plot_filename)
            plt.savefig(r2_comparison_plot_path); plt.close()
            print(f"Experiment R2 comparison plot saved to: {r2_comparison_plot_path}")
        else:
            print("Could not generate R2 comparison plot (no R2 data or all NaN).")

        # Test MAE (Original Scale) 비교 시각화
        if 'Test_MAE_Original_Scale' in results_summary_df.columns and not results_summary_df['Test_MAE_Original_Scale'].isnull().all():
            plt.figure(figsize=(max(10, len(results_summary_df)*0.5), 7))
            sns.barplot(x='Experiment_Name', y='Test_MAE_Original_Scale', data=results_summary_df.sort_values('Test_MAE_Original_Scale', ascending=True)) # MAE는 낮을수록 좋음
            plt.title(f'Comparison of Test MAE (Original Scale) by Experiment ({CURRENT_TIME_STR})', fontsize=15)
            plt.xlabel("Experiment Name", fontsize=12)
            plt.ylabel("Test MAE (Original Scale)", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            mae_comparison_plot_filename = f"experiments_mae_comparison_{CURRENT_TIME_STR}.png"
            mae_comparison_plot_path = os.path.join(BASE_EXPERIMENT_DIR, mae_comparison_plot_filename)
            plt.savefig(mae_comparison_plot_path); plt.close()
            print(f"Experiment MAE comparison plot saved to: {mae_comparison_plot_path}")
        else:
            print("Could not generate MAE comparison plot (no MAE data or all NaN).")

    else:
        print("No experiments were run or no results were collected.")

    print(f"\n========== All Experiments Complete (Overall Timestamp: {CURRENT_TIME_STR}) ==========")