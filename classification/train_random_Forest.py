import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import ast
import os
from datetime import datetime
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import randint as sp_randint
import holidays

# --- 0. Configuration ---
INPUT_DATA_FILE = "/home/imes-server2/sunmin/termP/data/merged_data_with_seorakdong_visitors.csv"
DATE_COLUMN = 'Date'
VISITOR_COUNT_COL = 'Total_Visitor_Count'
RESCUE_COUNT_COL = 'Total_Rescued_Count'
TARGET_RESCUE_COL = 'Rescue_Event'

TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9

TOP_N_FEATURES_RESCUE = 30
RANDOM_SEED = 42 # 일관성을 위한 랜덤 시드

# --- Directory Setup ---
CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_RESULTS_DIR = "./rescue_model_training_results_with_plots" # 디렉토리명 변경
RUN_SPECIFIC_DIR = os.path.join(BASE_RESULTS_DIR, CURRENT_TIME_STR)
VISUALIZATION_SAVE_DIR = os.path.join(RUN_SPECIFIC_DIR, "visualizations") # 시각화 저장용 디렉토리
os.makedirs(RUN_SPECIFIC_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)
print(f"All results will be saved in: {RUN_SPECIFIC_DIR}")

# --- 1. Data Handling Functions ---
def load_data(file_path, date_col):
    # ... (이전과 동일) ...
    print(f"\n--- 1. Loading Data from {file_path} ---")
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
        df.sort_values(by=date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print("Data loaded and sorted.")
        return df
    except FileNotFoundError: print(f"Error: File not found at {file_path}"); exit()
    except Exception as e: print(f"Error loading data: {e}"); exit()

def initial_preprocess_for_rescue(df, date_col, rescue_count_col, visitor_col, target_col_name):
    # ... (이전과 동일) ...
    print(f"\n--- 2. Initial Preprocessing for Rescue Model ---")
    df_p = df.copy()
    if date_col not in df_p.columns: print(f"Error: Date column '{date_col}' not found."); exit()
    if not pd.api.types.is_datetime64_any_dtype(df_p[date_col]):
        try: df_p[date_col] = pd.to_datetime(df_p[date_col]); print(f"'{date_col}' column converted to datetime.")
        except Exception as e: print(f"Error converting '{date_col}' to datetime: {e}"); exit()
    if rescue_count_col not in df_p.columns: print(f"Error: Rescue count column '{rescue_count_col}' not found."); exit()
    df_p[rescue_count_col] = pd.to_numeric(df_p[rescue_count_col], errors='coerce').fillna(0).astype(int)
    df_p[target_col_name] = (df_p[rescue_count_col] > 0).astype(int)
    print(f"Target column '{target_col_name}' created.")
    if visitor_col in df_p.columns: df_p[visitor_col] = pd.to_numeric(df_p[visitor_col], errors='coerce').fillna(0).astype(int)
    list_like_cols_to_parse = ['Accident_Cause_List', 'Accident_Outcome_List']
    for col_parse in list_like_cols_to_parse:
        if col_parse in df_p.columns:
            df_p[col_parse] = df_p[col_parse].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else ([] if pd.isna(x) else x))
    print("Initial preprocessing for rescue model complete.")
    return df_p

# --- 2b. Visualization Helper Functions ---
def plot_class_distribution(series, title, filename, save_dir):
    plt.figure(figsize=(6, 4))
    series.value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename)); plt.close()
    print(f"Class distribution plot saved: {filename}")

def plot_feature_importance(model, feature_names, top_n, title, filename, save_dir):
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_importances = importances.nlargest(top_n)
        plt.figure(figsize=(10, max(6, top_n // 2))) # Dynamic height
        top_importances.sort_values().plot(kind='barh')
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename)); plt.close()
        print(f"Feature importance plot saved: {filename}")
    else:
        print(f"Model {type(model).__name__} does not have feature_importances_ attribute.")

def plot_confusion_matrix_heatmap(cm, classes, title, filename, save_dir, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename)); plt.close()
    print(f"Confusion matrix heatmap saved: {filename}")


# --- 3. Feature Engineering Function (for Rescue Model) ---
def engineer_features_for_rescue(df_input, date_col, visitor_col_for_lag, rescue_event_col_for_lag):
    # ... (이전과 동일한 로직, print문은 필요시 주석 해제) ...
    print("\n--- 3. Engineering Features for Rescue Model ---")
    df_eng = df_input.copy()
    if date_col in df_eng.columns:
        df_eng['is_weekend'] = (df_eng[date_col].dt.dayofweek >= 5).astype(int)
        df_eng['month_num'] = df_eng[date_col].dt.month
        df_eng = pd.get_dummies(df_eng, columns=['month_num'], prefix='month', drop_first=False)
    if visitor_col_for_lag and visitor_col_for_lag in df_eng.columns:
        for lag in [1, 7, 14, 30]: df_eng[f'{visitor_col_for_lag}_Lag{lag}'] = df_eng[visitor_col_for_lag].shift(lag).fillna(0)
        for window in [7, 14, 30]: df_eng[f'{visitor_col_for_lag}_Roll{window}_Mean'] = df_eng[visitor_col_for_lag].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
    else:
        for lag in [1, 7, 14, 30]: df_eng[f'{VISITOR_COUNT_COL}_Lag{lag}'] = 0
        for window in [7, 14, 30]: df_eng[f'{VISITOR_COUNT_COL}_Roll{window}_Mean'] = 0
    if rescue_event_col_for_lag and rescue_event_col_for_lag in df_eng.columns:
        df_eng['rescue_event_yesterday'] = df_eng[rescue_event_col_for_lag].shift(1).fillna(0)
    else: df_eng['rescue_event_yesterday'] = 0
    weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'), 'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    for condition, (col, window, new_col) in weather_rules.items():
        if col in df_eng.columns:
            is_cond = (df_eng[col] > 0 if condition == 'rain' else df_eng[col] < 0).astype(int)
            df_eng[new_col] = (is_cond.rolling(window=window, min_periods=window).sum() == window).astype(int).fillna(0)
    if rescue_event_col_for_lag and rescue_event_col_for_lag in df_eng.columns and date_col in df_eng.columns:
        try:
            df_eng['year_week_iso_temp'] = df_eng[date_col].dt.isocalendar().year.astype(str) + '-' + df_eng[date_col].dt.isocalendar().week.astype(str).str.zfill(2)
            weekly_rescue_agg = df_eng.groupby('year_week_iso_temp')[rescue_event_col_for_lag].transform('max')
            df_eng['last_week_date_temp'] = df_eng[date_col] - pd.to_timedelta(7, unit='D')
            df_eng['last_year_week_iso_temp'] = df_eng['last_week_date_temp'].dt.isocalendar().year.astype(str) + '-' + df_eng['last_week_date_temp'].dt.isocalendar().week.astype(str).str.zfill(2)
            week_map = pd.Series(weekly_rescue_agg.values, index=df_eng['year_week_iso_temp']).drop_duplicates().to_dict()
            df_eng['rescue_in_last_week'] = df_eng['last_year_week_iso_temp'].map(week_map).fillna(0).astype(int)
            df_eng.drop(columns=['year_week_iso_temp', 'last_week_date_temp', 'last_year_week_iso_temp'], inplace=True, errors='ignore')
        except: df_eng['rescue_in_last_week'] = 0
    else: df_eng['rescue_in_last_week'] = 0
    time_cols = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    for orig, new in time_cols.items():
        if orig in df_eng.columns:
            try: df_eng[new] = pd.to_datetime(df_eng[orig], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
            except: df_eng[new] = -1
            df_eng.drop(columns=[orig], inplace=True, errors='ignore')
    temp_c, hum_c = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    if temp_c in df_eng.columns and hum_c in df_eng.columns:
        df_eng['Temp_Humidity_Interaction'] = df_eng[temp_c].fillna(df_eng[temp_c].mean()) * df_eng[hum_c].fillna(df_eng[hum_c].mean())
    if date_col in df_eng.columns and 'is_weekend' in df_eng.columns: # Ensure is_weekend exists
        min_year_h, max_year_h = df_eng[date_col].dt.year.min(), df_eng[date_col].dt.year.max()
        if pd.isna(min_year_h) or pd.isna(max_year_h): current_year_h = df_eng[date_col].dt.year.iloc[0]; kr_holidays_r = holidays.KR(years=current_year_h)
        else: kr_holidays_r = holidays.KR(years=range(min_year_h, max_year_h + 1))
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

# --- 4. Data Splitting, Feature Selection & Scaling (for Training Rescue Model) ---
# prepare_and_split_data_for_rescue_training 함수 수정
def prepare_and_split_data_for_rescue_training(df_engineered, target_col, date_col, 
                                               train_r, val_r, test_r, top_n):
    print("\n--- 4. Preparing and Splitting Data for Rescue Model Training ---")
    
    explicit_exclude_train = [date_col, target_col, RESCUE_COUNT_COL, VISITOR_COUNT_COL, 
                              'Accident_Cause_List', 'Accident_Outcome_List']
    potential_features_train = [col for col in df_engineered.columns if col not in explicit_exclude_train]
    
    if not potential_features_train: # potential_features_train이 비어있는지 확인
        print("오류: X를 구성할 피처가 없습니다. explicit_exclude_train 목록을 확인하세요."); exit()

    X_full = df_engineered[potential_features_train].copy()
    y_full = df_engineered[target_col].copy()
    dates_full = df_engineered[date_col].copy() # dates_full 정의

    # 데이터 분할
    n_total = len(X_full)
    n_train = int(n_total * train_r); n_val = int(n_total * val_r)
    
    X_train_raw, y_train, dates_train = X_full.iloc[:n_train], y_full.iloc[:n_train], dates_full.iloc[:n_train] # <--- dates_train 할당
    X_val_raw, y_val, dates_val = X_full.iloc[n_train:n_train+n_val], y_full.iloc[n_train:n_train+n_val], dates_full.iloc[n_train:n_train+n_val]
    X_test_raw, y_test, dates_test = X_full.iloc[n_train+n_val:], y_full.iloc[n_train+n_val:], dates_full.iloc[n_train+n_val:]
    print(f"Data split: Train {len(X_train_raw)}, Validation {len(X_val_raw)}, Test {len(X_test_raw)}")
    
    X_train_for_imp = X_train_raw.copy(); imputer_means_train = {}
    
    for col in X_train_for_imp.columns:
        if X_train_for_imp[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train_for_imp[col]): mean_val = X_train_for_imp[col].mean(); X_train_for_imp[col].fillna(mean_val, inplace=True); imputer_means_train[col] = mean_val
            else: mode_val = X_train_for_imp[col].mode()[0] if not X_train_for_imp[col].mode().empty else "Unknown"; X_train_for_imp[col].fillna(mode_val, inplace=True); imputer_means_train[col] = mode_val
    
    rf_imp = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1) # Classifier for importance
    X_numeric_for_imp = X_train_for_imp.select_dtypes(include=np.number)
    
    if X_numeric_for_imp.empty: 
        print("Error: No numeric features for importance calculation."); exit()
    
    rf_imp.fit(X_numeric_for_imp, y_train)
    
    importances = pd.DataFrame({'Feature': X_numeric_for_imp.columns, 'Importance': rf_imp.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    selected_features = importances.head(min(top_n, len(importances)))['Feature'].tolist()
    
    print(f"Top {len(selected_features)} features selected: {selected_features}")
    plot_feature_importance(rf_imp, X_numeric_for_imp.columns, top_n, "Feature Importances (Rescue Model Training)", "rescue_feature_importances.png", VISUALIZATION_SAVE_DIR) # 피처 중요도 시각화
    
    X_train_sel = X_train_raw[selected_features].copy(); X_val_sel = X_val_raw[selected_features].copy(); X_test_sel = X_test_raw[selected_features].copy()
    
    imputer_means_selected_train = {k: v for k, v in imputer_means_train.items() if k in selected_features}
    X_train_filled = X_train_sel.fillna(imputer_means_selected_train)
    X_val_filled = X_val_sel.fillna(imputer_means_selected_train)
    X_test_filled = X_test_sel.fillna(imputer_means_selected_train)
    scaler = MinMaxScaler(); 
    
    binary_like_cols_rescue = [col for col in selected_features if X_train_filled[col].nunique(dropna=False) <= 2]
    one_hot_prefixes_rescue = ('month_num_', 'day_of_week_') # 실제 사용된 prefix로 수정
    cols_to_scale = [col for col in selected_features if col not in binary_like_cols_rescue and not any(col.startswith(p) for p in one_hot_prefixes_rescue) and pd.api.types.is_numeric_dtype(X_train_filled[col])]
    
    if cols_to_scale:
        X_train_filled[cols_to_scale] = scaler.fit_transform(X_train_filled[cols_to_scale])
        X_val_filled[cols_to_scale] = scaler.transform(X_val_filled[cols_to_scale])
        X_test_filled[cols_to_scale] = scaler.transform(X_test_filled[cols_to_scale])
        print(f"Features scaled: {cols_to_scale}")
    else: scaler = None; cols_to_scale = []
    return X_train_filled, X_val_filled, X_test_filled, y_train, y_val, y_test, \
           dates_train, dates_val, dates_test, \
           selected_features, scaler, imputer_means_selected_train, cols_to_scale

# --- 5. Model Training and Tuning (for Rescue Model) ---
def train_and_tune_rescue_model(X_train, y_train, X_val, y_val, param_dist, n_iter, scorer, seed):
    # ... (이전과 동일, RandomForestClassifier 사용) ...
    print(f"\n--- 5. Training and Tuning Rescue Model (RandomForestClassifier) ---")
    estimator = RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight='balanced')
    cv_strategy = 5 
    random_search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring=scorer, cv=cv_strategy, random_state=seed, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    scorer_name = scorer if isinstance(scorer, str) else scorer._score_func.__name__ # make_scorer 객체 처리
    print(f"Best CV Score ({scorer_name}): {random_search.best_score_:.4f}")
    tuned_model = random_search.best_estimator_
    preds_val = tuned_model.predict(X_val); probs_val = tuned_model.predict_proba(X_val)[:, 1]
    val_metrics = {'Accuracy': accuracy_score(y_val, preds_val), 'Precision': precision_score(y_val, preds_val, zero_division=0),
                   'Recall': recall_score(y_val, preds_val, zero_division=0), 'F1-Score': f1_score(y_val, preds_val, zero_division=0),
                   'ROC_AUC': roc_auc_score(y_val, probs_val)}
    print(f"Validation metrics (default threshold): {val_metrics}")
    cm_val = confusion_matrix(y_val, preds_val)
    plot_confusion_matrix_heatmap(cm_val, classes=[0,1], title='Validation Confusion Matrix', filename='rescue_val_cm_heatmap.png', save_dir=VISUALIZATION_SAVE_DIR)
    return tuned_model, random_search.best_params_, val_metrics

# --- 6. Model Evaluation and Saving (for Rescue Model) ---
def evaluate_and_save_rescue_model(model, X_test, y_test, scaler_obj, imputer_means_obj, 
                                   cols_scaled_list, selected_features_list, best_params, 
                                   save_dir, input_file, ratios_str, val_metrics, timestamp):
    # ... (이전과 동일, Confusion Matrix 시각화 추가) ...
    print(f"\n--- 6. Evaluating and Saving Rescue Model ---")
    preds_test = model.predict(X_test); probs_test = model.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, probs_test)
    print(f"Test ROC AUC: {roc_auc_test:.4f}")
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'ROC Curve (Test AUC = {roc_auc_test:.4f})'); plt.plot([0,1],[0,1], linestyle='--', label='Random Guess'); plt.title('ROC Curve (Test Set)'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(); plt.savefig(os.path.join(save_dir, 'rescue_roc_curve.png')); plt.close()
    prec, rec, thresh = precision_recall_curve(y_test, probs_test)
    plt.figure(figsize=(8,6)); plt.plot(rec[:-1], prec[:-1], marker='.', label='Precision-Recall Curve'); plt.title('PR Curve (Test Set)'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid(); plt.savefig(os.path.join(save_dir, 'rescue_pr_curve.png')); plt.close()
    f1s = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9); opt_thresh = thresh[np.argmax(f1s)] if len(f1s) > 0 else 0.5
    print(f"Optimal Threshold (F1 max): {opt_thresh:.4f}")
    preds_adj = (probs_test >= opt_thresh).astype(int)
    final_metrics = {'Accuracy': accuracy_score(y_test, preds_adj), 'Precision': precision_score(y_test, preds_adj, zero_division=0), 
                     'Recall': recall_score(y_test, preds_adj, zero_division=0), 'F1-Score': f1_score(y_test, preds_adj, zero_division=0),
                     'ROC_AUC': roc_auc_test}
    print(f"Test Performance (Adjusted Threshold {opt_thresh:.4f}): {final_metrics}")
    cm_test = confusion_matrix(y_test, preds_adj)
    plot_confusion_matrix_heatmap(cm_test, classes=[0,1], title='Test Confusion Matrix (Adjusted Threshold)', filename='rescue_test_cm_heatmap.png', save_dir=save_dir)
    log_path = os.path.join(save_dir, 'rescue_model_evaluation_log.txt') # ... (로그 내용 이전과 동일) ...
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp}\nInput File: {input_file}\nSplit Ratios: {ratios_str}\nBest Hyperparameters: {best_params}\nValidation Metrics: {val_metrics}\nTest Performance (Adj. Threshold): {final_metrics}\nOptimal Threshold: {opt_thresh:.4f}\nSelected Features ({len(selected_features_list)}):\n" + "\n".join(selected_features_list))
    model_pkg_path = os.path.join(save_dir, "rescue_model_package.pkl")
    model_package = {'model': model, 'scaler': scaler_obj, 'imputer_means': imputer_means_obj, 'cols_scaled_at_fit': cols_scaled_list, 
                     'features': selected_features_list, 'best_hyperparameters': best_params, 'optimal_threshold': opt_thresh,
                     'model_type': type(model).__name__, 'training_timestamp': timestamp, 'test_metrics_adjusted_thresh': final_metrics}
    joblib.dump(model_package, model_pkg_path)
    print(f"Rescue model package saved to: {model_pkg_path}")

# --- Main Execution ---
if __name__ == '__main__':
    df_raw = load_data(INPUT_DATA_FILE, DATE_COLUMN)
    
    # 타겟 변수 분포 시각화 (원본)
    plot_class_distribution(df_raw[RESCUE_COUNT_COL].apply(lambda x: 1 if x > 0 else 0), 
                            "Original Target Distribution (Rescue Event)", 
                            "rescue_target_dist_raw.png", VISUALIZATION_SAVE_DIR)

    df_processed = initial_preprocess_for_rescue(df_raw, DATE_COLUMN, RESCUE_COUNT_COL, 
                                                 VISITOR_COUNT_COL, TARGET_RESCUE_COL)
    
    # 전처리 후 타겟 변수 분포 시각화
    plot_class_distribution(df_processed[TARGET_RESCUE_COL], 
                            "Preprocessed Target Distribution (Rescue Event)", 
                            "rescue_target_dist_processed.png", VISUALIZATION_SAVE_DIR)

    df_engineered = engineer_features_for_rescue(df_processed, DATE_COLUMN, 
                                                 VISITOR_COUNT_COL, 
                                                 TARGET_RESCUE_COL) 

    X_train, X_val, X_test, y_train, y_val, y_test, \
    _, _, _, \
    selected_features, scaler, imputer_means, cols_scaled = prepare_and_split_data_for_rescue_training(
        df_engineered, TARGET_RESCUE_COL, DATE_COLUMN,
        TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, TOP_N_FEATURES_RESCUE
    )
    
    # SMOTE 적용 전 학습 데이터 클래스 분포
    plot_class_distribution(y_train, "Train Target Distribution (Before SMOTE)", 
                            "rescue_train_dist_before_smote.png", VISUALIZATION_SAVE_DIR)
    
    print(f"\nApplying SMOTE to training data. Original shape: {X_train.shape}")
    smote = SMOTE(random_state=RANDOM_SEED) # 전역 RANDOM_SEED 사용
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. Resampled shape: {X_train_sm.shape}")
    print(f"Class distribution after SMOTE: {pd.Series(y_train_sm).value_counts(normalize=True)}")
    
    # SMOTE 적용 후 학습 데이터 클래스 분포
    plot_class_distribution(y_train_sm, "Train Target Distribution (After SMOTE)", 
                            "rescue_train_dist_after_smote.png", VISUALIZATION_SAVE_DIR)

    rf_param_dist = {
        'n_estimators': sp_randint(100, 500), 'max_depth': sp_randint(5, 21),
        'min_samples_split': sp_randint(2, 11), 'min_samples_leaf': sp_randint(1, 11),
        'max_features': ['sqrt', 'log2', None], # None은 모든 피처 사용
        'class_weight': ['balanced', 'balanced_subsample', None] 
    }
    N_ITER_RESCUE_SEARCH = 50 
    SCORER_RESCUE = 'recall' 

    rescue_model, best_params, val_metrics = train_and_tune_rescue_model(
        X_train_sm, y_train_sm, X_val, y_val, 
        rf_param_dist, N_ITER_RESCUE_SEARCH, SCORER_RESCUE, RANDOM_SEED
    )
    
    evaluate_and_save_rescue_model(
        rescue_model, X_test, y_test, scaler, imputer_means, cols_scaled,
        selected_features, best_params, RUN_SPECIFIC_DIR, INPUT_DATA_FILE,
        f"{TRAIN_RATIO*100}%/{VALIDATION_RATIO*100}%/{TEST_RATIO*100}%",
        val_metrics, CURRENT_TIME_STR
    )
    
    print("\n========== Rescue Model Training Pipeline Complete ==========")