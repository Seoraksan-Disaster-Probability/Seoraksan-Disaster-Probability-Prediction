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
import platform
from matplotlib import font_manager

# --- 0. Configuration ---
INPUT_DATA_FILE = "data/preprocessed_merged_data.csv" # 이미 전처리된 데이터 사용 가정
DATE_COLUMN = 'Date'
VISITOR_COUNT_COL_FOR_FE = 'Total_Visitor_Count' # 조난 모델 FE 시 사용할 탐방객 수 컬럼명
RESCUE_COUNT_COL_FOR_TARGET = 'Total_Rescued_Count'  # 조난 모델 타겟 생성 기준
TARGET_RESCUE_COL = 'Rescue_Event'        # 최종 타겟 변수 (0 또는 1)

TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9

TOP_N_FEATURES_RESCUE = 30
RANDOM_SEED = 42
APPLY_SMOTE_GLOBAL = True # SMOTE 적용 여부 전역 플래그 (이름 변경 고려)

# --- 실험 조건 정의 --- <--- 이 부분을 추가하거나 주석 해제하세요!
EXPERIMENT_CONDITIONS_RESCUE = [
    {"name": "RF_SMOTE_RecallScoring", "model_type": "rf", "apply_smote": True, "scorer": "recall", "n_iter": 50, "class_weight_rf": None}, # SMOTE 사용 시 class_weight는 None 또는 기본값
    {"name": "RF_NoSMOTE_Balanced_F1Scoring", "model_type": "rf", "apply_smote": False, "scorer": "f1", "n_iter": 50, "class_weight_rf": "balanced"},
    {"name": "RF_SMOTE_F1Scoring", "model_type": "rf", "apply_smote": True, "scorer": "f1", "n_iter": 50, "class_weight_rf": None},
    # 다른 모델 또는 조건 추가 가능
    # {"name": "LGBM_SMOTE_RecallScoring", "model_type": "lgbm", "apply_smote": True, "scorer": "recall", "n_iter": 50},
]
# -------------------------

# --- Directory Setup ---
CURRENT_TIME_STR_MAIN = datetime.now().strftime("%Y%m%d_%H%M%S") # 변수명 일관성
BASE_RESULTS_DIR_MAIN = "./rescue_model_experiment_runs_final" # 디렉토리명 명확화
# RUN_SPECIFIC_DIR 은 루프 내에서 exp_run_dir로 생성됨
# VISUALIZATION_SAVE_DIR 도 루프 내에서 exp_viz_dir로 생성됨
os.makedirs(BASE_RESULTS_DIR_MAIN, exist_ok=True) # 실험 전체 기본 디렉토리 생성
print(f"All experiment results will be saved under: {BASE_RESULTS_DIR_MAIN}")

# --- Matplotlib Font Setup ---
if platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else:
    try: plt.rc('font', family='NanumGothic')
    except: print("Warning: NanumGothic font not found. Korean in plots might be broken.")
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Data Handling Functions ---
def load_data(file_path, date_col):
    print(f"\n--- 1. Loading Data from {file_path} ---")
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
        df.sort_values(by=date_col, inplace=True); df.reset_index(drop=True, inplace=True)
        print("Data loaded and sorted."); return df
    except FileNotFoundError: print(f"Error: File not found at {file_path}"); exit()
    except Exception as e: print(f"Error loading data: {e}"); exit()

def initial_preprocess_for_rescue(df, date_col, rescue_count_col, visitor_col, target_col_name):
    print(f"\n--- 2. Initial Preprocessing for Rescue Model ---")
    df_p = df.copy()
    if date_col not in df_p.columns: print(f"Error: Date column '{date_col}' not found."); exit()
    if not pd.api.types.is_datetime64_any_dtype(df_p[date_col]):
        try: df_p[date_col] = pd.to_datetime(df_p[date_col])
        except Exception as e: print(f"Error converting '{date_col}' to datetime: {e}"); exit()
    if rescue_count_col not in df_p.columns: print(f"Error: Rescue count col '{rescue_count_col}' not found."); exit()
    df_p[rescue_count_col] = pd.to_numeric(df_p[rescue_count_col], errors='coerce').fillna(0).astype(int)
    df_p[target_col_name] = (df_p[rescue_count_col] > 0).astype(int)
    if visitor_col in df_p.columns: df_p[visitor_col] = pd.to_numeric(df_p[visitor_col], errors='coerce').fillna(0).astype(int)
    list_like_cols = ['Accident_Cause_List', 'Accident_Outcome_List']
    for col_p in list_like_cols:
        if col_p in df_p.columns:
            df_p[col_p] = df_p[col_p].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else ([] if pd.isna(x) else x))
    print("Initial preprocessing complete."); return df_p

# --- 2b. Visualization Helper Functions ---
def plot_class_distribution(series, title, filename, save_dir):
    plt.figure(figsize=(6,4))
    series.value_counts().plot(kind='bar')
    plt.title(title); plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Plot saved: {filename}")
    
def plot_feature_importance(model, feature_names, top_n, title, filename, save_dir):
    if hasattr(model, 'feature_importances_'): 
        importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
        plt.figure(figsize=(10, max(6,top_n//2)))
        importances.sort_values().plot(kind='barh')
        plt.title(title); plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(); print(f"Plot saved: {filename}")
        
    else: 
        print(f"Model {type(model).__name__} lacks feature_importances_.")
def plot_confusion_matrix_heatmap(cm, classes, title, filename, save_dir, normalize=False, cmap=plt.cm.Blues):
    if normalize: 
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]; fmt='.2f'
    else: 
        fmt='d'
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close(); print(f"Plot saved: {filename}")

# --- 3. Feature Engineering Function (for Rescue Model) ---
def engineer_features_for_rescue(df_input, date_col, visitor_col_name, rescue_event_col_name):
    print("\n--- 3. Engineering Features for Rescue Model ---")
    df_eng = df_input.copy()
    df_eng.reset_index(drop=True, inplace=True)

    # Date-based features
    if date_col in df_eng.columns:
        df_eng['is_weekend'] = (df_eng[date_col].dt.dayofweek >= 5).astype(int)
        df_eng['month_num_for_dummies'] = df_eng[date_col].dt.month # 임시 컬럼명
        df_eng = pd.get_dummies(df_eng, columns=['month_num_for_dummies'], prefix='month', drop_first=False)
        # month_10과 같은 특정 월 피처는 여기서 생성하거나, 또는 원핫인코딩된 컬럼(예: month_10)을 직접 사용
        if 'month_10' not in df_eng.columns and 'month_num_for_dummies_10' in df_eng.columns : # pd.get_dummies 결과에 따라
             df_eng.rename(columns={'month_num_for_dummies_10':'month_10'}, inplace=True)
        elif 'month_10' not in df_eng.columns:
             df_eng['month_10'] = 0 # 10월이 없는 경우 대비

    # Lagged/Rolling features (using visitor_col_name)
    if visitor_col_name and visitor_col_name in df_eng.columns:
        for lag in [1, 7, 14, 30]: df_eng[f'{visitor_col_name}_Lag{lag}'] = df_eng[visitor_col_name].shift(lag).fillna(0)
        for window in [7, 14, 30]: df_eng[f'{visitor_col_name}_Roll{window}_Mean'] = df_eng[visitor_col_name].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
    else:
        for lag in [1, 7, 14, 30]: df_eng[f'{VISITOR_COUNT_COL_FOR_FE}_Lag{lag}'] = 0 # Fallback
        for window in [7, 14, 30]: df_eng[f'{VISITOR_COUNT_COL_FOR_FE}_Roll{window}_Mean'] = 0
        if visitor_col_name: print(f"  Warning: Visitor column '{visitor_col_name}' not found for Lag/Rolling. Filled with 0.")

    # Yesterday's rescue event (using rescue_event_col_name)
    if rescue_event_col_name and rescue_event_col_name in df_eng.columns:
        df_eng['rescue_event_yesterday'] = df_eng[rescue_event_col_name].shift(1).fillna(0)
    else:
        df_eng['rescue_event_yesterday'] = 0
        if rescue_event_col_name: print(f"  Warning: Rescue event column '{rescue_event_col_name}' not found. 'rescue_event_yesterday' filled with 0.")

    # Consecutive weather
    weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'), 
                     'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    for cond, (col_w, win, new_c) in weather_rules.items():
        if col_w in df_eng.columns:
            is_c = (df_eng[col_w] > 0 if cond == 'rain' else df_eng[col_w] < 0).astype(int)
            df_eng[new_c] = (is_c.rolling(window=win, min_periods=win).sum() == win).astype(int).fillna(0)

    # Rescue in last week
    if rescue_event_col_name and rescue_event_col_name in df_eng.columns and date_col in df_eng.columns:
        try:
            df_eng['temp_year_week'] = df_eng[date_col].dt.isocalendar().year.astype(str) + '-' + df_eng[date_col].dt.isocalendar().week.astype(str).str.zfill(2)
            weekly_agg = df_eng.groupby('temp_year_week')[rescue_event_col_name].transform('max')
            df_eng['temp_last_week_date'] = df_eng[date_col] - pd.to_timedelta(7, unit='D')
            df_eng['temp_last_year_week'] = df_eng['temp_last_week_date'].dt.isocalendar().year.astype(str) + '-' + df_eng['temp_last_week_date'].dt.isocalendar().week.astype(str).str.zfill(2)
            week_map_s = pd.Series(weekly_agg.values, index=df_eng['temp_year_week']).drop_duplicates()
            df_eng['rescue_in_last_week'] = df_eng['temp_last_year_week'].map(week_map_s).fillna(0).astype(int)
        except: df_eng['rescue_in_last_week'] = 0
        finally: df_eng.drop(columns=['temp_year_week', 'temp_last_week_date', 'temp_last_year_week'], inplace=True, errors='ignore')
    else: df_eng['rescue_in_last_week'] = 0
        
    # Time transformation
    time_map = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    for orig, new in time_map.items():
        if orig in df_eng.columns:
            try: df_eng[new] = pd.to_datetime(df_eng[orig], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
            except: df_eng[new] = -1
            df_eng.drop(columns=[orig], inplace=True, errors='ignore')
                
    # Interaction term
    temp_c, hum_c = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    if temp_c in df_eng.columns and hum_c in df_eng.columns:
        mean_t, mean_h = df_eng[temp_c].mean(), df_eng[hum_c].mean() # Use overall mean for imputation consistency
        df_eng['Temp_Humidity_Interaction'] = df_eng[temp_c].fillna(mean_t) * df_eng[hum_c].fillna(mean_h)
        
    # Holiday features
    if date_col in df_eng.columns and 'is_weekend' in df_eng.columns:
        min_yr, max_yr = df_eng[date_col].dt.year.min(), df_eng[date_col].dt.year.max()
        if pd.isna(min_yr) or pd.isna(max_yr): yr = df_eng[date_col].dt.year.iloc[0]; kr_h = holidays.KR(years=yr)
        else: kr_h = holidays.KR(years=range(min_yr, max_yr + 1))
        df_eng['temp_is_official_holiday'] = df_eng[date_col].apply(lambda d: d in kr_h)
        df_eng['temp_is_day_off'] = (df_eng['is_weekend'] | df_eng['temp_is_official_holiday']).astype(int)
        thresh = 3
        df_eng['temp_day_off_group'] = (df_eng['temp_is_day_off'].diff() != 0).astype(int).cumsum()
        df_eng['temp_consecutive_off'] = df_eng.groupby('temp_day_off_group')['temp_is_day_off'].transform('sum')
        df_eng['is_base_long_h'] = ((df_eng['temp_is_day_off'] == 1) & (df_eng['temp_consecutive_off'] >= thresh)).astype(int)
        df_eng['temp_prev_off'] = df_eng['temp_is_day_off'].shift(1).fillna(0).astype(int)
        df_eng['temp_next_off'] = df_eng['temp_is_day_off'].shift(-1).fillna(0).astype(int)
        df_eng['is_bridge_cand'] = ((df_eng['temp_is_day_off'] == 0) & (df_eng['temp_prev_off'] == 1) & (df_eng['temp_next_off'] == 1)).astype(int)
        df_eng['is_extended_h'] = df_eng['is_base_long_h'].copy()
        df_eng.loc[df_eng['is_bridge_cand'].values == 1, 'is_extended_h'] = 1
        df_eng.loc[df_eng['is_bridge_cand'].shift(-1).fillna(False).values, 'is_extended_h'] = 1
        df_eng.loc[df_eng['is_bridge_cand'].shift(1).fillna(False).values, 'is_extended_h'] = 1
        df_eng['temp_final_h_group'] = (df_eng['is_extended_h'].diff() != 0).astype(int).cumsum()
        df_eng['temp_consecutive_ext_off'] = df_eng.groupby('temp_final_h_group')['is_extended_h'].transform('sum')
        df_eng['is_final_long_holiday_rescue'] = ((df_eng['is_extended_h'] == 1) & (df_eng['temp_consecutive_ext_off'] >= thresh)).astype(int)
        cols_to_drop_h = ['temp_is_official_holiday', 'temp_is_day_off', 'temp_day_off_group', 'temp_consecutive_off', 
                          'is_base_long_h', 'temp_prev_off', 'temp_next_off', 'is_bridge_cand', 
                          'is_extended_h', 'temp_final_h_group', 'temp_consecutive_ext_off']
        df_eng.drop(columns=cols_to_drop_h, inplace=True, errors='ignore')
    print("Feature engineering for rescue model complete.")
    return df_eng

# --- 4. Data Splitting, Feature Selection & Scaling (for Training Rescue Model) ---
def prepare_and_split_data_for_rescue_training(df_engineered, target_col, date_col, 
                                               train_r, val_r, test_r, top_n, viz_save_dir):
    print("\n--- 4. Preparing and Splitting Data for Rescue Model Training ---")
    exclude_from_X = [date_col, target_col, RESCUE_COUNT_COL_FOR_TARGET, VISITOR_COUNT_COL_FOR_FE, 
                      'Accident_Cause_List', 'Accident_Outcome_List']
    potential_X_cols = [col for col in df_engineered.columns if col not in exclude_from_X]
    if not potential_X_cols: print("Error: No potential features for X."); exit()

    X_full = df_engineered[potential_X_cols].copy()
    y_full = df_engineered[target_col].copy()
    dates_full = df_engineered[date_col].copy()

    n_total = len(X_full); n_train = int(n_total * train_r); n_val = int(n_total * val_r)
    X_train_raw = X_full.iloc[:n_train]; y_train = y_full.iloc[:n_train]; dates_train = dates_full.iloc[:n_train]
    X_val_raw = X_full.iloc[n_train:n_train+n_val]; y_val = y_full.iloc[n_train:n_train+n_val]; dates_val = dates_full.iloc[n_train:n_train+n_val]
    X_test_raw = X_full.iloc[n_train+n_val:]; y_test = y_full.iloc[n_train+n_val:]; dates_test = dates_full.iloc[n_train+n_val:]
    print(f"Data split: Train {len(X_train_raw)}, Validation {len(X_val_raw)}, Test {len(X_test_raw)}")

    X_train_for_imp = X_train_raw.copy(); imputer_stats_train = {}
    for col in X_train_for_imp.columns:
        if X_train_for_imp[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train_for_imp[col]): fill_val = X_train_for_imp[col].mean()
            else: fill_val = X_train_for_imp[col].mode()[0] if not X_train_for_imp[col].mode().empty else "Unknown"
            X_train_for_imp[col].fillna(fill_val, inplace=True); imputer_stats_train[col] = fill_val
    
    imp_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    X_numeric_imp = X_train_for_imp.select_dtypes(include=np.number)
    if X_numeric_imp.empty: print("Error: No numeric features for importance calc."); selected_features = X_train_for_imp.columns.tolist()
    else:
        imp_model.fit(X_numeric_imp, y_train)
        imp_df = pd.DataFrame({'Feature': X_numeric_imp.columns, 'Importance': imp_model.feature_importances_}).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        selected_features = imp_df.head(min(top_n, len(imp_df)))['Feature'].tolist()
        plot_feature_importance(imp_model, X_numeric_imp.columns, top_n, "Feature Importances (Rescue Training)", "rescue_feat_imp.png", viz_save_dir)
    print(f"Top {len(selected_features)} features selected: {selected_features}")

    imputer_stats_selected = {feat: imputer_stats_train[feat] for feat in selected_features if feat in imputer_stats_train}
    X_train_sel = X_train_raw[selected_features].fillna(imputer_stats_selected)
    X_val_sel = X_val_raw[selected_features].fillna(imputer_stats_selected)
    X_test_sel = X_test_raw[selected_features].fillna(imputer_stats_selected)
    
    scaler_obj = MinMaxScaler()
    binary_cols = [col for col in selected_features if X_train_sel[col].nunique(dropna=False) <= 2]
    one_hot_prefixes = ('month_rescue_', 'day_of_week_') # Ensure these are actual prefixes from your FE
    cols_to_scale = [col for col in selected_features if col not in binary_cols and not any(col.startswith(p) for p in one_hot_prefixes) and pd.api.types.is_numeric_dtype(X_train_sel[col])]
    
    if cols_to_scale:
        X_train_sel[cols_to_scale] = scaler_obj.fit_transform(X_train_sel[cols_to_scale])
        X_val_sel[cols_to_scale] = scaler_obj.transform(X_val_sel[cols_to_scale])
        X_test_sel[cols_to_scale] = scaler_obj.transform(X_test_sel[cols_to_scale])
        print(f"Features scaled: {cols_to_scale}")
    else: scaler_obj = None; cols_to_scale = []
        
    return X_train_sel, X_val_sel, X_test_sel, y_train, y_val, y_test, \
           dates_train, dates_val, dates_test, \
           selected_features, scaler_obj, imputer_stats_selected, cols_to_scale

# --- 5. Model Training and Tuning (for Rescue Model) ---
def train_and_tune_rescue_model(X_train, y_train, X_val, y_val, 
                                model_type_str, param_dist, n_iter, scorer, seed, 
                                viz_save_path): # <--- 시각화 저장 경로 인자 추가
    print(f"\n--- 5. Training and Tuning Rescue Model ({model_type_str.upper()}) ---")
    if model_type_str.lower() == 'rf':
        # estimator 생성 시 class_weight는 param_dist를 통해 RandomizedSearchCV가 설정함
        estimator = RandomForestClassifier(random_state=seed, n_jobs=-1) 
    else: 
        raise ValueError(f"Unsupported model type: {model_type_str}")

    cv = 5 
    random_search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring=scorer, cv=cv, random_state=seed, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    tuned_model = random_search.best_estimator_
    print(f"Best parameters: {best_params}")
    print(f"Best CV Score ({scorer}): {best_score:.4f}")

    preds_val = tuned_model.predict(X_val); probs_val = tuned_model.predict_proba(X_val)[:, 1]
    val_metrics = {'Accuracy': accuracy_score(y_val, preds_val), 
                   'Precision': precision_score(y_val, preds_val, zero_division=0),
                   'Recall': recall_score(y_val, preds_val, zero_division=0), 
                   'F1-Score': f1_score(y_val, preds_val, zero_division=0),
                   'ROC_AUC': roc_auc_score(y_val, probs_val)}
    print(f"Validation metrics (default threshold): {val_metrics}")
    cm_val = confusion_matrix(y_val, preds_val)
    plot_confusion_matrix_heatmap(cm_val, [0,1], f'Validation CM ({model_type_str.upper()})', f'rescue_val_cm_{model_type_str.lower()}.png', save_dir=viz_save_path)
    return tuned_model, best_params, val_metrics

# --- 6. Model Evaluation and Saving (for Rescue Model) ---
def evaluate_and_save_rescue_model(model, X_test, y_test, scaler, imputer_stats, cols_scaled,
                                   selected_feats, best_params_model, save_path, input_src_file, 
                                   split_ratios_str, val_metrics_results, time_stamp, model_name_prefix_str):
    print(f"\n--- 6. Evaluating and Saving {model_name_prefix_str.upper()} ---")
    preds_test_final = model.predict(X_test); probs_test_final = model.predict_proba(X_test)[:, 1]
    roc_auc_final = roc_auc_score(y_test, probs_test_final); print(f"Test ROC AUC: {roc_auc_final:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test, probs_test_final) # ROC Curve data
    plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_final:.2f})'); plt.plot([0,1],[0,1],'--'); plt.title(f'ROC Curve ({model_name_prefix_str})'); plt.savefig(os.path.join(save_path, f'{model_name_prefix_str}_roc.png')); plt.close()
    
    prec, rec, thresh = precision_recall_curve(y_test, probs_test_final) # PR Curve data
    plt.figure(figsize=(8,6)); plt.plot(rec[:-1], prec[:-1], marker='.'); plt.title(f'PR Curve ({model_name_prefix_str})'); plt.savefig(os.path.join(save_path, f'{model_name_prefix_str}_pr.png')); plt.close()
    
    f1_scores_pr = (2*prec[:-1]*rec[:-1])/(prec[:-1]+rec[:-1]+1e-9)
    optimal_threshold = thresh[np.argmax(f1_scores_pr)] if len(f1_scores_pr)>0 and len(prec[:-1])>0 else 0.5
    print(f"Optimal Threshold (F1 max): {optimal_threshold:.4f}")
    
    preds_adjusted_final = (probs_test_final >= optimal_threshold).astype(int)
    final_eval_metrics = {'Accuracy': accuracy_score(y_test, preds_adjusted_final), 
                          'Precision': precision_score(y_test, preds_adjusted_final, zero_division=0), 
                          'Recall': recall_score(y_test, preds_adjusted_final, zero_division=0), 
                          'F1-Score': f1_score(y_test, preds_adjusted_final, zero_division=0),
                          'ROC_AUC': roc_auc_final}
    print(f"Test Performance (Adj. Threshold {optimal_threshold:.4f}): {final_eval_metrics}")
    cm_test_final = confusion_matrix(y_test, preds_adjusted_final)
    plot_confusion_matrix_heatmap(cm_test_final, [0,1], f'Test CM ({model_name_prefix_str}, Adj. Thresh)', f'{model_name_prefix_str}_test_cm.png', save_path)
    
    log_file_path = os.path.join(save_path, f'{model_name_prefix_str}_eval_log.txt')
    with open(log_file_path, 'w', encoding='utf-8') as f:
        log_content = (f"Timestamp: {time_stamp}\nInput File: {input_src_file}\nSplit Ratios: {split_ratios_str}\n"
                       f"Best Hyperparameters: {best_params_model}\nValidation Metrics: {val_metrics_results}\n"
                       f"Test Performance (Adj. Threshold): {final_eval_metrics}\nOptimal Threshold: {optimal_threshold:.4f}\n"
                       f"Selected Features ({len(selected_feats)}):\n" + "\n".join(selected_feats))
        f.write(log_content)
        
    model_package_file_path = os.path.join(save_path, f"{model_name_prefix_str}_package.pkl")
    imputer_to_save_dict = {k:v for k,v in imputer_stats.items() if k in selected_feats}
    model_package_content = {'model': model, 'scaler': scaler, 'imputer_means': imputer_to_save_dict, 
                             'cols_scaled_at_fit': cols_scaled if cols_scaled else [], 
                             'features': selected_feats, 'best_hyperparameters': best_params_model, 
                             'optimal_threshold': optimal_threshold, 'model_type': type(model).__name__, 
                             'training_timestamp': time_stamp, 'test_metrics_adjusted_thresh': final_eval_metrics}
    joblib.dump(model_package_content, model_package_file_path)
    print(f"Model package saved to: {model_package_file_path}")

# --- Main Execution ---
if __name__ == '__main__':
    all_experiment_results = []

    for exp_condition in EXPERIMENT_CONDITIONS_RESCUE:
        exp_name = exp_condition['name']
        print(f"\n\n========== Experiment: {exp_name} ==========")
        
        current_exp_run_dir = os.path.join(BASE_RESULTS_DIR_MAIN, f"{exp_name}_{CURRENT_TIME_STR_MAIN}")
        current_exp_viz_dir = os.path.join(current_exp_run_dir, "visualizations")
        os.makedirs(current_exp_run_dir, exist_ok=True)
        os.makedirs(current_exp_viz_dir, exist_ok=True)
        print(f"Results for this experiment in: {current_exp_run_dir}")

        df_raw_data = load_data(INPUT_DATA_FILE, DATE_COLUMN)
        plot_class_distribution(df_raw_data[RESCUE_COUNT_COL_FOR_TARGET].apply(lambda x: 1 if x > 0 else 0), 
                                "Original Target (Rescue Event)", "rescue_target_dist_raw.png", current_exp_viz_dir)

        df_preprocessed_data = initial_preprocess_for_rescue(df_raw_data, DATE_COLUMN, RESCUE_COUNT_COL_FOR_TARGET, 
                                                             VISITOR_COUNT_COL_FOR_FE, TARGET_RESCUE_COL)
        plot_class_distribution(df_preprocessed_data[TARGET_RESCUE_COL], 
                                "Preprocessed Target (Rescue Event)", "rescue_target_dist_processed.png", current_exp_viz_dir)

        df_engineered_data = engineer_features_for_rescue(df_preprocessed_data, DATE_COLUMN, 
                                                          VISITOR_COUNT_COL_FOR_FE, TARGET_RESCUE_COL) 

        X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, \
        _, _, _, \
        selected_features_p, scaler_p, imputer_means_p, cols_scaled_p = prepare_and_split_data_for_rescue_training(
            df_engineered_data, TARGET_RESCUE_COL, DATE_COLUMN,
            TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, 
            TOP_N_FEATURES_RESCUE, current_exp_viz_dir
        )
        
        X_train_to_fit, y_train_to_fit = X_train_p, y_train_p
        if exp_condition.get("apply_smote", False):
            plot_class_distribution(y_train_p, "Train Target (Before SMOTE)", "rescue_train_dist_before_smote.png", current_exp_viz_dir)
            smote_obj = SMOTE(random_state=RANDOM_SEED)
            X_train_to_fit, y_train_to_fit = smote_obj.fit_resample(X_train_p, y_train_p)
            print(f"SMOTE applied. Resampled shape: {X_train_to_fit.shape}")
            plot_class_distribution(y_train_to_fit, "Train Target (After SMOTE)", "rescue_train_dist_after_smote.png", current_exp_viz_dir)
        else:
            print("\nSMOTE not applied.")

        # 모델 타입별 파라미터 분포 정의
        current_model_type = exp_condition.get("model_type", "rf")
        param_dist_current_exp = {}

        if current_model_type == "rf":
            # RandomForestClassifier의 class_weight는 RandomizedSearchCV의 param_distributions에 포함
            # condition에서 class_weight_rf 값을 가져와서 리스트 형태로 param_dist_current_exp에 추가
            
            # SMOTE를 사용하지 않고 class_weight_rf가 None이면 'balanced'를 기본으로 사용
            rf_class_weight_value = exp_condition.get("class_weight_rf")
            if rf_class_weight_value is None and not exp_condition.get("apply_smote", False):
                rf_class_weight_value = 'balanced'
            
            param_dist_current_exp = {
                'n_estimators': sp_randint(100, 500), 
                'max_depth': sp_randint(5, 21),
                'min_samples_split': sp_randint(2, 11), 
                'min_samples_leaf': sp_randint(1, 11),
                'max_features': ['sqrt', 'log2', None],
                # class_weight는 RandomizedSearchCV가 선택할 수 있도록 리스트로 전달
                # 또는, 특정 값을 고정하고 싶다면 해당 값만 리스트에 넣음
                'class_weight': [rf_class_weight_value] # <--- 수정된 부분: param_dist에 포함
            }
        # elif current_model_type == "lgbm":
            # param_dist_current_exp = { ... } # LightGBM용 파라미터 분포

        # train_and_tune_rescue_model 호출 시 class_weight_option 인자 제거
        trained_model_exp, best_params_exp, val_metrics_exp = train_and_tune_rescue_model(
            X_train_to_fit, y_train_to_fit, X_val_p, y_val_p, 
            model_type_str=current_model_type,
            param_dist=param_dist_current_exp, 
            n_iter=exp_condition.get("n_iter", 50), 
            scorer=exp_condition.get("scorer", "recall"), 
            seed=RANDOM_SEED,
            viz_save_path=current_exp_viz_dir # <--- 시각화 저장 경로 전달
        )
        
        evaluate_and_save_rescue_model(
            trained_model_exp, X_test_p, y_test_p, scaler_p, imputer_means_p, cols_scaled_p,
            selected_features_p, best_params_exp, 
            current_exp_run_dir, 
            INPUT_DATA_FILE,
            f"{TRAIN_RATIO*100:.0f}%/{VALIDATION_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}%",
            val_metrics_exp, 
            CURRENT_TIME_STR_MAIN, # Main timestamp for all experiments in this run
            model_name_prefix_str=exp_condition['name']
        )
        
        saved_pkg_path_exp = os.path.join(current_exp_run_dir, f"{exp_condition['name']}_package.pkl")
        if os.path.exists(saved_pkg_path_exp):
            loaded_pkg = joblib.load(saved_pkg_path_exp)
            test_metrics_adj = loaded_pkg.get('test_metrics_adjusted_thresh', {})
            all_experiment_results.append({
                "Experiment_Name": exp_condition['name'],
                "Test_Accuracy_Adj": test_metrics_adj.get('Accuracy', np.nan),
                "Test_Precision_Adj": test_metrics_adj.get('Precision', np.nan),
                "Test_Recall_Adj": test_metrics_adj.get('Recall', np.nan),
                "Test_F1_Score_Adj": test_metrics_adj.get('F1-Score', np.nan),
                "Test_ROC_AUC": test_metrics_adj.get('ROC_AUC', np.nan),
                "Optimal_Threshold": loaded_pkg.get('optimal_threshold', np.nan),
                "Best_Params": best_params_exp
            })

    if all_experiment_results:
        summary_df_all_exp = pd.DataFrame(all_experiment_results)
        print("\n\n========== All Rescue Model Experiment Results Summary ==========")
        print(summary_df_all_exp.to_string())
        summary_file_path_all_exp = os.path.join(BASE_RESULTS_DIR_MAIN, f"all_rescue_experiments_summary_{CURRENT_TIME_STR_MAIN}.csv")
        summary_df_all_exp.to_csv(summary_file_path_all_exp, index=False, encoding='utf-8-sig')
        print(f"\nAll experiment summary saved to: {summary_file_path_all_exp}")

        if 'Test_F1_Score_Adj' in summary_df_all_exp.columns:
            plt.figure(figsize=(12, len(summary_df_all_exp) * 0.5 + 2)) # Dynamic height
            sns.barplot(x='Test_F1_Score_Adj', y='Experiment_Name', data=summary_df_all_exp.sort_values('Test_F1_Score_Adj', ascending=False), orient='h')
            plt.title('Comparison of Test F1-Scores (Adjusted Threshold) by Experiment')
            plt.xlabel('Test F1-Score (Adjusted Threshold)'); plt.ylabel('Experiment Name')
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_RESULTS_DIR_MAIN, f"all_rescue_exp_f1_comparison_{CURRENT_TIME_STR_MAIN}.png")); plt.close()
            print("All experiment F1 comparison plot saved.")
            
    print("\n========== All Rescue Model Experiments Complete ==========")
