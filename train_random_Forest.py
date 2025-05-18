import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # 현재 주력 모델
# import lightgbm as lgb # 다른 모델 추가 시 필요
# import xgboost as xgb # 다른 모델 추가 시 필요
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
INPUT_DATA_FILE = "data/preprocessed_data.csv" # 이미 전처리된 데이터 사용 가정
DATE_COLUMN = 'Date'
VISITOR_COUNT_COL_FOR_FE = 'Total_Visitor_Count' # 조난 모델 FE 시 사용할 탐방객 수 컬럼명
RESCUE_COUNT_COL_FOR_TARGET = 'Total_Rescued_Count'
TARGET_RESCUE_COL = 'Rescue_Event'

CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_RESULTS_DIR = "./classification/visitor_model_training_results" # 디렉토리명 변경
RUN_SPECIFIC_DIR = os.path.join(BASE_RESULTS_DIR, CURRENT_TIME_STR)
VISUALIZATION_DIR = os.path.join(RUN_SPECIFIC_DIR, "visualizations")
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9

TOP_N_FEATURES_RESCUE = 30
RANDOM_SEED = 42

# --- 실험 조건 정의 ---
EXPERIMENT_CONDITIONS_RESCUE = [
    {"name": "RF_SMOTE_RecallScoring", "model_type": "rf", "apply_smote": True, "scorer": "recall", "n_iter": 50},
    {"name": "RF_NoSMOTE_BalancedClassWeight_F1Scoring", "model_type": "rf", "apply_smote": False, "scorer": "f1", "n_iter": 50, "class_weight_rf": "balanced"},
    # {"name": "LGBM_SMOTE_RecallScoring", "model_type": "lgbm", "apply_smote": True, "scorer": "recall", "n_iter": 50}, # LGBM 추가 예시
]

# --- Directory Setup ---
CURRENT_TIME_STR_MAIN = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_RESULTS_DIR_MAIN = "./rescue_model_experiment_runs"
os.makedirs(BASE_RESULTS_DIR_MAIN, exist_ok=True)
print(f"All experiment results will be saved under: {BASE_RESULTS_DIR_MAIN}")

# --- Matplotlib Font Setup ---
# (이전과 동일)
if platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
else:
    try: plt.rc('font', family='NanumGothic')
    except: print("Warning: NanumGothic font not found. Korean in plots might be broken.")
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Data Handling and Initial Preprocessing ---
# (load_data, initial_preprocess_for_rescue - 이전과 동일)
def load_data(file_path, date_col):
    print(f"\n--- Loading Data from {file_path} ---")
    try:
        df = pd.read_csv(file_path, parse_dates=[date_col])
        df.sort_values(by=date_col, inplace=True); df.reset_index(drop=True, inplace=True)
        print("Data loaded and sorted."); return df
    except: print(f"Error loading {file_path}"); exit()

def initial_preprocess_for_rescue(df, date_col, rescue_count_col, visitor_col, target_col_name):
    print(f"\n--- Initial Preprocessing for Rescue Model ---")
    df_p = df.copy()
    if date_col not in df_p.columns: print(f"Error: Date column '{date_col}' not found."); exit()
    if not pd.api.types.is_datetime64_any_dtype(df_p[date_col]):
        try: df_p[date_col] = pd.to_datetime(df_p[date_col]);
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
# (plot_class_distribution, plot_feature_importance, plot_confusion_matrix_heatmap - 이전과 동일)
def plot_class_distribution(series, title, filename, save_dir):
    plt.figure(figsize=(6,4)); series.value_counts().plot(kind='bar'); plt.title(title); plt.xlabel("Class"); plt.ylabel("Frequency"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(save_dir, filename)); plt.close(); print(f"Plot saved: {filename}")
def plot_feature_importance(model, feature_names, top_n, title, filename, save_dir):
    if hasattr(model, 'feature_importances_'): importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n); plt.figure(figsize=(10, max(6,top_n//2))); importances.sort_values().plot(kind='barh'); plt.title(title); plt.xlabel("Importance"); plt.tight_layout(); plt.savefig(os.path.join(save_dir, filename)); plt.close(); print(f"Plot saved: {filename}")
    else: print(f"Model {type(model).__name__} lacks feature_importances_.")
def plot_confusion_matrix_heatmap(cm, classes, title, filename, save_dir, normalize=False, cmap=plt.cm.Blues):
    if normalize: cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]; fmt='.2f'
    else: fmt='d'
    plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes); plt.title(title); plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout(); plt.savefig(os.path.join(save_dir, filename)); plt.close(); print(f"Plot saved: {filename}")

# --- 3. Feature Engineering Function (for Rescue Model) ---
def engineer_features_for_rescue(df_input, date_col, visitor_col_for_fe, rescue_event_col_for_fe):
    # ... (이전 train_random_Forest.py의 engineer_features 함수 로직과 동일하게 채워야 함) ...
    # 이 함수는 VISITOR_COUNT_COL_FOR_FE (탐방객 수)와 ORIGINAL_RESCUE_EVENT_COL_FOR_FE (원본 조난 이벤트)를
    # 인자로 받아 해당 컬럼명으로 Lag/Rolling 등을 생성해야 함.
    print("\n--- 3. Engineering Features for Rescue Model ---")
    df_eng = df_input.copy(); df_eng.reset_index(drop=True, inplace=True)
    # (여기에 train_random_Forest.py의 engineer_features 함수 로직을 정확히 복사/붙여넣기)
    # 예시로 일부만 남김 (실제로는 전체 로직 필요)
    if date_col in df_eng.columns:
        df_eng['is_weekend'] = (df_eng[date_col].dt.dayofweek >= 5).astype(int)
        df_eng['month_num_rescue'] = df_eng[date_col].dt.month
        df_eng = pd.get_dummies(df_eng, columns=['month_num_rescue'], prefix='month_rescue', drop_first=False)
    if visitor_col_for_fe and visitor_col_for_fe in df_eng.columns:
        for lag in [1, 7, 14, 30]: df_eng[f'{visitor_col_for_fe}_Lag{lag}'] = df_eng[visitor_col_for_fe].shift(lag).fillna(0)
    # ... (나머지 FE 로직) ...
    print("Feature engineering for rescue model complete.")
    return df_eng
def prepare_and_split_data_for_rescue_training(
    df_engineered_input, 
    target_column_name, 
    date_column_name, 
    train_ratio, 
    validation_ratio, 
    test_ratio, # test_ratio 인자 추가
    top_n_features,
    visualization_save_directory
):
    print("\n--- 4. Preparing and Splitting Data for Rescue Model Training ---")

    explicit_exclude_columns = [
        date_column_name, 
        target_column_name, 
        RESCUE_COUNT_COL_FOR_TARGET, 
        VISITOR_COUNT_COL_FOR_FE, # VISITOR_COUNT_COL_FOR_FE는 전역 변수
        'Accident_Cause_List', 
        'Accident_Outcome_List'
    ]
    
    feature_candidate_columns = [
        col for col in df_engineered_input.columns if col not in explicit_exclude_columns
    ]

    if not feature_candidate_columns:
        print("Error: No potential features for X. Check explicit_exclude_columns.")
        exit()

    X_full_features = df_engineered_input[feature_candidate_columns].copy()
    y_full_target = df_engineered_input[target_column_name].copy()
    dates_full_series = df_engineered_input[date_column_name].copy() # <--- dates_full 정의

    # 데이터 분할
    num_total_samples = len(X_full_features)
    num_train_samples = int(num_total_samples * train_ratio)
    num_validation_samples = int(num_total_samples * validation_ratio)

    X_train_raw = X_full_features.iloc[:num_train_samples]
    y_train = y_full_target.iloc[:num_train_samples]
    dates_train = dates_full_series.iloc[:num_train_samples] # <--- dates_train 생성

    X_validation_raw = X_full_features.iloc[num_train_samples : num_train_samples + num_validation_samples]
    y_validation = y_full_target.iloc[num_train_samples : num_train_samples + num_validation_samples]
    dates_validation = dates_full_series.iloc[num_train_samples : num_train_samples + num_validation_samples] # <--- dates_validation 생성

    X_test_raw = X_full_features.iloc[num_train_samples + num_validation_samples:]
    y_test = y_full_target.iloc[num_train_samples + num_validation_samples:]
    dates_test = dates_full_series.iloc[num_train_samples + num_validation_samples:] # <--- dates_test 생성
    
    print(f"Data split: Train {len(X_train_raw)}, Validation {len(X_validation_raw)}, Test {len(X_test_raw)}")

    # ... (이하 특성 중요도 계산, 결측치 처리, 스케일링 로직은 이전 답변과 동일) ...
    X_train_for_importance = X_train_raw.copy()
    imputer_statistics_train = {} 
    for col in X_train_for_importance.columns:
        if X_train_for_importance[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train_for_importance[col]):
                fill_value = X_train_for_importance[col].mean()
            else:
                fill_value = X_train_for_importance[col].mode()
                if not fill_value.empty: fill_value = fill_value[0]
                else: fill_value = "Unknown" 
            X_train_for_importance[col].fillna(fill_value, inplace=True)
            imputer_statistics_train[col] = fill_value
    importance_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    X_numeric_for_importance = X_train_for_importance.select_dtypes(include=np.number)
    if X_numeric_for_importance.empty: print("Error: No numeric features for importance calculation."); selected_feature_names = X_train_for_importance.columns.tolist()
    else:
        importance_model.fit(X_numeric_for_importance, y_train)
        feature_importances_df = pd.DataFrame({'Feature': X_numeric_for_importance.columns, 'Importance': importance_model.feature_importances_})
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        num_to_select = min(top_n_features, len(feature_importances_df))
        selected_feature_names = feature_importances_df.head(num_to_select)['Feature'].tolist()
        plot_feature_importance(importance_model, X_numeric_for_importance.columns, top_n_features, "Feature Importances (Rescue Model Training)", "rescue_model_feature_importances.png", visualization_save_directory)
    print(f"Top {len(selected_feature_names)} features selected: {selected_feature_names}")
    imputer_statistics_for_selected_features = {feature: imputer_statistics_train[feature] for feature in selected_feature_names if feature in imputer_statistics_train}
    X_train_selected = X_train_raw[selected_feature_names].copy(); X_validation_selected = X_validation_raw[selected_feature_names].copy(); X_test_selected = X_test_raw[selected_feature_names].copy()
    X_train_imputed = X_train_selected.fillna(imputer_statistics_for_selected_features); X_validation_imputed = X_validation_selected.fillna(imputer_statistics_for_selected_features); X_test_imputed = X_test_selected.fillna(imputer_statistics_for_selected_features)
    scaler_object = MinMaxScaler()
    binary_like_columns = [col for col in selected_feature_names if X_train_imputed[col].nunique(dropna=False) <= 2]
    one_hot_encoded_prefixes = ('month_rescue_', 'day_of_week_') 
    columns_to_scale = [col for col in selected_feature_names if col not in binary_like_columns and not any(col.startswith(p) for p in one_hot_encoded_prefixes) and pd.api.types.is_numeric_dtype(X_train_imputed[col])]
    X_train_scaled = X_train_imputed.copy(); X_validation_scaled = X_validation_imputed.copy(); X_test_scaled = X_test_imputed.copy()
    if columns_to_scale:
        X_train_scaled[columns_to_scale] = scaler_object.fit_transform(X_train_imputed[columns_to_scale])
        X_validation_scaled[columns_to_scale] = scaler_object.transform(X_validation_imputed[columns_to_scale])
        X_test_scaled[columns_to_scale] = scaler_object.transform(X_test_imputed[columns_to_scale])
        print(f"Features scaled: {columns_to_scale}")
    else: scaler_object = None; columns_to_scale = []
    
    return X_train_scaled, X_validation_scaled, X_test_scaled, \
           y_train, y_validation, y_test, \
           dates_train, dates_validation, dates_test, selected_feature_names, scaler_object, \
           imputer_statistics_for_selected_features, columns_to_scale

# --- 5. Model Training and Tuning (for Rescue Model) ---
def train_and_tune_rescue_model(X_train, y_train, X_val, y_val, model_type_str, param_dist, n_iter, scorer, seed, class_weight_option=None):
    print(f"\n--- 5. Training and Tuning Rescue Model ({model_type_str.upper()}) ---")
    if model_type_str.lower() == 'rf':
        estimator = RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight=class_weight_option) # class_weight 인자 사용
    # elif model_type_str.lower() == 'lgbm':
        # estimator = lgb.LGBMClassifier(random_state=seed, n_jobs=-1, class_weight=class_weight_option, verbosity=-1)
        # fit_params = {}
        # if X_val is not None and y_val is not None and not X_val.empty:
        #     fit_params['callbacks'] = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        #     fit_params['eval_set'] = [(X_val, y_val)]
        #     fit_params['eval_metric'] = 'logloss' # 또는 'auc'
    else: raise ValueError(f"Unsupported model type for rescue: {model_type_str}")

    cv = 5 # 또는 TimeSeriesSplit(n_splits=CV_N_SPLITS_RESCUE)
    random_search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring=scorer, cv=cv, random_state=seed, n_jobs=-1, verbose=1)
    
    # if model_type_str.lower() == 'lgbm' and fit_params: random_search.fit(X_train, y_train, **fit_params)
    # else: random_search.fit(X_train, y_train)
    random_search.fit(X_train, y_train) # 현재는 RF만 가정

    print(f"Best parameters: {random_search.best_params_}")
    scorer_name = scorer; print(f"Best CV Score ({scorer_name}): {random_search.best_score_:.4f}")
    tuned_model = random_search.best_estimator_
    preds_val = tuned_model.predict(X_val); probs_val = tuned_model.predict_proba(X_val)[:, 1]
    val_metrics = {'Accuracy': accuracy_score(y_val, preds_val), 'Precision': precision_score(y_val, preds_val, zero_division=0),
                   'Recall': recall_score(y_val, preds_val, zero_division=0), 'F1-Score': f1_score(y_val, preds_val, zero_division=0),
                   'ROC_AUC': roc_auc_score(y_val, probs_val)}
    print(f"Validation metrics (default threshold): {val_metrics}")
    cm_val = confusion_matrix(y_val, preds_val)
    plot_confusion_matrix_heatmap(cm_val, classes=[0,1], title=f'Validation CM ({model_type_str.upper()})', 
                                  filename=f'rescue_val_cm_{model_type_str.lower()}.png', save_dir=VISUALIZATION_DIR)
    return tuned_model, random_search.best_params_, val_metrics

# --- 6. Model Evaluation and Saving (for Rescue Model) ---
def evaluate_and_save_rescue_model(model, X_test, y_test, scaler_obj, imputer_means_obj, 
                                   cols_scaled_list, selected_features_list, best_params, 
                                   save_dir, input_file, ratios_str, val_metrics, timestamp, model_name_prefix):
    # ... (이전과 동일, model_name_prefix 사용) ...
    print(f"\n--- 6. Evaluating and Saving {model_name_prefix.upper()} ---")
    preds_test = model.predict(X_test); probs_test = model.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, probs_test); print(f"Test ROC AUC: {roc_auc_test:.4f}")
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_test:.2f})'); plt.plot([0,1],[0,1],'--'); plt.title(f'ROC Curve ({model_name_prefix})'); plt.savefig(os.path.join(save_dir, f'{model_name_prefix}_roc.png')); plt.close()
    prec, rec, thresh = precision_recall_curve(y_test, probs_test)
    plt.figure(figsize=(8,6)); plt.plot(rec[:-1], prec[:-1], marker='.'); plt.title(f'PR Curve ({model_name_prefix})'); plt.savefig(os.path.join(save_dir, f'{model_name_prefix}_pr.png')); plt.close()
    f1s = (2*prec[:-1]*rec[:-1])/(prec[:-1]+rec[:-1]+1e-9); opt_thresh = thresh[np.argmax(f1s)] if len(f1s)>0 and len(prec[:-1])>0 else 0.5
    print(f"Optimal Threshold (F1 max): {opt_thresh:.4f}")
    preds_adj = (probs_test >= opt_thresh).astype(int)
    final_metrics = {'Accuracy': accuracy_score(y_test, preds_adj), 'Precision': precision_score(y_test, preds_adj, zero_division=0), 
                     'Recall': recall_score(y_test, preds_adj, zero_division=0), 'F1-Score': f1_score(y_test, preds_adj, zero_division=0),
                     'ROC_AUC': roc_auc_test}
    print(f"Test Performance (Adj. Threshold {opt_thresh:.4f}): {final_metrics}")
    cm_test = confusion_matrix(y_test, preds_adj)
    plot_confusion_matrix_heatmap(cm_test, classes=[0,1], title=f'Test CM ({model_name_prefix}, Adj. Thresh)', filename=f'{model_name_prefix}_test_cm.png', save_dir=save_dir)
    log_path = os.path.join(save_dir, f'{model_name_prefix}_eval_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f: f.write(f"Timestamp: {timestamp}\nInput: {input_file}\nSplit: {ratios_str}\nParams: {best_params}\nVal Metrics: {val_metrics}\nTest Metrics (Adj): {final_metrics}\nOpt Thresh: {opt_thresh:.4f}\nFeatures ({len(selected_features_list)}):\n" + "\n".join(selected_features_list))
    model_pkg_path = os.path.join(save_dir, f"{model_name_prefix}_package.pkl") # 일관된 패키지명
    imputer_to_save = {k:v for k,v in imputer_means_obj.items() if k in selected_features_list}
    model_package = {'model': model, 'scaler': scaler_obj, 'imputer_means': imputer_to_save, 'cols_scaled_at_fit': cols_scaled_list if cols_scaled_list else [], 
                     'features': selected_features_list, 'best_hyperparameters': best_params, 'optimal_threshold': opt_thresh,
                     'model_type': type(model).__name__, 'training_timestamp': timestamp, 'test_metrics_adjusted_thresh': final_metrics}
    joblib.dump(model_package, model_pkg_path); print(f"Model package saved: {model_pkg_path}")

# --- Main Execution ---
if __name__ == '__main__':
    all_experiment_results_summary = [] # 모든 실험 결과 요약 저장

    for exp_idx, condition in enumerate(EXPERIMENT_CONDITIONS_RESCUE):
        print(f"\n\n========== Experiment {exp_idx + 1}: {condition['name']} ==========")
        exp_run_dir = os.path.join(BASE_RESULTS_DIR_MAIN, f"{condition['name']}_{CURRENT_TIME_STR_MAIN}") # 실험별 하위 디렉토리
        exp_viz_dir = os.path.join(exp_run_dir, "visualizations")
        os.makedirs(exp_run_dir, exist_ok=True)
        os.makedirs(exp_viz_dir, exist_ok=True)
        print(f"Results for this experiment will be saved in: {exp_run_dir}")

        df_raw = load_data(INPUT_DATA_FILE, DATE_COLUMN)
        plot_class_distribution(df_raw[RESCUE_COUNT_COL_FOR_TARGET].apply(lambda x: 1 if x > 0 else 0), 
                                "Original Target (Rescue Event)", "rescue_target_dist_raw.png", exp_viz_dir)

        df_processed = initial_preprocess_for_rescue(df_raw, DATE_COLUMN, RESCUE_COUNT_COL_FOR_TARGET, 
                                                     VISITOR_COUNT_COL_FOR_FE, TARGET_RESCUE_COL)
        plot_class_distribution(df_processed[TARGET_RESCUE_COL], 
                                "Preprocessed Target (Rescue Event)", "rescue_target_dist_processed.png", exp_viz_dir)

        df_engineered = engineer_features_for_rescue(df_processed, DATE_COLUMN, 
                                                     VISITOR_COUNT_COL_FOR_FE, TARGET_RESCUE_COL) 

# main 블록
        X_train, X_val, X_test, y_train, y_val, y_test, dates_train_vis, dates_val_vis, dates_test_vis,\
        selected_features, scaler, imputer_means, cols_scaled = prepare_and_split_data_for_rescue_training(
            df_engineered, 
            TARGET_RESCUE_COL, 
            DATE_COLUMN,
            TRAIN_RATIO, 
            VALIDATION_RATIO, 
            TEST_RATIO, # test_ratio 인자 전달
            TOP_N_FEATURES_RESCUE,
            VISUALIZATION_DIR
        )
        
        X_train_final, y_train_final = X_train, y_train
        if condition.get("apply_smote", False):
            plot_class_distribution(y_train, "Train Target (Before SMOTE)", "rescue_train_dist_before_smote.png", exp_viz_dir)
            print(f"\nApplying SMOTE to training data. Original shape: {X_train.shape}")
            smote = SMOTE(random_state=RANDOM_SEED)
            X_train_final, y_train_final = smote.fit_resample(X_train, y_train) # SMOTE 적용된 데이터 사용
            print(f"SMOTE applied. Resampled shape: {X_train_final.shape}")
            plot_class_distribution(y_train_final, "Train Target (After SMOTE)", "rescue_train_dist_after_smote.png", exp_viz_dir)
        else:
            print("\nSMOTE not applied to training data.")

        # 모델 타입별 파라미터 분포 정의
        current_model_type = condition.get("model_type", "rf")
        param_dist_current = {}
        if current_model_type == "rf":
            param_dist_current = {
                'n_estimators': sp_randint(100, 500), 'max_depth': sp_randint(5, 21),
                'min_samples_split': sp_randint(2, 11), 'min_samples_leaf': sp_randint(1, 11),
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [condition.get("class_weight_rf")] # 실험 조건에서 가져오거나 기본값
            }
            if param_dist_current['class_weight'][0] is None and not condition.get("apply_smote", False):
                 param_dist_current['class_weight'] = ['balanced'] # SMOTE 안쓰면 balanced 기본으로
        # elif current_model_type == "lgbm":
            # param_dist_current = { ... } # LightGBM용 파라미터 분포

        rescue_model, best_params, val_metrics = train_and_tune_rescue_model(
            X_train_final, y_train_final, X_val, y_val, 
            model_type_str=current_model_type,
            param_dist=param_dist_current, 
            n_iter=condition.get("n_iter", 50), 
            scorer=condition.get("scorer", "recall"), 
            seed=RANDOM_SEED,
            class_weight_option=condition.get("class_weight_rf") # RF 전용 class_weight 전달
        )
        
        # evaluate_and_save_rescue_model 호출 시 인자 순서 및 개수 맞추기
        evaluate_and_save_rescue_model(
            rescue_model, X_test, y_test, scaler, imputer_means, cols_scaled,
            selected_features, best_params, 
            exp_run_dir, # 각 실험별 디렉토리
            INPUT_DATA_FILE,
            f"{TRAIN_RATIO*100:.0f}%/{VALIDATION_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}%",
            val_metrics, 
            CURRENT_TIME_STR, # 또는 exp_time_str
            model_name_prefix=condition['name'] # 실험 이름으로 모델 접두사 사용
        )
        
        # 실험 결과 요약에 추가 (저장된 패키지에서 테스트 결과 로드)
        saved_pkg_path = os.path.join(exp_run_dir, "rescue_model_package.pkl")
        if os.path.exists(saved_pkg_path):
            loaded_exp_pkg = joblib.load(saved_pkg_path)
            test_m = loaded_exp_pkg.get('test_metrics_adjusted_thresh', {})
            all_experiment_results_summary.append({
                "Experiment_Name": condition['name'],
                "Test_Accuracy": test_m.get('Accuracy', np.nan),
                "Test_Precision": test_m.get('Precision', np.nan),
                "Test_Recall": test_m.get('Recall', np.nan),
                "Test_F1_Score": test_m.get('F1-Score', np.nan),
                "Test_ROC_AUC": test_m.get('ROC_AUC', np.nan),
                "Optimal_Threshold": loaded_exp_pkg.get('optimal_threshold', np.nan),
                "Best_Params": best_params
            })

    # --- 모든 실험 결과 비교 요약 ---
    if all_experiment_results_summary:
        summary_df = pd.DataFrame(all_experiment_results_summary)
        print("\n\n========== All Rescue Model Experiment Results Summary ==========")
        print(summary_df.to_string())
        summary_file = os.path.join(BASE_RESULTS_DIR_MAIN, f"rescue_experiment_summary_{CURRENT_TIME_STR_MAIN}.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\nExperiment summary saved to: {summary_file}")

        # 주요 지표 비교 시각화 (예: Test F1-Score)
        if 'Test_F1_Score' in summary_df.columns:
            plt.figure(figsize=(12, 7))
            sns.barplot(x='Test_F1_Score', y='Experiment_Name', data=summary_df.sort_values('Test_F1_Score', ascending=False), orient='h')
            plt.title('Comparison of Test F1-Scores by Experiment (Rescue Model)')
            plt.xlabel('Test F1-Score (Adjusted Threshold)')
            plt.ylabel('Experiment Name')
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_RESULTS_DIR_MAIN, f"rescue_exp_f1_comparison_{CURRENT_TIME_STR_MAIN}.png")); plt.close()
            print("Experiment F1 comparison plot saved.")
            
    print("\n========== All Rescue Model Experiments Complete ==========")
    
    
# import os
# import ast
# import joblib
# import holidays
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# from datetime import datetime
# from imblearn.over_sampling import SMOTE
# from scipy.stats import randint as sp_randint
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
#     precision_recall_curve, confusion_matrix, roc_curve
# )


# # --- 0. Configuration ---
# INPUT_DATA_FILE = "data/preprocessed_data.csv"
# DATE_COLUMN = 'Date'
# VISITOR_COUNT_COL = 'Total_Visitor_Count'
# RESCUE_COUNT_COL = 'Total_Rescued_Count'
# TARGET_RESCUE_COL = 'Rescue_Event'

# TRAIN_RATIO = 0.6
# VALIDATION_RATIO = 0.2
# TEST_RATIO = 0.2
# assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9

# TOP_N_FEATURES_RESCUE = 30
# RANDOM_SEED = 42 # 일관성을 위한 랜덤 시드

# # --- Directory Setup ---
# CURRENT_TIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
# BASE_RESULTS_DIR = "./classification/rescue_model_training_results" # 디렉토리명 변경
# RUN_SPECIFIC_DIR = os.path.join(BASE_RESULTS_DIR, CURRENT_TIME_STR)
# VISUALIZATION_SAVE_DIR = os.path.join(RUN_SPECIFIC_DIR, "visualizations") # 시각화 저장용 디렉토리
# os.makedirs(RUN_SPECIFIC_DIR, exist_ok=True)
# os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)
# print(f"All results will be saved in: {RUN_SPECIFIC_DIR}")

# # --- 1. Data Handling Functions ---
# def load_data(file_path, date_col):
#     print(f"\n--- 1. Loading Data from {file_path} ---")
#     try:
#         df = pd.read_csv(file_path, parse_dates=[date_col])
#         df.sort_values(by=date_col, inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         print("Data loaded and sorted.")
#         return df
#     except FileNotFoundError: print(f"Error: File not found at {file_path}"); exit()
#     except Exception as e: print(f"Error loading data: {e}"); exit()

# def initial_preprocess_for_rescue(df, date_col, rescue_count_col, visitor_col, target_col_name):
#     print(f"\n--- 2. Initial Preprocessing for Rescue Model ---")
#     df_p = df.copy()
#     if date_col not in df_p.columns: print(f"Error: Date column '{date_col}' not found."); exit()
#     if not pd.api.types.is_datetime64_any_dtype(df_p[date_col]):
#         try: df_p[date_col] = pd.to_datetime(df_p[date_col]); print(f"'{date_col}' column converted to datetime.")
#         except Exception as e: print(f"Error converting '{date_col}' to datetime: {e}"); exit()
#     if rescue_count_col not in df_p.columns: print(f"Error: Rescue count column '{rescue_count_col}' not found."); exit()
#     df_p[rescue_count_col] = pd.to_numeric(df_p[rescue_count_col], errors='coerce').fillna(0).astype(int)
#     df_p[target_col_name] = (df_p[rescue_count_col] > 0).astype(int)
#     print(f"Target column '{target_col_name}' created.")
#     if visitor_col in df_p.columns: df_p[visitor_col] = pd.to_numeric(df_p[visitor_col], errors='coerce').fillna(0).astype(int)
#     list_like_cols_to_parse = ['Accident_Cause_List', 'Accident_Outcome_List']
#     for col_parse in list_like_cols_to_parse:
#         if col_parse in df_p.columns:
#             df_p[col_parse] = df_p[col_parse].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else ([] if pd.isna(x) else x))
#     print("Initial preprocessing for rescue model complete.")
#     return df_p

# # --- 2b. Visualization Helper Functions ---
# def plot_class_distribution(series, title, filename, save_dir):
#     plt.figure(figsize=(6, 4))
#     series.value_counts().plot(kind='bar')
#     plt.title(title)
#     plt.xlabel("Class")
#     plt.ylabel("Frequency")
#     plt.xticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, filename)); plt.close()
#     print(f"Class distribution plot saved: {filename}")

# def plot_feature_importance(model, feature_names, top_n, title, filename, save_dir):
#     if hasattr(model, 'feature_importances_'):
#         importances = pd.Series(model.feature_importances_, index=feature_names)
#         top_importances = importances.nlargest(top_n)
#         plt.figure(figsize=(10, max(6, top_n // 2))) # Dynamic height
#         top_importances.sort_values().plot(kind='barh')
#         plt.title(title)
#         plt.xlabel("Importance")
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, filename)); plt.close()
#         print(f"Feature importance plot saved: {filename}")
#     else:
#         print(f"Model {type(model).__name__} does not have feature_importances_ attribute.")

# def plot_confusion_matrix_heatmap(cm, classes, title, filename, save_dir, normalize=False, cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         fmt = '.2f'
#     else:
#         fmt = 'd'
    
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
#     plt.title(title)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, filename)); plt.close()
#     print(f"Confusion matrix heatmap saved: {filename}")


# # --- 3. Feature Engineering Function (for Rescue Model) ---
# def engineer_features_for_rescue(df_input, date_col, visitor_col_for_lag, rescue_event_col_for_lag):
#     print("\n--- 3. Engineering Features for Rescue Model ---")
#     df_eng = df_input.copy()
#     if date_col in df_eng.columns:
#         df_eng['is_weekend'] = (df_eng[date_col].dt.dayofweek >= 5).astype(int)
#         df_eng['month_num'] = df_eng[date_col].dt.month
#         df_eng = pd.get_dummies(df_eng, columns=['month_num'], prefix='month', drop_first=False)
        
#     if visitor_col_for_lag and visitor_col_for_lag in df_eng.columns:
#         for lag in [1, 7, 14, 30]: 
#             df_eng[f'{visitor_col_for_lag}_Lag{lag}'] = df_eng[visitor_col_for_lag].shift(lag).fillna(0)
            
#         for window in [7, 14, 30]: 
#             df_eng[f'{visitor_col_for_lag}_Roll{window}_Mean'] = df_eng[visitor_col_for_lag].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
    
#     else:
#         for lag in [1, 7, 14, 30]: 
#             df_eng[f'{VISITOR_COUNT_COL}_Lag{lag}'] = 0
#         for window in [7, 14, 30]: 
#             df_eng[f'{VISITOR_COUNT_COL}_Roll{window}_Mean'] = 0
            
#     if rescue_event_col_for_lag and rescue_event_col_for_lag in df_eng.columns:
#         df_eng['rescue_event_yesterday'] = df_eng[rescue_event_col_for_lag].shift(1).fillna(0)
        
#     else: 
#         df_eng['rescue_event_yesterday'] = 0
        
#     weather_rules = {'rain': ('Precipitation_mm(mm)', 3, 'consecutive_rain_3days'), 'freeze': ('MinTempC(℃)', 2, 'consecutive_freeze_2days')}
    
#     for condition, (col, window, new_col) in weather_rules.items():
#         if col in df_eng.columns:
#             is_cond = (df_eng[col] > 0 if condition == 'rain' else df_eng[col] < 0).astype(int)
#             df_eng[new_col] = (is_cond.rolling(window=window, min_periods=window).sum() == window).astype(int).fillna(0)
            
#     if rescue_event_col_for_lag and rescue_event_col_for_lag in df_eng.columns and date_col in df_eng.columns:
#         try:
#             df_eng['year_week_iso_temp'] = df_eng[date_col].dt.isocalendar().year.astype(str) + '-' + df_eng[date_col].dt.isocalendar().week.astype(str).str.zfill(2)
#             weekly_rescue_agg = df_eng.groupby('year_week_iso_temp')[rescue_event_col_for_lag].transform('max')
#             df_eng['last_week_date_temp'] = df_eng[date_col] - pd.to_timedelta(7, unit='D')
#             df_eng['last_year_week_iso_temp'] = df_eng['last_week_date_temp'].dt.isocalendar().year.astype(str) + '-' + df_eng['last_week_date_temp'].dt.isocalendar().week.astype(str).str.zfill(2)
#             week_map = pd.Series(weekly_rescue_agg.values, index=df_eng['year_week_iso_temp']).drop_duplicates().to_dict()
#             df_eng['rescue_in_last_week'] = df_eng['last_year_week_iso_temp'].map(week_map).fillna(0).astype(int)
#             df_eng.drop(columns=['year_week_iso_temp', 'last_week_date_temp', 'last_year_week_iso_temp'], inplace=True, errors='ignore')
        
#         except: 
#             df_eng['rescue_in_last_week'] = 0
#     else: 
#         df_eng['rescue_in_last_week'] = 0
#     time_cols = {'TimeOfMaxTempC': 'Hour_Of_Max_Temp', 'TimeOfMinTempC': 'Hour_Of_Min_Temp'}
    
#     for orig, new in time_cols.items():
#         if orig in df_eng.columns:
#             try: 
#                 df_eng[new] = pd.to_datetime(df_eng[orig], format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
                
#             except: 
#                 df_eng[new] = -1
#             df_eng.drop(columns=[orig], inplace=True, errors='ignore')
#     temp_c, hum_c = 'MaxTempC(℃)', 'Avg_Humidity_pct(%rh)'
    
#     if temp_c in df_eng.columns and hum_c in df_eng.columns:
#         df_eng['Temp_Humidity_Interaction'] = df_eng[temp_c].fillna(df_eng[temp_c].mean()) * df_eng[hum_c].fillna(df_eng[hum_c].mean())
        
#     if date_col in df_eng.columns and 'is_weekend' in df_eng.columns: # Ensure is_weekend exists
#         min_year_h, max_year_h = df_eng[date_col].dt.year.min(), df_eng[date_col].dt.year.max()
        
#         if pd.isna(min_year_h) or pd.isna(max_year_h): 
#             current_year_h = df_eng[date_col].dt.year.iloc[0]; kr_holidays_r = holidays.KR(years=current_year_h)
            
#         else: 
#             kr_holidays_r = holidays.KR(years=range(min_year_h, max_year_h + 1))
            
#         df_eng['is_official_holiday'] = df_eng[date_col].apply(lambda date: date in kr_holidays_r)
#         df_eng['is_day_off_official'] = (df_eng['is_weekend'] | df_eng['is_official_holiday']).astype(int)
#         long_holiday_threshold = 3
        
#         df_eng['is_base_long_holiday'] = ((df_eng['is_day_off_official'] == 1) & (df_eng.groupby((df_eng['is_day_off_official'].diff(1) != 0).astype(int).cumsum())['is_day_off_official'].transform('sum') >= long_holiday_threshold)).astype(int)
#         df_eng['prev_day_is_off'] = df_eng['is_day_off_official'].shift(1).fillna(0).astype(int)
#         df_eng['next_day_is_off'] = df_eng['is_day_off_official'].shift(-1).fillna(0).astype(int)
        
#         df_eng['is_bridge_day_candidate'] = ((df_eng['is_day_off_official'] == 0) & (df_eng['prev_day_is_off'] == 1) & (df_eng['next_day_is_off'] == 1)).astype(int)
#         df_eng['is_extended_holiday'] = df_eng['is_base_long_holiday']
        
#         df_eng.loc[df_eng['is_bridge_day_candidate'].values == 1, 'is_extended_holiday'] = 1 # Use .values for boolean indexing
#         df_eng.loc[df_eng['is_bridge_day_candidate'].shift(-1).fillna(False).values, 'is_extended_holiday'] = 1
#         df_eng.loc[df_eng['is_bridge_day_candidate'].shift(1).fillna(False).values, 'is_extended_holiday'] = 1
        
#         df_eng['final_holiday_group'] = (df_eng['is_extended_holiday'].diff(1) != 0).astype(int).cumsum()
#         df_eng['consecutive_extended_days_off'] = df_eng.groupby('final_holiday_group')['is_extended_holiday'].transform('sum')
#         df_eng['is_final_long_holiday_rescue'] = ((df_eng['is_extended_holiday'] == 1) & (df_eng['consecutive_extended_days_off'] >= long_holiday_threshold)).astype(int)
        
#         df_eng.drop(columns=['is_official_holiday', 'is_day_off_official', 'day_off_group', 'consecutive_official_days_off', 'is_base_long_holiday', 'prev_day_is_off', 'next_day_is_off', 'is_bridge_day_candidate', 'is_extended_holiday', 'final_holiday_group', 'consecutive_extended_days_off'], inplace=True, errors='ignore')
    
#     print("Feature engineering for rescue model complete.")
#     return df_eng

# # --- 4. Data Splitting, Feature Selection & Scaling (for Training Rescue Model) ---
# # prepare_and_split_data_for_rescue_training 함수 수정
# def prepare_and_split_data_for_rescue_training(df_engineered, target_col, date_col, 
#                                                train_r, val_r, top_n):
#     print("\n--- 4. Preparing and Splitting Data for Rescue Model Training ---")
    
#     explicit_exclude_train = [date_col, target_col, RESCUE_COUNT_COL, VISITOR_COUNT_COL, 
#                               'Accident_Cause_List', 'Accident_Outcome_List']
#     potential_features_train = [col for col in df_engineered.columns if col not in explicit_exclude_train]
    
#     if not potential_features_train: # potential_features_train이 비어있는지 확인
#         print("오류: X를 구성할 피처가 없습니다. explicit_exclude_train 목록을 확인하세요."); exit()

#     X_full = df_engineered[potential_features_train].copy()
#     y_full = df_engineered[target_col].copy()
#     dates_full = df_engineered[date_col].copy() # dates_full 정의

#     # 데이터 분할
#     n_total = len(X_full)
#     n_train = int(n_total * train_r); n_val = int(n_total * val_r)
    
#     X_train_raw, y_train, dates_train = X_full.iloc[:n_train], y_full.iloc[:n_train], dates_full.iloc[:n_train] # <--- dates_train 할당
#     X_val_raw, y_val, dates_val = X_full.iloc[n_train:n_train+n_val], y_full.iloc[n_train:n_train+n_val], dates_full.iloc[n_train:n_train+n_val]
#     X_test_raw, y_test, dates_test = X_full.iloc[n_train+n_val:], y_full.iloc[n_train+n_val:], dates_full.iloc[n_train+n_val:]
#     print(f"Data split: Train {len(X_train_raw)}, Validation {len(X_val_raw)}, Test {len(X_test_raw)}")
    
#     X_train_for_imp = X_train_raw.copy(); imputer_means_train = {}
    
#     for col in X_train_for_imp.columns:
#         if X_train_for_imp[col].isnull().any():
#             if pd.api.types.is_numeric_dtype(X_train_for_imp[col]): 
#                 mean_val = X_train_for_imp[col].mean(); X_train_for_imp[col].fillna(mean_val, inplace=True)
#                 imputer_means_train[col] = mean_val
                
#             else:
#                 mode_val = X_train_for_imp[col].mode()[0] if not X_train_for_imp[col].mode().empty else "Unknown" 
#                 X_train_for_imp[col].fillna(mode_val, inplace=True); imputer_means_train[col] = mode_val
    
#     rf_imp = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1) # Classifier for importance
#     X_numeric_for_imp = X_train_for_imp.select_dtypes(include=np.number)
    
#     if X_numeric_for_imp.empty: 
#         print("Error: No numeric features for importance calculation."); exit()
    
#     rf_imp.fit(X_numeric_for_imp, y_train)
    
#     importances = pd.DataFrame({'Feature': X_numeric_for_imp.columns, 'Importance': rf_imp.feature_importances_})
#     importances = importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
#     selected_features = importances.head(min(top_n, len(importances)))['Feature'].tolist()
    
#     print(f"Top {len(selected_features)} features selected: {selected_features}")
#     plot_feature_importance(rf_imp, X_numeric_for_imp.columns, top_n, "Feature Importances (Rescue Model Training)", "rescue_feature_importances.png", VISUALIZATION_SAVE_DIR) # 피처 중요도 시각화
    
#     X_train_sel = X_train_raw[selected_features].copy(); X_val_sel = X_val_raw[selected_features].copy(); X_test_sel = X_test_raw[selected_features].copy()
    
#     imputer_means_selected_train = {k: v for k, v in imputer_means_train.items() if k in selected_features}
#     X_train_filled = X_train_sel.fillna(imputer_means_selected_train)
#     X_val_filled = X_val_sel.fillna(imputer_means_selected_train)
#     X_test_filled = X_test_sel.fillna(imputer_means_selected_train)
#     scaler = MinMaxScaler(); 
    
#     binary_like_cols_rescue = [col for col in selected_features if X_train_filled[col].nunique(dropna=False) <= 2]
#     one_hot_prefixes_rescue = ('month_num_', 'day_of_week_') # 실제 사용된 prefix로 수정
#     cols_to_scale = [col for col in selected_features if col not in binary_like_cols_rescue and not any(col.startswith(p) for p in one_hot_prefixes_rescue) and pd.api.types.is_numeric_dtype(X_train_filled[col])]
    
#     if cols_to_scale:
#         X_train_filled[cols_to_scale] = scaler.fit_transform(X_train_filled[cols_to_scale])
#         X_val_filled[cols_to_scale] = scaler.transform(X_val_filled[cols_to_scale])
#         X_test_filled[cols_to_scale] = scaler.transform(X_test_filled[cols_to_scale])
#         print(f"Features scaled: {cols_to_scale}")
#     else: scaler = None; cols_to_scale = []
#     return X_train_filled, X_val_filled, X_test_filled, y_train, y_val, y_test, \
#            dates_train, dates_val, dates_test, \
#            selected_features, scaler, imputer_means_selected_train, cols_to_scale

# # --- 5. Model Training and Tuning (for Rescue Model) ---
# def train_and_tune_rescue_model(X_train, y_train, X_val, y_val, param_dist, n_iter, scorer, seed):
#     # ... (이전과 동일, RandomForestClassifier 사용) ...
#     print(f"\n--- 5. Training and Tuning Rescue Model (RandomForestClassifier) ---")
#     estimator = RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight='balanced')
#     cv_strategy = 5 
#     random_search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring=scorer, cv=cv_strategy, random_state=seed, n_jobs=-1, verbose=1)
#     random_search.fit(X_train, y_train)
#     print(f"Best parameters: {random_search.best_params_}")
#     scorer_name = scorer if isinstance(scorer, str) else scorer._score_func.__name__ # make_scorer 객체 처리
#     print(f"Best CV Score ({scorer_name}): {random_search.best_score_:.4f}")
#     tuned_model = random_search.best_estimator_
#     preds_val = tuned_model.predict(X_val); probs_val = tuned_model.predict_proba(X_val)[:, 1]
#     val_metrics = {'Accuracy': accuracy_score(y_val, preds_val), 'Precision': precision_score(y_val, preds_val, zero_division=0),
#                    'Recall': recall_score(y_val, preds_val, zero_division=0), 'F1-Score': f1_score(y_val, preds_val, zero_division=0),
#                    'ROC_AUC': roc_auc_score(y_val, probs_val)}
#     print(f"Validation metrics (default threshold): {val_metrics}")
#     cm_val = confusion_matrix(y_val, preds_val)
#     plot_confusion_matrix_heatmap(cm_val, classes=[0,1], title='Validation Confusion Matrix', filename='rescue_val_cm_heatmap.png', save_dir=VISUALIZATION_SAVE_DIR)
#     return tuned_model, random_search.best_params_, val_metrics

# # --- 6. Model Evaluation and Saving (for Rescue Model) ---
# def evaluate_and_save_rescue_model(model, X_test, y_test, scaler_obj, imputer_means_obj, 
#                                    cols_scaled_list, selected_features_list, best_params, 
#                                    save_dir, input_file, ratios_str, val_metrics, timestamp):
#     # ... (이전과 동일, Confusion Matrix 시각화 추가) ...
#     print(f"\n--- 6. Evaluating and Saving Rescue Model ---")
#     preds_test = model.predict(X_test); probs_test = model.predict_proba(X_test)[:, 1]
#     roc_auc_test = roc_auc_score(y_test, probs_test)
#     print(f"Test ROC AUC: {roc_auc_test:.4f}")
#     fpr, tpr, _ = roc_curve(y_test, probs_test)
#     plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'ROC Curve (Test AUC = {roc_auc_test:.4f})'); plt.plot([0,1],[0,1], linestyle='--', label='Random Guess'); plt.title('ROC Curve (Test Set)'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(); plt.savefig(os.path.join(save_dir, 'rescue_roc_curve.png')); plt.close()
#     prec, rec, thresh = precision_recall_curve(y_test, probs_test)
#     plt.figure(figsize=(8,6)); plt.plot(rec[:-1], prec[:-1], marker='.', label='Precision-Recall Curve'); plt.title('PR Curve (Test Set)'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid(); plt.savefig(os.path.join(save_dir, 'rescue_pr_curve.png')); plt.close()
#     f1s = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9); opt_thresh = thresh[np.argmax(f1s)] if len(f1s) > 0 else 0.5
#     print(f"Optimal Threshold (F1 max): {opt_thresh:.4f}")
#     preds_adj = (probs_test >= opt_thresh).astype(int)
#     final_metrics = {'Accuracy': accuracy_score(y_test, preds_adj), 'Precision': precision_score(y_test, preds_adj, zero_division=0), 
#                      'Recall': recall_score(y_test, preds_adj, zero_division=0), 'F1-Score': f1_score(y_test, preds_adj, zero_division=0),
#                      'ROC_AUC': roc_auc_test}
#     print(f"Test Performance (Adjusted Threshold {opt_thresh:.4f}): {final_metrics}")
#     cm_test = confusion_matrix(y_test, preds_adj)
#     plot_confusion_matrix_heatmap(cm_test, classes=[0,1], title='Test Confusion Matrix (Adjusted Threshold)', filename='rescue_test_cm_heatmap.png', save_dir=save_dir)
#     log_path = os.path.join(save_dir, 'rescue_model_evaluation_log.txt') # ... (로그 내용 이전과 동일) ...
#     with open(log_path, 'w', encoding='utf-8') as f:
#         f.write(f"Timestamp: {timestamp}\nInput File: {input_file}\nSplit Ratios: {ratios_str}\nBest Hyperparameters: {best_params}\nValidation Metrics: {val_metrics}\nTest Performance (Adj. Threshold): {final_metrics}\nOptimal Threshold: {opt_thresh:.4f}\nSelected Features ({len(selected_features_list)}):\n" + "\n".join(selected_features_list))
#     model_pkg_path = os.path.join(save_dir, "rescue_model_package.pkl")
#     model_package = {'model': model, 'scaler': scaler_obj, 'imputer_means': imputer_means_obj, 'cols_scaled_at_fit': cols_scaled_list, 
#                      'features': selected_features_list, 'best_hyperparameters': best_params, 'optimal_threshold': opt_thresh,
#                      'model_type': type(model).__name__, 'training_timestamp': timestamp, 'test_metrics_adjusted_thresh': final_metrics}
#     joblib.dump(model_package, model_pkg_path)
#     print(f"Rescue model package saved to: {model_pkg_path}")

# # --- Main Execution ---
# if __name__ == '__main__':
#     df_raw = load_data(INPUT_DATA_FILE, DATE_COLUMN)
    
#     # 타겟 변수 분포 시각화 (원본)
#     plot_class_distribution(df_raw[RESCUE_COUNT_COL].apply(lambda x: 1 if x > 0 else 0), 
#                             "Original Target Distribution (Rescue Event)", 
#                             "rescue_target_dist_raw.png", VISUALIZATION_SAVE_DIR)

#     df_processed = initial_preprocess_for_rescue(df_raw, DATE_COLUMN, RESCUE_COUNT_COL, 
#                                                  VISITOR_COUNT_COL, TARGET_RESCUE_COL)
    
#     # 전처리 후 타겟 변수 분포 시각화
#     plot_class_distribution(df_processed[TARGET_RESCUE_COL], 
#                             "Preprocessed Target Distribution (Rescue Event)", 
#                             "rescue_target_dist_processed.png", VISUALIZATION_SAVE_DIR)

#     df_engineered = engineer_features_for_rescue(df_processed, DATE_COLUMN, 
#                                                  VISITOR_COUNT_COL, 
#                                                  TARGET_RESCUE_COL) 

#     X_train, X_val, X_test, y_train, y_val, y_test, \
#     _, _, _, \
#     selected_features, scaler, imputer_means, cols_scaled = prepare_and_split_data_for_rescue_training(
#         df_engineered, TARGET_RESCUE_COL, DATE_COLUMN,
#         TRAIN_RATIO, VALIDATION_RATIO, TOP_N_FEATURES_RESCUE
#     )
    
#     # SMOTE 적용 전 학습 데이터 클래스 분포
#     plot_class_distribution(y_train, "Train Target Distribution (Before SMOTE)", 
#                             "rescue_train_dist_before_smote.png", VISUALIZATION_SAVE_DIR)
    
#     print(f"\nApplying SMOTE to training data. Original shape: {X_train.shape}")
#     smote = SMOTE(random_state=RANDOM_SEED) # 전역 RANDOM_SEED 사용
#     X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
#     print(f"SMOTE applied. Resampled shape: {X_train_sm.shape}")
#     print(f"Class distribution after SMOTE: {pd.Series(y_train_sm).value_counts(normalize=True)}")
    
#     # SMOTE 적용 후 학습 데이터 클래스 분포
#     plot_class_distribution(y_train_sm, "Train Target Distribution (After SMOTE)", 
#                             "rescue_train_dist_after_smote.png", VISUALIZATION_SAVE_DIR)

#     rf_param_dist = {
#         'n_estimators': sp_randint(100, 500), 'max_depth': sp_randint(5, 21),
#         'min_samples_split': sp_randint(2, 11), 'min_samples_leaf': sp_randint(1, 11),
#         'max_features': ['sqrt', 'log2', None], # None은 모든 피처 사용
#         'class_weight': ['balanced', 'balanced_subsample', None] 
#     }
#     N_ITER_RESCUE_SEARCH = 50 
#     SCORER_RESCUE = 'recall' 

#     rescue_model, best_params, val_metrics = train_and_tune_rescue_model(
#         X_train_sm, y_train_sm, X_val, y_val, 
#         rf_param_dist, N_ITER_RESCUE_SEARCH, SCORER_RESCUE, RANDOM_SEED
#     )
    
#     evaluate_and_save_rescue_model(
#         rescue_model, X_test, y_test, scaler, imputer_means, cols_scaled,
#         selected_features, best_params, RUN_SPECIFIC_DIR, INPUT_DATA_FILE,
#         f"{TRAIN_RATIO*100}%/{VALIDATION_RATIO*100}%/{TEST_RATIO*100}%",
#         val_metrics, CURRENT_TIME_STR
#     )
    
#     print("\n========== Rescue Model Training Pipeline Complete ==========")