import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# --- 0. Configuration ---
# Input File Paths
ACCIDENT_RAW_FILE = "./accident/전국 산악사고 구조활동현황(2017~2021).csv"
WEATHER_DATA_DIR = './weather/'
VISITOR_RAW_FILE = "./accident/국립공원공단_국립공원 시간별 일별 탐방객 통계_20221024.csv"

# Output File
FINAL_OUTPUT_MERGED_FILE = "preprocessed_merged_data.csv" # 최종 출력 파일명

# Common Settings
COMMON_DATE_COLUMN = 'Date' # 최종적으로 사용할 날짜 컬럼명 (영문)
START_DATE_STR = "2018-01-01"
END_DATE_STR = "2022-09-30"

# Specific Settings
TARGET_DISTRICT_FOR_VISITORS = "설악동"
TARGET_WEATHER_STATION_NAME = '속초'
ACCIDENT_CITY_FILTER = '속초시'

# --- Directory Setup for EDA ---
CURRENT_TIME_STR_EDA = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_RESULTS_DIR_EDA = "./data_preprocessing_pipeline_outputs"
RUN_SPECIFIC_DIR_EDA = os.path.join(BASE_RESULTS_DIR_EDA, CURRENT_TIME_STR_EDA)
os.makedirs(RUN_SPECIFIC_DIR_EDA, exist_ok=True)
print(f"EDA and intermediate results will be saved in: {RUN_SPECIFIC_DIR_EDA}")

# --- Matplotlib Font Setup ---
if platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False

# --- Column Mapping Dictionaries ---
# Weather: Standard Korean -> Final English
weather_col_map_std_korean_to_english = {
    '일시': COMMON_DATE_COLUMN, '최대풍속(m/s)': 'MaxWindSpd(m/s)', '평균풍속(m/s)': 'AvgWindSpd(m/s)',
    '최고기온(℃)': 'MaxTempC(℃)', '최저기온(℃)': 'MinTempC(℃)', '평균기온(℃)': 'AvgTempC(℃)',
    '최고기온시각': 'TimeOfMaxTempC', '최저기온시각': 'TimeOfMinTempC', '일교차': 'Diurnal_Temperature_Range',
    '평균습도(%rh)': 'Avg_Humidity_pct(%rh)', '강수량(mm)': 'Precipitation_mm(mm)',
    '지점번호': 'Station_ID', '지점명': 'Station_Name'
}
# Accident: Original Korean -> Final English
accident_col_map_korean_to_english = {
    '신고년월일': COMMON_DATE_COLUMN, '발생장소_구': 'Accident_City_District',
    '구조인원': 'Total_Rescued_Count', '사고원인코드명_사고종별': 'Accident_Cause_Raw',
    '처리결과코드': 'Outcome_Code_Raw'
}
# Visitor: Original Korean -> Final English
visitor_col_map_korean_to_english = {
    '일자': COMMON_DATE_COLUMN, '관리지구': 'District_Name',
    '전체 탐방객수': 'Total_Visitor_Count' # 최종적으로 사용할 탐방객 수 컬럼명
}

# --- Helper Functions ---
def try_read_csv_with_encodings(file_path):
    for enc in ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip().str.replace('\t', '', regex=False)
            return df
        except UnicodeDecodeError: continue
        except Exception as e: print(f"Error reading {file_path} with {enc}: {e}"); continue
    print(f"Warning: Could not read or decode {file_path}. Skipping this file.")
    return None

def normalize_and_rename_cols(df, col_map_to_english):
    df.columns = [col.strip().replace('\t', '') for col in df.columns]

    if col_map_to_english: # 2. Rename to English
        rename_map = {k: v for k, v in col_map_to_english.items() if k in df.columns}
        if rename_map: df.rename(columns=rename_map, inplace=True)
    return df


def process_date_column(df, current_date_col, final_date_col):
    if current_date_col not in df.columns:
        print(f"Warning: Date column '{current_date_col}' not found. Skipping date processing."); return df
    df[current_date_col] = df[current_date_col].astype(str).str.replace('.', '-', regex=False).str.strip()
    df[final_date_col] = pd.to_datetime(df[current_date_col], errors='coerce')
    if current_date_col != final_date_col:
        df.drop(columns=[current_date_col], inplace=True, errors='ignore')
    df.dropna(subset=[final_date_col], inplace=True)
    return df

def explore_dataframe(df, df_name, save_dir):
    if df is None or df.empty: print(f"\n--- Exploring {df_name} ---\nDataFrame is empty. Skipping."); return
    print(f"\n--- Exploring {df_name} (Shape: {df.shape}) ---")
    print("Head:\n", df.head())
    print("\nMissing Values:\n", df.isnull().sum().sort_values(ascending=False)) # 결측치 많은 순으로 정렬
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty: print("\nDescriptive Statistics (Numeric):\n", numeric_df.describe())
    
    if not numeric_df.empty:
        numeric_sample = numeric_df.sample(min(1000, len(numeric_df))) # 샘플링 유지
        num_cols = len(numeric_sample.columns)
        if num_cols > 0:
            n_plots_per_row = 5 # 한 줄에 표시할 그래프 수
            n_rows = (num_cols + n_plots_per_row - 1) // n_plots_per_row
            
            # 전체 그림(figure) 크기를 컬럼 수에 따라 좀 더 유동적으로 조절
            fig_width = 15 
            fig_height = max(5, n_rows * 3.5) # 행당 높이 조절
            
            fig, axes = plt.subplots(n_rows, n_plots_per_row, figsize=(fig_width, fig_height))
            axes = axes.flatten() # 다차원 배열을 1차원으로 만들어 순회 용이하게 함

            for i, col in enumerate(numeric_sample.columns):
                if i < len(axes): # 생성된 subplot 개수만큼만 그림
                    sns.histplot(numeric_sample[col].dropna(), kde=True, bins=30, ax=axes[i])
                    # --- 제목 글씨 크기 조절 ---
                    axes[i].set_title(f'Distribution of {col}', fontsize=10) # 예: 폰트 크기를 10으로 설정
                    axes[i].set_xlabel(col, fontsize=9) # x축 레이블 크기도 조절 (선택적)
                    axes[i].set_ylabel('Count', fontsize=9) # y축 레이블 크기도 조절 (선택적)
                else:
                    break # 더 이상 그릴 subplot이 없으면 중단
            
            # 남는 빈 subplot 숨기기
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(pad=2.0) # subplot 간 간격 및 전체 레이아웃 조정, pad 값으로 여백 조절
            plt.savefig(os.path.join(save_dir, f"{df_name}_numeric_distributions.png")); plt.close(fig) # fig 명시적 닫기
            print(f"Numeric distributions plot saved for {df_name}.")

# --- Data Processing Functions ---
def process_weather_data(weather_dir, k_to_e_map, date_col_final, station_name, eda_save_dir):
    print("\n--- 1. Processing Weather Data ---")
    csv_files = glob.glob(os.path.join(weather_dir, '*.csv'))
    csv_files = [f for f in csv_files if not (f.endswith('combined_weather_data.csv') or FINAL_OUTPUT_MERGED_FILE in f)]
    if not csv_files: print("No weather CSV files found."); return pd.DataFrame(columns=[date_col_final])

    df_weather_merged = pd.DataFrame(pd.date_range(start=START_DATE_STR, end=END_DATE_STR, freq='D'), columns=[date_col_final])

    for idx, file in enumerate(csv_files):
        print(f"  Processing weather file: {file}")
        df_w_raw = try_read_csv_with_encodings(file)
        if df_w_raw is None: continue
        
        df_w_english = normalize_and_rename_cols(df_w_raw, col_map_to_english=k_to_e_map) # 영문명으로 변경

        if not df_w_english.empty and date_col_final in df_w_english.columns:
            df_w_english = df_w_english.groupby(date_col_final, as_index=False).first()
            explore_dataframe(df_w_english, f"weather_processed_{os.path.basename(file)}_{idx}", eda_save_dir)
            
            cols_to_merge = [col for col in df_w_english.columns if col == date_col_final or col not in df_weather_merged.columns]
            if len(cols_to_merge) > 1:
                if not pd.api.types.is_datetime64_any_dtype(df_weather_merged[date_col_final]):
                    df_weather_merged[date_col_final] = pd.to_datetime(df_weather_merged[date_col_final], errors='coerce')
                    df_weather_merged.dropna(subset=[date_col_final], inplace=True)
                    
                # df_w_english의 Date 컬럼도 확실히 datetime64[ns]로 만듦
                if not pd.api.types.is_datetime64_any_dtype(df_w_english[date_col_final]):
                    df_w_english[date_col_final] = pd.to_datetime(df_w_english[date_col_final], errors='coerce')
                    df_w_english.dropna(subset=[date_col_final], inplace=True)
                df_weather_merged = pd.merge(df_weather_merged, df_w_english[cols_to_merge], on=date_col_final, how='left')
    
    if df_weather_merged.empty or len(df_weather_merged.columns) <= 1:
        print("Warning: No valid weather data could be merged."); return pd.DataFrame(columns=[date_col_final])
    print(f"Weather data processed and merged. Shape: {df_weather_merged.shape}")
    return df_weather_merged

def process_accident_data(file_path, k_to_e_map, date_col_final, city_filter, start_str, end_str, eda_save_dir):
    print("\n--- 2. Processing Accident Data ---")
    df_raw = try_read_csv_with_encodings(file_path)
    if df_raw is None: return pd.DataFrame(columns=[date_col_final])
    
    df_processed = normalize_and_rename_cols(df_raw, col_map_to_english=k_to_e_map)
    df_processed = process_date_column(df_processed, date_col_final, date_col_final) # 이미 영문명 'Date'

    city_col_actual = 'Accident_City_District' # 영문명 사용
    if city_col_actual in df_processed.columns and city_filter:
        df_filtered_city = df_processed[df_processed[city_col_actual] == city_filter].copy()
    else: df_filtered_city = df_processed.copy()

    if not df_filtered_city.empty and date_col_final in df_filtered_city.columns:
        start_dt, end_dt = pd.to_datetime(start_str), pd.to_datetime(end_str)
        df_filtered_date = df_filtered_city[(df_filtered_city[date_col_final] >= start_dt) & (df_filtered_city[date_col_final] <= end_dt)]
        if df_filtered_date.empty: print(f"No accident data for '{city_filter}' in date range."); return pd.DataFrame(columns=[date_col_final, 'Total_Rescued_Count', 'Accident_Cause_List', 'Accident_Outcome_List'])

        def list_agg(s): return [str(i) for i in s if pd.notna(i) and str(i).strip() != '']
        agg_rules = {'Total_Rescued_Count': ('Total_Rescued_Count', 'sum')}
        if 'Accident_Cause_Raw' in df_filtered_date.columns: agg_rules['Accident_Cause_List'] = ('Accident_Cause_Raw', list_agg)
        if 'Outcome_Code_Raw' in df_filtered_date.columns: agg_rules['Accident_Outcome_List'] = ('Outcome_Code_Raw', list_agg)
        df_summary = df_filtered_date.groupby(date_col_final, as_index=False).agg(**agg_rules)
        explore_dataframe(df_summary, f"accident_summary_{city_filter.lower()}", eda_save_dir)
        print(f"{city_filter} accident data summarized. Shape: {df_summary.shape}")
        return df_summary
    print(f"No accident data for '{city_filter}' to summarize."); return pd.DataFrame(columns=[date_col_final, 'Total_Rescued_Count', 'Accident_Cause_List', 'Accident_Outcome_List'])

def process_visitor_data(file_path, k_to_e_map, date_col_final, district_filter, start_str, end_str, eda_save_dir):
    print("\n--- 3. Processing Visitor Data ---")
    df_raw = try_read_csv_with_encodings(file_path)
    if df_raw is None: return pd.DataFrame(columns=[date_col_final])

    df_processed = normalize_and_rename_cols(df_raw, col_map_to_english=k_to_e_map)
    df_processed = process_date_column(df_processed, date_col_final, date_col_final)
    
    start_dt, end_dt = pd.to_datetime(start_str), pd.to_datetime(end_str)
    df_filtered_date = df_processed[(df_processed[date_col_final] >= start_dt) & (df_processed[date_col_final] <= end_dt)]
    if df_filtered_date.empty: print("No visitor data in date range."); return pd.DataFrame(columns=[date_col_final, 'Total_Visitor_Count'])

    df_district_daily = pd.DataFrame(columns=[date_col_final, 'Total_Visitor_Count']) # 최종 컬럼명 사용
    if 'District_Name' in df_filtered_date.columns and district_filter:
        df_district_raw = df_filtered_date[df_filtered_date['District_Name'] == district_filter].copy()
        if not df_district_raw.empty:
            # 'Total_Visitor_Count' 컬럼이 visitor_col_map_korean_to_english에 의해 이미 생성됨
            df_district_daily = df_district_raw.groupby(date_col_final, as_index=False)['Total_Visitor_Count'].sum()
    explore_dataframe(df_district_daily, f"visitor_{district_filter.lower()}_daily", eda_save_dir)
    print(f"{district_filter} visitor data processed. Shape: {df_district_daily.shape}")
    return df_district_daily

# --- Main Data Processing Pipeline ---
if __name__ == '__main__':
    df_weather = process_weather_data(WEATHER_DATA_DIR, weather_col_map_std_korean_to_english, COMMON_DATE_COLUMN, 
                                      TARGET_WEATHER_STATION_NAME, RUN_SPECIFIC_DIR_EDA)
    
    df_accident = process_accident_data(ACCIDENT_RAW_FILE, accident_col_map_korean_to_english, 
                                        COMMON_DATE_COLUMN, ACCIDENT_CITY_FILTER, START_DATE_STR, END_DATE_STR, RUN_SPECIFIC_DIR_EDA)
    
    df_seorakdong_visitor = process_visitor_data(
        VISITOR_RAW_FILE, visitor_col_map_korean_to_english, COMMON_DATE_COLUMN, 
        TARGET_DISTRICT_FOR_VISITORS, START_DATE_STR, END_DATE_STR, RUN_SPECIFIC_DIR_EDA
    )

    print("\n--- 4. Merging All Processed Data ---")
    try:
        start_dt_m = pd.to_datetime(START_DATE_STR); end_dt_m = pd.to_datetime(END_DATE_STR)
        final_df = pd.DataFrame(pd.date_range(start=start_dt_m, end=end_dt_m, freq='D'), columns=[COMMON_DATE_COLUMN])
    except Exception as e: print(f"Error creating date range for final merge: {e}"); exit()

    dataframes_to_merge = [df_accident, df_weather, df_seorakdong_visitor] 
    df_names_for_log = ["AccidentSummary", "WeatherFinal", "SeorakdongVisitor"]
    
    for i, df_component in enumerate(dataframes_to_merge):
        df_name = df_names_for_log[i]
        if df_component is not None and not df_component.empty and COMMON_DATE_COLUMN in df_component.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_component[COMMON_DATE_COLUMN]):
                df_component[COMMON_DATE_COLUMN] = pd.to_datetime(df_component[COMMON_DATE_COLUMN], errors='coerce')
            final_df = pd.merge(final_df, df_component, on=COMMON_DATE_COLUMN, how='left')
            print(f"  Merged {df_name} data. Current shape: {final_df.shape}")
        else:
            print(f"  Skipping merge for {df_name} data (None, empty, or no date column).")

    print("\n--- 5. Applying Final Touches and EDA on Merged Data ---")
    for col in final_df.columns:
        if col == COMMON_DATE_COLUMN: continue
        # 결측치 처리 전에 실제 타입 확인
        # 첫 번째 유효한 값의 타입을 기준으로 리스트인지, 숫자인지, 그 외인지 판단
        first_valid_idx = final_df[col].first_valid_index()
        if first_valid_idx is not None:
            first_valid_type = type(final_df[col].loc[first_valid_idx])
            if pd.api.types.is_numeric_dtype(final_df[col].dtype) or np.issubdtype(first_valid_type, np.number):
                final_df[col].fillna(0, inplace=True)
            elif first_valid_type == list:
                final_df[col] = final_df[col].apply(lambda x: x if isinstance(x, list) else [])
            else: # Default to empty string for other types or all NaN columns
                final_df[col].fillna("", inplace=True)
        else: # 컬럼 전체가 NaN인 경우
            if pd.api.types.is_numeric_dtype(final_df[col].dtype): final_df[col].fillna(0, inplace=True)
            else: final_df[col].fillna("", inplace=True) # 또는 [] 등 기본값 설정
    
    explore_dataframe(final_df, "final_merged_data_before_save", RUN_SPECIFIC_DIR_EDA)

    numeric_cols_final = final_df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols_final) > 1:
        plt.figure(figsize=(max(15, len(numeric_cols_final)), max(12, len(numeric_cols_final)-2))) # 크기 조정
        sns.heatmap(final_df[numeric_cols_final].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8}) # annot 글자 크기
        plt.title('Correlation Matrix of Final Merged Data', fontsize=15)
        plt.xticks(fontsize=10); plt.yticks(fontsize=10) # 눈금 글자 크기
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_SPECIFIC_DIR_EDA, "final_merged_data_correlation_heatmap.png")); plt.close()
        print("Final merged data correlation heatmap saved.")

    final_df[COMMON_DATE_COLUMN] = final_df[COMMON_DATE_COLUMN].dt.strftime('%Y-%m-%d')

    final_df.to_csv(FINAL_OUTPUT_MERGED_FILE, index=False, encoding='utf-8-sig')
    print(f"\n--- Final Merged Data Saved to {FINAL_OUTPUT_MERGED_FILE} ---")
    print(f"Final DataFrame shape: {final_df.shape}")
    print("Final columns:", final_df.columns.tolist())
    print("Sample of final data:\n", final_df.head())

    print("\n========== All Data Processing and Merging Complete ==========")