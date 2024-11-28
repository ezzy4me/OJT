import os
import pandas as pd

def load_ec2_data(file_path):
    """
    Load and process EC2 data from a CSV file.
    """
    folder_path = os.getcwd() + '/metrics_data'
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 각각의 .csv 파일을 읽어서 DataFrame으로 변환
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes[file] = df

    # dataframes 정렬
    for file_name, df in dataframes.items():
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        dataframes[file_name] = df
    
    # merge dataframes
    # it needs to be modified if there are multiple csv files
    print(dataframes.keys())
    combined_data = pd.merge(
        dataframes['CPUUtilization_AWSEC2_ec2_metrics.csv'][['Value']],
        dataframes['mem_used_percent_CWAgent_ec2_metrics.csv'][['Value']],
        left_index=True,
        right_index=True,
        how='outer',
        suffixes=('_CPU', '_Memory')
    )

    return combined_data

def generate_statistics(data):
    """
    Generate descriptive statistics for a DataFrame.
    """
    summary_dct = {}
    for infra, _ in data.items():
        stats = data[infra].describe()  # 주요 통계 정보 생성
        summary_dct[f"{infra}에 대한 통계 정보:"] = (
            f"평균: {stats['mean']:.3f}%, "
            f"최대값: {stats['max']:.3f}%, "
            f"최소값: {stats['min']:.3f}%, "
            f"표준편차: {stats['std']:.3f}, "
            f"1사분위: {stats['25%']:.3f}%, "
            f"중앙값: {stats['50%']:.3f}%",
            f"3사분위: {stats['75%']:.3f}%"
            )
        
    return summary_dct