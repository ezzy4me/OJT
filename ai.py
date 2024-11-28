import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
import os
import argparse

# OpenAI API 키 환경 변수에서 로드
openai_key = os.getenv("OPENAI_API_KEY")

# argparse를 사용하여 사용자 입력을 처리
def parse_arguments():
    parse_arguments = argparse.ArgumentParser("DEMO: ChatGPT를 사용한 FAQ 시스템")
    parse_arguments.add_argument("--infra-file", type=str, required=True, help="자동으로 metrics_data 폴더에서 EC2 데이터를 로드할 수 있도록 .csv 파일 경로를 지정했음.")
    parse_arguments.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI 모델 이름 (기본값: gpt-4o-mini)")
    return parse_arguments.parse_args()

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

def generate_prompt(ec2_stats, question):
    """
    Generate a prompt dynamically based on EC2 stats and a user question.
    
    Args:
        ec2_stats (dict): A dictionary where keys are resource names and values are tuples of stats.
        question (str): The question or task to ask ChatGPT.
    
    Returns:
        str: The formatted prompt.
    """
    # Dynamically generate the resource stats section
    resource_stats = "\n".join(
        f"- {key} {' '.join(value)}" for key, value in ec2_stats.items()
    )
    
    # Define the template
    TEMPLATE = """
        당신은 AWS EC2 인스턴스의 전문가입니다. 아래는 EC2 인스턴스의 리소스 사용 데이터입니다.

        {resource_stats}

        주어진 데이터를 바탕으로 주요 트렌드와 이상치를 요약하고, 필요한 경우 최적화 제안을 추가하세요.

        질문: {question}
    """
    return TEMPLATE.format(resource_stats=resource_stats, question=question)


def ask_gpt(prompt, openai_key=openai_key):
    """
    Send a prompt to ChatGPT and return its response.

    Args:
        prompt (str): The prompt to send to ChatGPT.
        openai_key (str): OpenAI API key for authentication.
    
    Returns:
        str: ChatGPT's response.
    """
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(
        openai_api_key=openai_key,
        model_name="gpt-4",
        temperature=0.5,
    )

    # Send the prompt as a HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content

def process_ec2_question(file_path, question):
    """
    Process an EC2-related question by generating statistics, creating a prompt, and asking GPT.
    """
    # Load EC2 data
    ec2_data = load_ec2_data(file_path)

    # Generate statistics
    ec2_stats = generate_statistics(ec2_data)

    # Create prompt
    prompt = generate_prompt(ec2_stats, question)

    # Get answer from GPT
    answer = ask_gpt(prompt)
    return answer

# Step 3: FAQ QnA 시스템
def main():
    # Parse arguments
    args = parse_arguments()

    # Load EC2 data
    file_path = args.infra_file
    print("Loading EC2 data...")
    ec2_data = load_ec2_data(file_path)

    # Generate statistics
    print("Generating statistics...")
    ec2_stats = generate_statistics(ec2_data)
    print("Statistics:")
    print(ec2_stats)

    # Ask a question to the FAQ system
    print("Asking FAQ system...")
    question = "현재 인프라 상황에 대해 말해주고 조치 방안을 제안해주세요."
    faq_response = process_ec2_question(file_path, question)
    print("FAQ Answer:")
    print(faq_response)

if __name__ == "__main__":
    main()