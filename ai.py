import os
import argparse
from ec2_statistics import load_ec2_data, generate_statistics
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Load the OpenAI API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

# Parse arguments using argparse
def parse_arguments():
    parse_arguments = argparse.ArgumentParser("DEMO: ChatGPT를 사용한 FAQ 시스템")
    parse_arguments.add_argument("--infra-file", type=str, default='', help="자동으로 metrics_data 폴더에서 EC2 데이터를 로드할 수 있도록 .csv 파일 경로를 지정했음.")
    parse_arguments.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI 모델 이름 (기본값: gpt-4o-mini)")
    parse_arguments.add_argument("--pdf-file", type=str, default='/pdf/AmazonElasticComputeCloudDevGuide.pdf' ,help="PDF 파일 경로")
    return parse_arguments.parse_args()


# Load and split PDF into chunks
def load_pdf_and_split(file_path):
    loader = PDFPlumberLoader(file_path)
    pdf_data = loader.load()
    print(f"Loading PDF file from {file_path}...")
    print(f"Number of pages: {len(pdf_data)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n"])
    split_data = text_splitter.split_documents(pdf_data)
    print(f"Split PDF into {len(split_data)} chunks.")
    return split_data

# Setup the vector database
def setup_embeddings_and_chroma(pdf_file):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    collection_name = "ec2_document"
    persist_dir = os.path.join(os.getcwd(), "ec2_document_embedding")
    split_data = load_pdf_and_split(pdf_file)

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    vecDB = Chroma.from_documents(split_data, embeddings, collection_name=collection_name, persist_directory=persist_dir)
    vecDB.persist()
    print("Vector database initialized and persisted.")
    return vecDB.as_retriever(search_type='similarity', search_kwargs={'k': 5})

def format_prompt(ec2_stats, question):
    """
    EC2 통계와 질문을 기반으로 PromptTemplate 객체와 데이터를 반환다.
    
    Args:
        ec2_stats (dict): EC2 인스턴스의 리소스 사용 데이터.
        question (str): 사용자 질문.

    Returns:
        tuple: (PromptTemplate 객체, dict 포맷팅 데이터)
    """
    # 리소스 통계를 문자열로 변환
    resource_stats = (
        "\n".join(f"- {key}: {' '.join(value)}" for key, value in ec2_stats.items())
        if ec2_stats else "No resource statistics available."
    )

    # 기본값 설정
    chat_history = "No chat history available."
    context = "No context available."
    question = question

    # PromptTemplate 생성
    prompt_template = PromptTemplate.from_template(
        template=f"""
            당신은 AWS EC2 인스턴스의 전문가입니다. 아래는 EC2 인스턴스의 리소스 사용 데이터입니다.
            {resource_stats}

            Previous Chat History:
            {chat_history}

            Question: 
            {question} 

            Context: 
            {context}

            Answer: 
        """
    )

    return prompt_template

# Setup the conversational retrieval chain
def setup_conversational_chain(retriever, prompt):
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o-mini", temperature=0.7)
    # prompt = PromptTemplate(template="질문: {question}\n답변:", language="ko")
    chain = (
        {   
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    # 세션 기록을 저장할 딕셔너리
    store = {}
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

# Main function with user input
def main():
    args = parse_arguments()
    print()
    # Load EC2 data
    ec2_data_file = os.getcwd()
    pdf_file = os.getcwd() + args.pdf_file
    print(f"Loading EC2 data from {ec2_data_file}...")
    ec2_data = load_ec2_data(ec2_data_file)

    # Generate statistics
    print("Generating EC2 statistics...")
    ec2_stats = generate_statistics(ec2_data)
    print("Statistics generated.")

    # Setup PDF retriever
    print("Setting up retriever from PDF...")
    retriever = setup_embeddings_and_chroma(pdf_file)

    # Setup conversational chain
    prompt_template = format_prompt(ec2_stats, "주어진 데이터를 바탕으로 주요 트렌드와 이상치를 요약하고, 필요한 경우 최적화 제안을 추가하세요.")

    chatQA = setup_conversational_chain(retriever, prompt=prompt_template)

    # 대화를 기록하는 RAG 체인 생성 
    rag_with_history = RunnableWithMessageHistory(
        chatQA,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    # RAG Chain 실행
    re_1 = rag_with_history.invoke(
        {"question": "aws에서 제공하는 컴퓨팅 instant의 이름은?"},  # 메시지 리스트 전달
        config={"configurable": {"session_id": "rag123"}},
    )

    print("Debug: Invoke result:", re_1)

if __name__ == "__main__":
    main()