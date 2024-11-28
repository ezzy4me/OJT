from ai import visualize_ec2_data, summarize_ec2_data, faq_answer

def main():
    cpu_file = "static/cpu_utilization.csv"
    mem_file = "static/mem_used_percent.csv"

    # Step 1: EC2 데이터 시각화
    print("Visualizing EC2 data...")
    visualize_ec2_data(cpu_file, mem_file)

    # Step 2: EC2 데이터 요약
    print("Summarizing EC2 data...")
    summary = summarize_ec2_data(cpu_file, mem_file)
    print("Summary:")
    print(summary)

    # Step 3: FAQ QnA 시스템
    print("Asking FAQ system...")
    question = "What is the best instance type for machine learning workloads?"
    faq_response = faq_answer(question)
    print("FAQ Answer:")
    print(faq_response)

if __name__ == "__main__":
    main()
