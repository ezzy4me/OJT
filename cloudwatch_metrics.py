import boto3
import csv
import os
from datetime import datetime
import argparse

def parse_arguments():
    """
    Parse command-line arguments for CloudWatch metrics retrieval, including AWS credentials.
    """
    parser = argparse.ArgumentParser(description="Retrieve CloudWatch metrics for an EC2 instance and save to CSV.")
    parser.add_argument("--access-key", type=str, required=True, help="AWS Access Key ID")
    parser.add_argument("--secret-key", type=str, required=True, help="AWS Secret Access Key")
    parser.add_argument("--instance-id", type=str, required=True, help="EC2 Instance ID (e.g., i-0abcdef1234567890)")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="List of CloudWatch metrics to retrieve (e.g., CPUUtilization diskio_write_bytes diskio_read_bytes mem_used_percent)"
    )
    parser.add_argument("--namespaces", type=str, nargs="+", required=True, help="List of namespaces to search metrics in (e.g., AWS/EC2 CWAgent)")
    parser.add_argument("--region", type=str, default='ap-northeast-2', required=True, help="AWS Region (e.g., ap-northeast-2)")
    parser.add_argument("--start-time", type=str, required=True, help="Start time in ISO format (e.g., 2024-11-21T00:00:00Z)")
    parser.add_argument("--end-time", type=str, required=True, help="End time in ISO format (e.g., 2024-11-21T23:59:59Z)")
    parser.add_argument("--output", type=str, default='ec2_metrics', required=True, help="Output CSV file name prefix (e.g., metrics.csv)")
    parser.add_argument("--period", type=int, default=300, help="Period in seconds for metrics (default: 300)")
    return parser.parse_args()

def get_metric_data(cloudwatch, namespaces, instance_id, metric_name, start_time, end_time, period):
    """
    Retrieve metric data from CloudWatch, trying multiple namespaces if needed.
    """
    for namespace in namespaces:
        response = cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'InstanceId',
                    'Value': instance_id
                },
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=['Average']
        )
        if response['Datapoints']:
            print(f"Found data for {metric_name} in namespace {namespace}")
            return response['Datapoints'], namespace
    print(f"No data found for {metric_name} in any of the provided namespaces")
    return [], None

def save_to_individual_csv(metric, namespace, filename, data):
    """
    Save individual metric data to a separate CSV file.
    """
    # 파일 이름에서 '/'를 '_'로 대체
    save_filename = ''.join(filename.split("/")[1:])
    save_filename = save_filename.replace("/", "")
    save_filename = filename.split("/")[0] + "/" + save_filename

    with open(save_filename, 'w', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Namespace', 'MetricName', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({'Timestamp': row['Timestamp'], 'Namespace': namespace, 'MetricName': metric, 'Value': row['Value']})
    print(f"Saved {metric} data to {save_filename}")


def main():
    # Parse arguments
    args = parse_arguments()

    # Parse ISO time to datetime objects
    start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%SZ")
    end_time = datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%SZ")

    # Initialize CloudWatch client with provided credentials
    cloudwatch = boto3.client(
        'cloudwatch',
        region_name=args.region,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key
    )
    test_metrics = cloudwatch.list_metrics(Namespace='CWAgent')
    for metric in test_metrics['Metrics']:
        print(metric)

    # Fetch and save metrics
    for metric in args.metrics:
        print(f"Retrieving data for metric: {metric}")
        datapoints, namespace = get_metric_data(
            cloudwatch,
            args.namespaces,
            args.instance_id,
            metric,
            start_time,
            end_time,
            args.period
        )
        if datapoints:
            # Prepare data for CSV
            metric_data = [
                {
                    'Timestamp': point['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Value': point['Average']
                } for point in datapoints
            ]
            # Save to CSV
            output_file = f"metrics_data/{metric}_{namespace}_{args.output}"
            save_to_individual_csv(metric, namespace, output_file, metric_data)
        else:
            print(f"No data found for metric: {metric}")

if __name__ == "__main__":
    main()