# Script to load credentials from AWS secrets manager
import os 
import boto3
import json
from botocore.exceptions import ClientError

def load_credentials():
    """Load credentials from AWS Secrets Manager, or .env file locally"""

    secret_name = 'mle_credentials'
    region = 'us-east-1'

    if os.getenv("AWS_EXECUTION_ENV") or os.getenv("SM_CURRENT_HOST"):
        # When running on EC2/SageMakejker - fetch from Secrets Manager
        client = boto3.client(
            service_name='secretsmanager',
            region_name=region
        )

        try:
            secret = client.get_secret_value(SecretId=secret_name)
        except ClientError as e:
            raise e

        credentials = json.loads(secret['SecretString'])

        for key, value in credentials.items():
            os.environ[key] = value
    else: #assume .env was already sourced locally, do nothing
        pass



