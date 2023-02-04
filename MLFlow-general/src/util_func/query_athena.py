# from source.configuration import config
import time

import boto3
import pandas as pd
from botocore.config import Config

from cloudwatch_logger import get_logger_for_cloudwatch

DATABASE = "mleventsdatabase-n8xjx9ow3qm5"
BUCKET = "sagemaker-us-east-1-882748442234"
ATHENA_QUERY_OUTPUT = "athena_query_results"
BOTO_CONFIG = Config(retries = {"max_attempts": 20, "mode": "legacy"})

athena_client = boto3.client("athena", config = BOTO_CONFIG)
s3_client = boto3.client("s3", config = BOTO_CONFIG)
logger = get_logger_for_cloudwatch(__name__)


def execute_query(
        query_string,
        database = f"{DATABASE}",
        output_location = f"s3://{BUCKET}/{ATHENA_QUERY_OUTPUT}/",
):
    response = athena_client.start_query_execution(
        QueryString = query_string,
        QueryExecutionContext = {"Database": database, },
        ResultConfiguration = {"OutputLocation": output_location, },
    )

    response = athena_client.get_query_execution(
        QueryExecutionId = response["QueryExecutionId"]
    )

    status = response["QueryExecution"]["Status"]["State"]

    while status not in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        logger.info("athena query is running")
        time.sleep(20)
        response = athena_client.get_query_execution(
            QueryExecutionId = response["QueryExecution"]["QueryExecutionId"]
        )
        status = response["QueryExecution"]["Status"]["State"]

    if status == "SUCCEEDED":
        results = pd.read_csv(
            response["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
        )
        return results
    elif status == 'FAILED':
        print(response)
        raise Exception(f"Athena Query failed error='{response['QueryExecution']['Status']['StateChangeReason']}'")


results = execute_query(
    "SELECT * FROM raw_events WHERE year = '2021' AND month = '09' AND day = '23' LIMIT 100;"
)
results.to_csv(
    "s3://tmg-machine-learning-models-dev/for-you-payer-training-data/test.csv",
    index = False, header = True, sep = ",", float_format = "%.10f"
)
logger.info(f"Creating csv file with key s3://tmg-machine-learning-models-dev/for-you-payer-training-data/test.csv")
