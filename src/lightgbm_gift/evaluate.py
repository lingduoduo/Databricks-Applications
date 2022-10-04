import json
import os

import boto3
import mlflow
import mlflow.lightgbm
import pandas as pd


def feature_encoder(training_data, encoding_features):
    feature_mappings = {}
    for c in encoding_features:
        temp = training_data[c].astype("category").cat
        training_data[c] = temp.codes + 1
        feature_mappings[c] = {cat: n for n, cat in enumerate(temp.categories, start = 1)}
    return training_data, feature_mappings


def check_status(app_name, region):
    sage_client = boto3.client('sagemaker', region_name = region)
    endpoint_description = sage_client.describe_endpoint(EndpointName = app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
    return endpoint_status


def query_endpoint(app_name, input_json, region):
    client = boto3.session.Session().client("sagemaker-runtime", region)
    response = client.invoke_endpoint(
        EndpointName = app_name,
        Body = input_json,
        ContentType = 'application/json; format=pandas-split',
    )
    preds = response['Body'].read().decode("ascii")
    preds = json.loads(preds)
    return preds


def main():
    # prepare train and test data
    local_file = "src/csv/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    if not os.path.exists(local_file) and not os.path.isfile(local_file):
        filename = "s3://tmg-machine-learning-models-dev/for-you-payer-training-data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    else:
        filename = local_file
    # load data
    df = pd.read_csv(filename)

    experiment_id = '0'
    run_id = '7f857d1c8ebb44bd80bd26424d537317'
    # run_id = mlflow.active_run().info.run_id
    region = 'us-east-1'
    endpoint_name = 'lightgbm-gift'

    # prediction by local loaded model
    loaded_model = mlflow.pyfunc.load_model(f'mlruns/{experiment_id}/{run_id}/artifacts/model')
    FEATURES = ["broadcaster_id", "viewer_id", "product_name", "ordered_time"]
    df, feature_mappings = feature_encoder(df, FEATURES)
    df["weight"] = 1
    print(df.iloc[[3]])
    pred = loaded_model.predict(df.iloc[[3]])
    print(f"local prediction: {pred}")

    # check endpoint status
    print("Application status is: {}".format(check_status(endpoint_name, region)))

    # evaluate inference from endpoint
    query_input = df.iloc[[3]].to_json(orient = "split")
    pred = query_endpoint(app_name = endpoint_name, input_json = query_input, region = region)
    print(f"cloud prediction: {pred}")


if __name__ == "__main__":
    main()
