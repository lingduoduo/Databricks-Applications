[![MLflow testing build](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml/badge.svg)](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml)

![AWS Cloud build](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoidkVqS2xWdGpvOHFCZ2hFd3BqalJoQ3gvT21GUXg1YjNxd0FFRFhyRStnSkVIT3dhNmloNksxVlNXTnBOSm8zVFQxdFFzbGNVSVZ2cHBVT3ZVb2tBOFlrPSIsIml2UGFyYW1ldGVyU3BlYyI6IjdhRnNJZ1pCN3BRKy92b0wiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main)

### MLflow - Model tracking and monitoring

MLflow is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry. It includes the following components:

- Projects – Package data science code in a format to reproduce runs on any platform
- Registry – Store, annotate, discover, and manage models in a central repository
- Tracking – Record and query experiments: code, data, configuration, and results
- Deployment – Deploy ML models in diverse serving environments

#### Setup the Enviornment

1. Start a virtual enviornment

Setup a Python virtual environment

```bash           
source ~/.venv/bin/activate 
```

or 

```bash
conda create -n mlflow_env python=3.9
```

2. Make install requirements

Need the Makefile and requirements.txt files to install the dependencies.

```bash
export PYTHONPATH=.
make all
```

or

```bash
pip install -r reqirements.txt
```

#### Run the models and track metrics in MLflow

1. Run training jobs

* Lightgbm native

```bash
python -m src.lightgbm_native.train --learning_rate 0.01
nohup python -m src.lightgbm_native.train --learning_rate 0.01 > nohup.out 2>&1 &
```

* Lightgbm regression

```bash
python -m src.lightgbm_gift.train --n_estimators 300 --learning_rate 1
nohup python -m src.lightgbm_gift.train --n_estimators 300 --learning_rate 1 > nohup.out 2>&1 &
```

* Two Tower Model

```bash
python -m src.tfrs_two_tower_gift.train --batch_size 16384 --learning_rate 0.05 --broadcaster_embedding_dimension 96 --viewer_embedding_dimension 96 --top_k 1000
nohup python -m src.tfrs_two_tower_gift.train --batch_size 16384 --learning_rate 0.05 --broadcaster_embedding_dimension 96 --viewer_embedding_dimension 96 --top_k 1000 > nohup.out 2>&1 &
```

* Deep and Cross Network

```bash
python -m src.tfrs_dcn_gift.train --batch_size 16384 --learning_rate 0.05
nohup python -m src.tfrs_dcn_gift.train --batch_size 16384 --learning_rate 0.05 > nohup.out 2>&1 &
```

* Listwise ranking

```bash
python -m src.tfrs_listwise_ranking_gift.train --batch_size 16384 --learning_rate 0.05
nohup python -m src.tfrs_listwise_ranking_gift.train --batch_size 16384 --learning_rate 0.05 > nohup.out 2>&1 &
```

2. Run mlflow

```bash
mlflow run .
mlflow run . -P learning_rate=0.01 -P n_estimators=300 
mlflow run . -P learning_rate=0.01 -P n_estimators=300 --experiment-name Baseline_Predictions
mlflow server --backend-store-uri sqlite:///:memory --default-artifact-root ./mlruns
```

3. Check model results and safely shut down

```bash
mlflow ui
ps -A | grep gunicorn
```

Take the PID and kill the process.

#### Deplpy the model to AWS

1. Build the docker image

```bash
docker build -t mlflow-tmg-ling .
docker tag mlflow-tmg-ling:latest 882748442234.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest

docker build -t mlflow-tfrs-dcn .
docker tag mlflow-tfrs-dcn:latest 882748442234.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest
```

2. Authentication token

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 882748442234.dkr.ecr.us-east-1.amazonaws.com
```

3. Push the local image to ECR

```bash
experiment_id = 0
run_id = e820dfefbda4487b8abf6ecdce65d728
cd mlruns/0/e820dfefbda4487b8abf6ecdce65d728/artifacts/model

mlflow sagemaker build-and-push-container
aws ecr describe-images --repository-name mlflow-pyfunc
docker push 882748442234.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest
```

4. Deploy Image from AWS ECR to AWS SageMaker

```bash
aws ecr describe-images --repository-name mlflow-pyfunc
python deploy.py
aws sagemaker list-endpoints
```

Check ECR, SageMaker and S3 accordingly.

5. Evaluate the predictions

```bash
python evaluate.py
```

6. Predict with Athena and SageMaker endpoint

```bash
DROP TABLE gift_testing_data;

CREATE EXTERNAL TABLE gift_testing_data
    (
        `viewer_id` string,
        `broadcaster_id` string,
        `product_name` string,
        `ordered_time` string,
        `count` int
    )
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
   'separatorChar' = ',',
   'quoteChar' = '"',
   'escapeChar' = '\\'
   )
STORED AS TEXTFILE
LOCATION 's3://tmg-machine-learning-models-dev/for-you-payer-training-data/'
TBLPROPERTIES('skip.header.line.count'='1')
;
USING EXTERNAL FUNCTION predict_avg_gift (broadcaster_id VARCHAR, 
    viewer_id VARCHAR, 
    product_name VARCHAR, 
    ordered_time VARCHAR
) 
RETURNS DOUBLE 
SAGEMAKER 'lightgbm-gift'
SELECT 
    predict_avg_gift("broadcaster_id","viewer_id", "product_name", "ordered_time") AS prediction
FROM gift_testing_data
LIMIT 10
;
```

### SageMaker - Using Docker containers with SageMaker

https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
https://sagemaker-examples.readthedocs.io/en/latest/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.html
https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/pipe_bring_your_own

Running your container during training

```angular2html
    /opt/ml
    |-- input
    |   |-- config
    |   |   |-- hyperparameters.json
    |   |    -- resourceConfig.json
    |    -- data
    |        -- <channel_name>
    |            -- <input data>
    |-- model
    |   -- <model files>
     -- output
        -- failure
```

Create a Docker image and train a model

1. Write a training script

2. Define a container with a Dockerfile

3. Build and tag the Docker image

4. Use the Docker image to start a training job

Pass arguments to the entry point using hyperparameters

1. Implement an argument parser in the entry point script 

2. Start a training job with hyperparameters

Read additional information using environment variables

Get information about the container environment

Execute the entry point

Running your container during hosting

```angular2html
/opt/ml
`-- model
    `-- <model files>
```
https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality

Tensorflow model - https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb

https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/xgboost_bring_your_own_model/xgboost_bring_your_own_model.ipynb

https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/search/ml_experiment_management_using_search.ipynb


### Databricks

- [Push Data Exploration](https://github.meetmecorp.com/pages/lhuang/push-sandbox/docs/pyspark-push-data-exploration_20220906.html)

- [Push Train Run Model](https://github.meetmecorp.com/pages/lhuang/push-sandbox/docs/pyspark-push-trail-run-model_20220907.html)

- [Sampled Data Profiling](https://github.meetmecorp.com/pages/lhuang/push-sandbox/docs/22-09-07-22_50-DataExploration-4defac725f71f5edbe4aa8d57edb9340.html)

- [Best Model Output from DataBricks AutoML](https://github.meetmecorp.com/pages/lhuang/push-sandbox/docs/22-09-07-22_50-LightGBM-f71a78bf7bf9afe8f71a156e47585a10.html)

