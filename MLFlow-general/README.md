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

### Databricks

- Install Databricks CLI
```
pip install databricks-cli
```

- Config Databricks token
```
databricks configure --token
```
Check access credentials `~/.databrickscfg`

- Test authentication setup
```
databricks workspace ls /Users/lhuang@themeetgroup.com
databricks workspace ls --absolute --long --id /Users/lhuang@themeetgroup.com
```

- Print available clusters
```
databricks clusters list --output JSON | jq '[ .clusters[] | { name: .cluster_name, id: .cluster_id } ]'
```

- Configure MLflow tracking uri
```
export MLFLOW_TRACKING_URI=databricks

databricks fs ls dbfs:/mnt
databricks fs ls dbfs:/FileStore
databricks fs ls dbfs:/databricks-results
```

- Test Run tutorial MLflow Projects on Databricks
```
mlflow run https://github.com/mlflow/mlflow\#examples/sklearn_elasticnet_wine -v 989bb1158944484e4ffbcb479f20353fbaa5ea09 -b databricks --backend-config cluster-spec.json --experiment-id 90a4601a64624ca8a2a0115a6f9eea09
```

- Download model artifacts
```
mlflow artifacts download -r 1cd4da38afc24117a9f4637708b24267 
```
