[![MLflow testing build](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml/badge.svg)](https://github.com/tmg-ling/mlflow-tmg-ling/actions/workflows/main.yml)

![AWS Cloud build](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoidkVqS2xWdGpvOHFCZ2hFd3BqalJoQ3gvT21GUXg1YjNxd0FFRFhyRStnSkVIT3dhNmloNksxVlNXTnBOSm8zVFQxdFFzbGNVSVZ2cHBVT3ZVb2tBOFlrPSIsIml2UGFyYW1ldGVyU3BlYyI6IjdhRnNJZ1pCN3BRKy92b0wiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main)

### About

MLflow is an open-source platform for managing ML lifecycles, including experimentation, deployment, and creation of a
central model registry. The MLflow Tracking component is an API that logs and loads the parameters, code versions, and
artifacts from ML model experiments.

- mlflow.tensorflow.autolog() enables you to automatically log the experiment in the local directory. It captures the
  metrics produced by the underlying ML library in use. MLflow Tracking is the module responsible for handling metrics
  and logs. By default, the metadata of an MLflow run is stored in the local filesystem.
    - The MLmodel file is the main definition of the project from an MLflow project with information related to how to
      run inference on the current model.
    - The metrics folder contains the training score value of this particular run of the training process, which can be
      used to benchmark the model with further model improvements down the line.
    - The params folder on the first listing of folders contains the default parameters of the logistic regression
      model, with the different default possibilities listed transparently and stored automatically.
- mlflow.set_experiment()
- mlflow.start_run() start and tear down the experiemnt in MLflow
- mlflow.log_param to log string-type test parameters
- mlflow.log_metrics to log numeric values
- mlflow.log_artifact to log the entire file that execute the function to ensure traceability of the model and codee
  that originated in the run

### Setup

1. Start a virtual enviornment

```bash
python3 -m venv ~/.venv                  
source ~/.venv/bin/activate 
```

2. Make install requirements

Need the following files to install requirement enviornment

- Makefile: Makefile
- requirements.txt: requirements.txt

```bash
make all
```

3. Run training jobs

- Train a model 
```bash
python train_gift_dcm.py --experiment_name gift_model --batch_size 16384 --learning_rate 0.05
python train_gift_lightgbm.py --n_estimators 300 --learning_rate 1
```

or run python in background

```bash
nohup python train_gift_dcm.py --experiment_name gift_model --batch_size 16384 --learning_rate 0.1 > nohup.out 2>&1 &
nohup python python train_gift_lightgbm.py --n_estimators 300 --learning_rate 1 > nohup.out 2>&1 &
```

4. Run mlflow ui

```bash
mlflow ui
```

5. safely shut down
```
ps -A | grep gunicorn
```
Take the PID and kill the process

6. Run mlflow job

```
mlflow run  -P alpha=0.4
```

```
yum update -y
yum install -y curl
http://localhost:5000
```

7. Start the serving API

8. Test the API
