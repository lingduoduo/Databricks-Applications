# Databricks notebook source
# MAGIC %md ##MLflow quickstart part 2: serving models using Amazon SageMaker
# MAGIC 
# MAGIC The [first part of this guide](https://docs.databricks.com/applications/mlflow/tracking-ex-scikit.html), **MLflow quickstart: model training and logging**, focuses on training a model and logging the training metrics, parameters, and model to the MLflow tracking server. 
# MAGIC 
# MAGIC ##### NOTE: Do not use *Run All* with this notebook. It takes several minutes to deploy and update models in SageMaker, and models cannot be queried until they are active.
# MAGIC 
# MAGIC This part of the guide consists of the following sections:
# MAGIC 
# MAGIC #### Setup
# MAGIC * Select a model to deploy using the MLflow tracking UI
# MAGIC 
# MAGIC #### Deploy a model
# MAGIC * Deploy the selected model to SageMaker using the MLflow API
# MAGIC * Check the status and health of the deployed model
# MAGIC   * Determine if the deployed model is active and ready to be queried
# MAGIC 
# MAGIC #### Query the deployed model
# MAGIC * Load an input vector that the deployed model can evaluate
# MAGIC * Query the deployed model using the input
# MAGIC 
# MAGIC #### Manage the deployment
# MAGIC * Update the deployed model using the MLflow API
# MAGIC * Query the updated model
# MAGIC 
# MAGIC #### Clean up the deployment
# MAGIC * Delete the model deployment using the MLflow API
# MAGIC 
# MAGIC As in the first part of the quickstart tutorial, this notebook uses ElasticNet models trained on the `diabetes` dataset in scikit-learn.

# COMMAND ----------

# MAGIC %md ## Prerequisites
# MAGIC 
# MAGIC ElasticNet models from the MLflow quickstart notebook in [part 1 of the quickstart guide](https://docs.databricks.com/applications/mlflow/tracking-ex-scikit.html).

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Ensure you are using or create a cluster specifying: 
# MAGIC   * **Python Version:** Python 3
# MAGIC   * An attached IAM role that supports SageMaker deployment. For information about setting up a cluster IAM role for SageMaker deployment, see the [SageMaker deployment guide](https://docs.databricks.com/administration-guide/cloud-configurations/aws/sagemaker.html).
# MAGIC 1. If you are running Databricks Runtime, uncomment and run Cmd 5 to install the required libraries. If you are running Databricks Runtime for Machine Learning, you can skip this step as the required libraries are already installed. 
# MAGIC 1. Attach this notebook to the cluster.

# COMMAND ----------

#dbutils.library.installPyPI("mlflow", version="1.0.0", extras="extras")
#dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Choose a run ID associated with an ElasticNet training run from [part 1 of the quickstart guide](https://docs.databricks.com/applications/mlflow/tracking-ex-scikit.html). You can find a run ID and model path from the experiment run, which can be found on the MLflow UI run details page:
# MAGIC 
# MAGIC ![image](https://docs.databricks.com/_static/images/mlflow/mlflow-deployment-example-run-info.png)

# COMMAND ----------

# MAGIC %md ### Set region, run ID, model URI
# MAGIC 
# MAGIC **Note**: You must create a new SageMaker endpoint for each new region.

# COMMAND ----------

region = "<region>"
run_id1 = "<run-id1>"
model_uri = "runs:/" + run_id1 + "/model"

# COMMAND ----------

# MAGIC %md ### Deploy a model
# MAGIC 
# MAGIC In this section, deploy the model you selected during **Setup** to SageMaker.

# COMMAND ----------

# MAGIC %md Specify a Docker image in Amazon's Elastic Container Registry (ECR). SageMaker uses this image to serve the model.  
# MAGIC To obtain the container URL, build the `mlflow-pyfunc` image and upload it to an ECR repository using the MLflow CLI: `mlflow sagemaker build-and-push-container`.

# COMMAND ----------

# MAGIC %md Define the ECR URL for the `mlflow-pyfunc` image that will be passed as an argument to MLflow's `deploy` function.

# COMMAND ----------

# Replace <ECR-URL> in the following line with the URL for your ECR docker image
# The ECR URL should have the following format: {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
image_ecr_url = "<ECR-URL>"

# COMMAND ----------

# MAGIC %md Use MLflow's SageMaker API to deploy your trained model to SageMaker. The `mlflow.sagemaker.deploy()` function creates a SageMaker endpoint as well as all intermediate SageMaker objects required for the endpoint.

# COMMAND ----------

import mlflow.sagemaker as mfs
app_name = "diabetes-class"
mfs.deploy(app_name=app_name, model_uri=model_uri, image_url=image_ecr_url, region_name=region, mode="create")

# COMMAND ----------

# MAGIC %md #### Using a single function, your model has now been deployed to SageMaker.

# COMMAND ----------

# MAGIC %md Check the status of your new SageMaker endpoint by running the following cell.
# MAGIC 
# MAGIC **Note**: The application status should be **Creating**. Wait until the status is **InService**; until then, query requests will fail.

# COMMAND ----------

import boto3

def check_status(app_name):
  sage_client = boto3.client('sagemaker', region_name=region)
  endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
  endpoint_status = endpoint_description["EndpointStatus"]
  return endpoint_status

print("Application status is: {}".format(check_status(app_name)))

# COMMAND ----------

# MAGIC %md ### Query the deployed model

# COMMAND ----------

# MAGIC %md #### Load sample input from the `diabetes` dataset

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn import datasets

# Load diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create a pandas DataFrame that serves as sample input for the deployed ElasticNet model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)
query_df = data.drop(["progression"], axis=1).iloc[[0]]

# Convert the sample input dataframe into a JSON-serialized pandas dataframe using the `split` orientation
input_json = query_df.to_json(orient="split")

# COMMAND ----------

print("Using input dataframe JSON: {}".format(input_json))

# COMMAND ----------

# MAGIC %md #### Evaluate the sample input by sending an HTTP request
# MAGIC Query the SageMaker endpoint REST API using the `sagemaker-runtime` API provided in `boto3`.

# COMMAND ----------

import json

def query_endpoint(app_name, input_json):
  client = boto3.session.Session().client("sagemaker-runtime", region)
  
  response = client.invoke_endpoint(
      EndpointName=app_name,
      Body=input_json,
      ContentType='application/json; format=pandas-split',
  )
  preds = response['Body'].read().decode("ascii")
  preds = json.loads(preds)
  print("Received response: {}".format(preds))
  return preds

print("Sending batch prediction request with input dataframe json: {}".format(input_json))

# Evaluate the input by posting it to the deployed model
prediction1 = query_endpoint(app_name=app_name, input_json=input_json)

# COMMAND ----------

# MAGIC %md ### Manage the deployment
# MAGIC 
# MAGIC You can update the deployed model by replacing it with the output of a different run. Specify the run ID associated with a different ElasticNet training run.

# COMMAND ----------

run_id2 = "<run-id2>"
model_uri = "runs:/" + run_id2 + "/model"

# COMMAND ----------

# MAGIC %md Call `mlflow.sagemaker.deploy()` in `replace` mode. This updates the `diabetes-class` application endpoint with the model corresponding to the new run ID.

# COMMAND ----------

mfs.deploy(app_name=app_name, model_uri=model_uri, image_url=image_ecr_url, region_name=region, mode="replace")

# COMMAND ----------

# MAGIC %md **Note**: The endpoint status should be **Updating**. Only after the endpoint status changes to **InService** do query requests use the updated model.

# COMMAND ----------

print("Application status is: {}".format(check_status(app_name)))

# COMMAND ----------

# MAGIC %md Query the updated model. You should get a different prediction.

# COMMAND ----------

prediction2 = query_endpoint(app_name=app_name, input_json=input_json)

# COMMAND ----------

# MAGIC %md Compare the predictions.

# COMMAND ----------

print("Run ID: {} Prediction: {}".format(run_id1, prediction1)) 
print("Run ID: {} Prediction: {}".format(run_id2, prediction2))

# COMMAND ----------

# MAGIC %md ### Clean up the deployment
# MAGIC 
# MAGIC When the model deployment is no longer needed, run the `mlflow.sagemaker.delete()` function to delete it.

# COMMAND ----------

# Specify the archive=False option to delete any SageMaker models and configurations
# associated with the specified application
mfs.delete(app_name=app_name, region_name=region, archive=False)

# COMMAND ----------

# MAGIC %md Verify that the SageMaker endpoint associated with the application has been deleted.

# COMMAND ----------

def get_active_endpoints(app_name):
  sage_client = boto3.client('sagemaker', region_name=region)
  app_endpoints = sage_client.list_endpoints(NameContains=app_name)["Endpoints"]
  return list(filter(lambda en : en == app_name, [str(endpoint["EndpointName"]) for endpoint in app_endpoints]))
  
print("The following endpoints exist for the `{an}` application: {eps}".format(an=app_name, eps=get_active_endpoints(app_name)))
