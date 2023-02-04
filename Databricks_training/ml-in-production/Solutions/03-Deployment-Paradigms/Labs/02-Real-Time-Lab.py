# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="54771b4e-fe73-4edb-8d87-9d9d4c2d7170"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Lab: Deploying a Real-time Model with MLflow Model Serving
# MAGIC MLflow Model Serving offers a fast way of serving pre-calculated predictions or creating predictions in real time. In this lab, you'll deploy a model using MLflow Model Serving.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Enable MLflow Model Serving for your registered model
# MAGIC  - Compute predictions in real time for your registered model via a REST API request
# MAGIC  
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> *You need <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">cluster creation</a> permissions to create a model serving endpoint. The instructor will either demo this notebook or enable cluster creation permission for the students from the Admin console.*

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="ad24ef8d-031e-435c-a1e0-e64de81b936d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC To start this off, we will need to load the data, build a model, and register that model.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We're building a random forest model to predict Airbnb listing prices.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import uuid

# Load data
df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X = df[["bathrooms", "bedrooms", "number_of_reviews"]]
y = df["price"]

# Start run
with mlflow.start_run(run_name="Random Forest Model") as run:
    # Train model
    n_estimators = 10
    max_depth = 5
    regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    regressor.fit(X, y)
    
    # Evaluate model
    y_pred = regressor.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    
    # Log params and metric
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse)
    
    # Log model
    mlflow.sklearn.log_model(regressor, "model")
    
# Register model
uid = uuid.uuid4().hex[:6]
model_name = f"{DA.unique_name}_rfr-model_{uid}"
model_uri = f"runs:/{run.info.run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
model_version = model_details.version

# COMMAND ----------

# MAGIC %md <i18n value="e3d6be06-1cc5-4ca6-9d81-ec237bba01bc"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Next, we will transition to model to staging.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md <i18n value="ad504f00-8d56-4ab4-aa94-6172107a1934"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Enable MLflow Model Serving for the Registered Model
# MAGIC 
# MAGIC Your first task is to enable MLflow Model Serving for the model that was just registered.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Check out the <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#enable-and-disable-model-serving" target="_blank">documentation</a> for a demo of how to enable model serving via the UI.
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/mlflow/demo_model_register.png" width="600" height="20"/>

# COMMAND ----------

# MAGIC %md <i18n value="44b586ec-ac4f-4a71-9c27-cc6ac5111ec8"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Compute Real-time Predictions
# MAGIC 
# MAGIC Now that your model is registered, you will query the model with inputs.
# MAGIC 
# MAGIC To do this, you'll first need the appropriate token and api_url.

# COMMAND ----------

import mlflow

# We need both a token for the API, which we can get from the notebook.
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an endpoint at which to execute our request which we can get from the Notebook's context
api_url = mlflow.utils.databricks_utils.get_webapp_url()
print(api_url)

# COMMAND ----------

# MAGIC %md <i18n value="cd88b865-b030-48f1-83f5-0f2f872cc757"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Enable the endpoint

# COMMAND ----------

import requests

url = f"{api_url}/api/2.0/mlflow/endpoints/enable"

r = requests.post(url, headers=headers, json={"registered_model_name": model_name})
assert r.status_code == 200, f"Expected an HTTP 200 response, received {r.status_code}"

# COMMAND ----------

# MAGIC %md <i18n value="3b7cf885-1789-49ac-be65-f61a9f8752d5"/>
# MAGIC 
# MAGIC We can redefine our two wait methods to ensure that the resources are ready before moving forward.

# COMMAND ----------

def wait_for_endpoint():
    import time
    while True:
        url = f"{api_url}/api/2.0/preview/mlflow/endpoints/get-status?registered_model_name={model_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("endpoint_status", {}).get("state", "UNKNOWN")
        if status == "ENDPOINT_STATE_READY": print("-"*80); return
        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds

# COMMAND ----------

def wait_for_version():
    import time
    while True:    
        url = f"{api_url}/api/2.0/preview/mlflow/endpoints/list-versions?registered_model_name={model_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        state = response.json().get("endpoint_versions")[0].get("state")
        if state == "VERSION_STATE_READY": print("-"*80); return
        else: print(f"Version not ready ({state}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds


# COMMAND ----------

# MAGIC %md <i18n value="2e33d989-988d-4673-853f-c7e0e568b3f9"/>
# MAGIC 
# MAGIC Next, create a function that takes a single record as input and returns the predicted value from the endpoint.

# COMMAND ----------

# ANSWER
import requests

def score_model(dataset: pd.DataFrame, model_name: str, token: str, api_url: str):
    url = f"{api_url}/model/{model_name}/1/invocations"
    data_json = dataset.to_dict(orient="split")
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)

    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()

# COMMAND ----------

# MAGIC %md <i18n value="ae8b57cb-67fe-4d36-91c2-ce4ec405e38e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now, use that function to score a single row of a Pandas DataFrame.

# COMMAND ----------

wait_for_endpoint()
wait_for_version()

# COMMAND ----------

# ANSWER

single_row_df = pd.DataFrame([[2, 2, 150]], columns=["bathrooms", "bedrooms", "number_of_reviews"])
score_model(single_row_df, model_name, token, api_url)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
