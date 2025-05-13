# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="d5223bc2-6b7c-4596-b945-ed2f2cb22d2e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Model Registry
# MAGIC 
# MAGIC MLflow Model Registry is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance.  This lesson explores how to manage models using the MLflow model registry.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Register a model using MLflow
# MAGIC  - Deploy that model into production
# MAGIC  - Update a model in production to new version including a staging phase for testing
# MAGIC  - Archive and delete models

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="c4f47d56-1cc8-4b97-b89f-63257dbb3e31"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Model Registry
# MAGIC 
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions (e.g. from staging to production), annotations (e.g. with comments, tags), and deployment management (e.g. which production jobs have requested a specific model version).
# MAGIC 
# MAGIC Model registry has the following features:<br><br>
# MAGIC 
# MAGIC * **Central Repository:** Register MLflow models with the MLflow Model Registry. A registered model has a unique name, version, stage, and other metadata.
# MAGIC * **Model Versioning:** Automatically keep track of versions for registered models when updated.
# MAGIC * **Model Stage:** Assigned preset or custom stages to each model version, like “Staging” and “Production” to represent the lifecycle of a model.
# MAGIC * **Model Stage Transitions:** Record new registration events or changes as activities that automatically log users, changes, and additional metadata such as comments.
# MAGIC * **CI/CD Workflow Integration:** Record stage transitions, request, review and approve changes as part of CI/CD pipelines for better control and governance.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">the MLflow docs</a> for more details on the model registry.

# COMMAND ----------

# MAGIC %md <i18n value="ae6da8a8-dcca-4d34-b7fc-f06395f339f0"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Registering a Model
# MAGIC 
# MAGIC The following workflow will work with either the UI or in pure Python.  This notebook will use pure Python.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Explore the UI throughout this lesson by clicking the "Models" tab on the left-hand side of the screen.

# COMMAND ----------

# MAGIC %md <i18n value="c06f6e05-ab27-41da-8466-c092b3fcc1f6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Train a model and log it to MLflow.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

n_estimators = 100
max_depth = 5

rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
rf.fit(X_train, y_train)

input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))

with mlflow.start_run(run_name="RF Model") as run:
    mlflow.sklearn.log_model(rf, "model", input_example=input_example, signature=signature)
    mlflow.log_metric("mse", mean_squared_error(y_test, rf.predict(X_test)))
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="ee061209-c472-4fea-9a05-52fc45a80bb4"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Create a unique model name so you don't clash with other workspace users.

# COMMAND ----------

import uuid

uid = uuid.uuid4().hex[:6]
model_name = f"airbnb-rf-model_{uid}"
model_name

# COMMAND ----------

# MAGIC %md <i18n value="0650f73b-bf51-4494-8262-c04f470c73bd"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Register the model.

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="99071a6c-e260-40d2-b795-dc9782d3fc2b"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Open the *Models* tab on the left of the screen to explore the registered model.**  Note the following:<br><br>
# MAGIC 
# MAGIC * It logged who trained the model and what code was used
# MAGIC * It logged a history of actions taken on this model
# MAGIC * It logged this model as a first version
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/301/registered_model_new.png" style="height: 600px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="cd30915c-1fa2-422e-9137-c9c509bed8f4"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Check the status.  It will initially be in **`PENDING_REGISTRATION`** status.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md <i18n value="2b32a505-bbbd-4c5b-9967-bab1054784c8"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now add a model description

# COMMAND ----------

client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

# COMMAND ----------

# MAGIC %md <i18n value="89c43607-2402-4a80-97ee-b6b1378d5563"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Add a version-specific description.

# COMMAND ----------

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using sklearn."
)

# COMMAND ----------

# MAGIC %md <i18n value="932e4581-9172-4720-833c-210174f8814e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Deploying a Model
# MAGIC 
# MAGIC The MLflow Model Registry defines several model stages: **`None`**, **`Staging`**, **`Production`**, and **`Archived`**. Each stage has a unique meaning. For example, **`Staging`** is meant for model testing, while **`Production`** is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages. In private preview, any user can transition a model to any stage. In the near future, administrators in your organization will be able to control these permissions on a per-user and per-model basis.
# MAGIC 
# MAGIC If you have permission to transition a model to a particular stage, you can make the transition directly by using the **`MlflowClient.update_model_version()`** function. If you do not have permission, you can request a stage transition using the REST API; for example: ***```%sh curl -i -X POST -H "X-Databricks-Org-Id: <YOUR_ORG_ID>" -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" https://<YOUR_DATABRICKS_WORKSPACE_URL>/api/2.0/preview/mlflow/transition-requests/create -d '{"comment": "Please move this model into production!", "model_version": {"version": 1, "registered_model": {"name": "power-forecasting-model"}}, "stage": "Production"}'
# MAGIC ```***

# COMMAND ----------

# MAGIC %md <i18n value="64af3f73-388f-42a4-86cc-a3e2e5f713db"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now that you've learned about stage transitions, transition the model to the **`Production`** stage.

# COMMAND ----------

import time

time.sleep(10) # In case the registration is still pending

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md <i18n value="d486cd3a-485e-4470-9017-6b03058901da"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Fetch the model's current status.

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md <i18n value="2bb842d8-88dd-487a-adfb-3d69aa9aa1a5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Fetch the latest model using a **`pyfunc`**.  Loading the model in this way allows us to use the model regardless of the package that was used to train it.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You can load a specific version of the model too.

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md <i18n value="a7ee0d75-f55c-477a-938b-719ea17c2b8c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Apply the model.

# COMMAND ----------

model_version_1.predict(X_test)

# COMMAND ----------

# MAGIC %md <i18n value="2cdee524-1465-4bfb-9e62-5ed1307b353e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Deploying a New Model Version
# MAGIC 
# MAGIC The MLflow Model Registry enables you to create multiple model versions corresponding to a single registered model. By performing stage transitions, you can seamlessly integrate new model versions into your staging or production environments.

# COMMAND ----------

# MAGIC %md <i18n value="d0bda674-ba42-4fa5-9d7c-194e567e1478"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Create a new model version and register that model when it's logged.

# COMMAND ----------

n_estimators = 300
max_depth = 10

rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
rf.fit(X_train, y_train)

input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))

with mlflow.start_run(run_name="RF Model") as run:
    # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
    # function to register the model with the MLflow Model Registry. This automatically
    # creates a new model version
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="sklearn-model",
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature
    )
    mlflow.log_metric("mse", mean_squared_error(y_test, rf.predict(X_test)))

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="27044c82-1822-4090-96a1-82e266b1ce98"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Check the UI to see the new model version.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/301/model_version_new.png" style="height: 600px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="7d124f10-d8ef-4af6-8e5d-6d0b3d5eecae"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Use the search functionality to grab the latest model version.

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(f"New model version: {new_model_version}")

# COMMAND ----------

# MAGIC %md <i18n value="d10030fe-a13e-4fcb-a13e-056d1bcb8683"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Add a description to this new version.

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="This model version is a random forest containing 300 decision trees and a max depth of 10 that was trained in scikit-learn."
)

# COMMAND ----------

# MAGIC %md <i18n value="039af76f-ed42-4d52-bd96-09f4853322be"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Put this new model version into **`Staging`**

# COMMAND ----------

time.sleep(10) # In case the registration is still pending

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md <i18n value="bba43d4f-da53-4f09-a324-a1fd6f3f7d74"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Since this model is now in staging, you can execute an automated CI/CD pipeline against it to test it before going into production.  Once that is completed, you can push that model into production.

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production",
    archive_existing_versions=True # Archive old versions of this model
)

# COMMAND ----------

# MAGIC %md <i18n value="2620b2ab-8fa7-4118-8212-5b9f19af7a8d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Deleting
# MAGIC 
# MAGIC You can now delete old versions of the model.

# COMMAND ----------

# MAGIC %md <i18n value="49c8fbc4-6b5d-4939-a8e7-2b6cc0945909"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Delete version 1.  
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You cannot delete a model that is not first archived.

# COMMAND ----------

client.delete_model_version(
    name=model_name,
    version=1
)

# COMMAND ----------

# MAGIC %md <i18n value="36e94f58-20c9-4a0e-bc57-3a5c258591a0"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Archive version 2 of the model too.

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

# MAGIC %md <i18n value="967a313e-45ca-4d7d-b423-3c660e0f6136"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now delete the entire registered model.

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------

# MAGIC %md <i18n value="cd34a8c0-ed7d-458b-ac28-55664acd3231"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Review
# MAGIC **Question:** How does MLflow tracking differ from the model registry?  
# MAGIC **Answer:** Tracking is meant for experimentation and development.  The model registry is designed to take a model from tracking and put it through staging and into production.  This is often the point that a data engineer or a machine learning engineer takes responsibility for the depoloyment process.
# MAGIC 
# MAGIC **Question:** Why do I need a model registry?  
# MAGIC **Answer:** Just as MLflow tracking provides end-to-end reproducibility for the machine learning training process, a model registry provides reproducibility and governance for the deployment process.  Since production systems are mission critical, components can be isolated with ACL's so only specific individuals can alter production models.  Version control and CI/CD workflow integration is also a critical dimension of deploying models into production.
# MAGIC 
# MAGIC **Question:** What can I do programatically versus using the UI?  
# MAGIC **Answer:** Most operations can be done using the UI or in pure Python.  A model must be tracked using Python, but from that point on everything can be done either way.  For instance, a model logged using the MLflow tracking API can then be registered using the UI and can then be pushed into production.

# COMMAND ----------

# MAGIC %md <i18n value="9aeb78b7-1eba-4bd9-ae48-ab4cd792012f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Next we'll take a look at, [Webhooks and Testing]($./03a-Webhooks-and-Testing)

# COMMAND ----------

# MAGIC %md <i18n value="7cae8131-c410-449a-9581-f52899a6c799"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Model Registry?  
# MAGIC **A:** Check out <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">the MLflow documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
