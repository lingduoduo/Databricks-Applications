# Databricks notebook source
# MAGIC %md 
# MAGIC ### Setup Bronze Table Connection

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt')

# COMMAND ----------

dbutils.fs.ls('/databricks')

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

bronze_df = spark.table("ml_push.bronze_l7_push_meetme")

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Feature Engineering

# COMMAND ----------

# import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
from pyspark.sql.window import Window

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####Label

# COMMAND ----------

silver_df = bronze_df.withColumn('open_flag', F.col('open_flag').cast(BooleanType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Features

# COMMAND ----------

silver_df = silver_df.withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id')))

# COMMAND ----------

for col_name in ['utc_day_of_week', 'utc_hour']:
    silver_df = silver_df.withColumn(col_name, F.col(col_name).cast(StringType()))

# COMMAND ----------

columns = [
    'open_flag',
    'device_type',
    'utc_day_of_week',
    'utc_hour',
    'broadcaster_id'
]

# COMMAND ----------

df_train, df_val = silver_df.select(columns).randomSplit([0.8, 0.2], seed=12345)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Save our `train` and `val` datasets into delta lake tables for future use.

# COMMAND ----------

database_name = 'ml_push'

silver_train_tbl_path = '/FileStore/shared_uploads/{}/push_data/silver_train/'.format(user)
silver_val_tbl_path = '/FileStore/shared_uploads/{}/push_data/silver_train/'.format(user)

silver_train_tbl_name = 'silver_l7_push_meetme_train'
silver_val_tbl_name = 'silver_l7_push_meetme_val'

# COMMAND ----------

import shutil
# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+silver_train_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_val_tbl_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Creating Silver Delta Table from Spark Dataframe

# COMMAND ----------

# save as delta table
df_train.write.format('delta').mode('overwrite').save(silver_train_tbl_path)
df_val.write.format('delta').mode('overwrite').save(silver_val_tbl_path)

# COMMAND ----------

# # Create silver table
# _ = spark.sql('''
#   CREATE TABLE `{}`.{}
#   USING DELTA 
#   LOCATION '{}'
#   '''.format(database_name,silver_train_tbl_name,silver_train_tbl_path))

# _ = spark.sql('''
#   CREATE TABLE `{}`.{}
#   USING DELTA 
#   LOCATION '{}'
#   '''.format(database_name,silver_val_tbl_name,silver_val_tbl_path))

# COMMAND ----------

for c in columns:
    (df_train
    .groupBy(c)
    .agg(F.mean(F.when(F.col('open_flag')==True, 1).otherwise(0)).alias('open rates'))
    .sort('open rates', ascending=False)
    .show())

# COMMAND ----------

for c in columns:
    (df_val
    .groupBy(c)
    .agg(F.mean(F.when(F.col('open_flag')==True, 1).otherwise(0)).alias('open rates'))
    .sort('open rates', ascending=False)
    .show())

# COMMAND ----------

# MAGIC %md
# MAGIC Create AutoML model to help us automatically test different models and parameters and reduce time manually testing and tweaking ML models. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###AutoML Model

# COMMAND ----------

silver_train_df = spark.table("ml_push.silver_l7_push_meetme_train")
silver_val_df = spark.table("ml_push.silver_l7_push_meetme_val")

# COMMAND ----------

silver_train_df.printSchema()

# COMMAND ----------

silver_val_df.printSchema()

# COMMAND ----------

import databricks.automl

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC data_dir is DBFS path used to store the training dataset. This path is visible to both driver and worker nodes. If empty, AutoML saves the training dataset as an MLflow artifact.

# COMMAND ----------

summary = databricks.automl.classify(silver_train_df, target_col='open_flag', primary_metric="f1", data_dir='dbfs:/automl/ml_push', timeout_minutes=30)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Register Model 

# COMMAND ----------

import mlflow
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# COMMAND ----------

model_uri = summary.best_trial.model_path

# COMMAND ----------

import uuid
 
uid = uuid.uuid4().hex[:6]
model_name = f"Test-Stage-Model_{uid}"
model_name

# COMMAND ----------

# Register models in the Model Registry
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name, version=model_version)
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

# COMMAND ----------

wait_until_ready(model_details.name, model_details.version)

# COMMAND ----------

### Add model and model version descriptions.
client.update_registered_model(
  name=model_details.name,
  description="This model predict push open probability."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using AutoML."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying a Model
# MAGIC 
# MAGIC The MLflow Model Registry defines several model stages: **`None`**, **`Staging`**, **`Production`**, and **`Archived`**. 
# MAGIC 
# MAGIC Each stage has a unique meaning. For example, **`Staging`** is meant for model testing, while **`Production`** is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages. In private preview, any user can transition a model to any stage. In the near future, administrators in your organization will be able to control these permissions on a per-user and per-model basis.
# MAGIC 
# MAGIC If you have permission to transition a model to a particular stage, you can make the transition directly by using the **`MlflowClient.update_model_version()`** function. 
# MAGIC 
# MAGIC If you do not have permission, you can request a stage transition using the REST API; for example: 
# MAGIC ***
# MAGIC ```
# MAGIC %sh curl -i -X POST -H "X-Databricks-Org-Id: <YOUR_ORG_ID>" -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" https://<YOUR_DATABRICKS_WORKSPACE_URL>/api/2.0/preview/mlflow/transition-requests/create -d '{"comment": "Please move this model into production!", "model_version": {"version": 1, "registered_model": {"name": "power-forecasting-model"}}, "stage": "Production"}'
# MAGIC ```
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Transition the model to the Production stage

# COMMAND ----------

### Transition a model version and retrieve details 
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production',
)
model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Fetch the latest model and run predictions

# COMMAND ----------

# model_name = "Test-Stage-Model_3053e3"
model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(new_model_version)

# COMMAND ----------

latest_version_info = client.get_latest_versions(model_name, stages=['Production'])
latest_stage_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_stage_version))

# COMMAND ----------

# Prepare train dataset
silver_train_pdf = silver_train_df.toPandas()
y_train = silver_train_pdf["open_flag"]
X_train =  silver_train_pdf.drop("open_flag", axis=1)

# Run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
silver_train_pdf["predicted"] = model.predict(X_train)
display(silver_train_pdf)

# COMMAND ----------

# Fetch model params
model = mlflow.sklearn.load_model(model_uri)
model.get_params()['classifier']

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Deploying a new Model with Infer Signature

# COMMAND ----------

from mlflow.models.signature import infer_signature

# COMMAND ----------

model_name = "Test-Stage-Model_3053e3"
input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))
 
with mlflow.start_run(run_name="LGBM Model") as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature
    )
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC Use the search functionality to grab the latest model version.

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(f"New model version: {new_model_version}")

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="This model version is a model with signiture."
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Put this new model version into Staging

# COMMAND ----------

### Transition a model version and retrieve details 
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Staging',
)

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since this model is now in staging, you can execute an automated CI/CD pipeline against it to test it before going into production. Once that is completed, you can push that model into production.

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production",
    archive_existing_versions=True # Archive old versions of this model
)

# COMMAND ----------

import os
import shutil

# COMMAND ----------

# Save models to DBFS
model_name = 'Test-Stage-Model_3053e3'
model_uri = "models:/{model_name}/production".format(model_name=model_name)
model = mlflow.sklearn.load_model(model_uri)

latest_version_info = client.get_latest_versions(model_name, stages=["production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))

modelpath = "/dbfs/mnt/tmg-stage-ml-artifacts/model-%s-%s" % (model_name, latest_production_version)
print(modelpath)
mlflow.sklearn.save_model(model, modelpath)

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

user_modelpath = '/dbfs/Users/{0}/ml-artifacts/model-{1}-{2}'.format(user, model_name, latest_production_version)
print(user_modelpath)
shutil.rmtree(user_modelpath, ignore_errors=True)
mlflow.sklearn.save_model(model, user_modelpath)

# COMMAND ----------

display(dbutils.fs.ls(user_modelpath.replace('/dbfs', 'dbfs:')))

# COMMAND ----------

shutil.rmtree('dbfs:/mnt/tmg-stage-ml-artifacts/model-Test-Stage-Model_3053e3-4/', ignore_errors=True)

# COMMAND ----------

dbutils.fs.mv('dbfs:/Users/lhuang@themeetgroup.com/ml-artifacts/model-Test-Stage-Model_3053e3-4', 'dbfs:/mnt/tmg-stage-ml-artifacts/model-Test-Stage-Model_3053e3-4', recurse=True)

# COMMAND ----------

display(dbutils.fs.ls(modelpath.replace('/dbfs', 'dbfs:')))

# COMMAND ----------

import boto3
import os
from pathlib import Path

s3 = boto3.resource('s3')

bucket = s3.Bucket('bucket')

key = 'tmg-stage-ml-artifacts/model-Test-Stage-Model_3053e3-4'
objs = list(bucket.objects.filter(Prefix=key))

for obj in objs:
    # print(obj.key)

    # remove the file name from the object key
    obj_path = os.path.dirname(obj.key)

    # create nested directory structure
    Path(obj_path).mkdir(parents=True, exist_ok=True)

    # save file with full path locally
    bucket.download_file(obj.key, obj.key)

# COMMAND ----------


