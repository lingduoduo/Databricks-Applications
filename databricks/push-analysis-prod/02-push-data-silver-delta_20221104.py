# Databricks notebook source
# MAGIC %md 
# MAGIC ### Setup Bronze Table Connection

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt')

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

bronze_df = spark.sql("select *, dayofweek(from_unixtime(send_ts, 'yyyy-MM-dd HH:mm:ss')) AS utc_day_of_week, hour(from_unixtime(send_ts)) AS utc_hour from ml_push.l7_push_meetme_source_partitioned where calculated_time = '2022-11-04 19:19:55.364'")

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
    'utc_day_of_week',
    'utc_hour',
    'broadcaster_id'
]

# COMMAND ----------

df_train, df_val = silver_df.select(columns).randomSplit([0.8, 0.2], seed=12345)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-prod-ml-outputs/')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Save our `train` and `val` datasets into delta lake tables for future use.

# COMMAND ----------

import time
ts = time.time()

# COMMAND ----------

database_name = 'ml_push'

silver_train_tbl_path = '/mnt/tmg-prod-ml-outputs/push_data/silver_train/{}'.format(ts)
silver_val_tbl_path = '/mnt/tmg-prod-ml-outputs/push_data/silver_train/{}'.format(ts)

silver_train_tbl_name = 'silver_l7_push_meetme_train'
silver_val_tbl_name = 'silver_l7_push_meetme_val'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Creating Silver Delta Table from Spark Dataframe

# COMMAND ----------

# import shutil
# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
# shutil.rmtree('/dbfs'+silver_train_tbl_path, ignore_errors=True)
# shutil.rmtree('/dbfs'+silver_val_tbl_path, ignore_errors=True)

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

summary = databricks.automl.classify(silver_train_df, target_col='open_flag', primary_metric="f1", data_dir='dbfs:/automl/ml_push', timeout_minutes=60)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Register Model & Promote to Production

# COMMAND ----------

import mlflow
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# COMMAND ----------

model_uri = summary.best_trial.model_path

# COMMAND ----------

# Assign model name, i.e. best model coming out of pipeline
model_name = "Test-Stage-Model"

# COMMAND ----------

# Register models in the Model Registry
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
        name=model_name,
        version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

# COMMAND ----------

wait_until_ready(model_details.name, model_details.version)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-stage-ml-artifacts')

# COMMAND ----------

dbutils.fs.unmount("/mnt/tmg-stage-ml-artifacts")

# COMMAND ----------

aws_bucket_name = "tmg-stage-ml-artifacts"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-stage-ml-artifacts')

# COMMAND ----------

# Save models to DBFS
model_uri = "models:/{model_name}/production".format(model_name=model_name)
model = mlflow.sklearn.load_model(model_uri)
# modelpath = "/dbfs/mnt/tmg-stage-ml-outputs/model-%s-%s" % (model_details.name, model_details.version)
modelpath = "/dbfs/mnt/tmg-stage-ml-artifacts/model-%s-%s" % ("newtest", 3)
print(modelpath)
mlflow.sklearn.save_model(model, modelpath)

# COMMAND ----------

# Fetch model params
model.get_params()['classifier']

# COMMAND ----------

### Add model and model version descriptions.
client = MlflowClient()
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

### Transition a model version and retrieve details 
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='production',
)
model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

latest_version_info = client.get_latest_versions(model_name, stages=["production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))

# COMMAND ----------

### Load model using registered model name and version
model_version_uri = f"models:/{model_name}/{latest_production_version}".format(model_name=model_name, latest_production_version=latest_production_version)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_latest_version = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

### Load model using production model
model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model_production = mlflow.pyfunc.load_model(model_production_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Inference

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

model_name = "Test-Stage-Model"
model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(new_model_version)

# COMMAND ----------

latest_version_info = client.get_latest_versions(model_name, stages=['Production'])
latest_stage_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_stage_version))

# COMMAND ----------

### Load model using production model
model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model = mlflow.pyfunc.load_model(model_production_uri)

# COMMAND ----------

# Prepare train dataset
silver_train_pdf = silver_train_df.toPandas()
y_train = silver_train_pdf["open_flag"]
X_train =  silver_train_pdf.drop("open_flag", axis=1)

# COMMAND ----------

# Run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
silver_train_pdf["predicted"] = model.predict(X_train)
display(silver_train_pdf)

# COMMAND ----------

# Prepare test dataset
silver_val_pdf = silver_val_df.toPandas()
y_val = silver_val_pdf["open_flag"]
X_val =  silver_val_pdf.drop("open_flag", axis=1)

# COMMAND ----------

# Run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
silver_val_pdf["predicted"] = model.predict(X_val)
display(silver_val_pdf)

# COMMAND ----------


