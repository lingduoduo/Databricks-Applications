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

# Create silver table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_train_tbl_name,silver_train_tbl_path))

_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_val_tbl_name,silver_val_tbl_path))

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


