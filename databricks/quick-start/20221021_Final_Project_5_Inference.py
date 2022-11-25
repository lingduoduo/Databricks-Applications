# Databricks notebook source
# MAGIC %md
# MAGIC #Inference for Final Project Group 5

# COMMAND ----------

#%pip install spark-nlp==3.3.3 wordcloud contractions gensim pyldavis==3.2.0

# COMMAND ----------

databaseName_widget = dbutils.widgets.text("database_name", "group5_finalproject", "Database Name")
databaseName = dbutils.widgets.get("database_name") 

spark.sql(f"use {databaseName}")

# COMMAND ----------

import pandas as pd
import sparknlp
import mlflow
import mlflow.spark
import tempfile
import pickle
import time
import traceback
import pyspark
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

from pyspark.sql.functions import *
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

# Get data in twitter table
sdf_twitter = spark.sql("SELECT * FROM crypto_df_tf")

# Select model for prediction
model_name = "best_model_sentimentDL"

databaseName_widget = dbutils.widgets.text("database_name", "group5_finalproject", "Database Name")
databaseName = dbutils.widgets.get("database_name") 

spark.sql(f"use {databaseName}")

# Assign MLflow client and get model for prediction from production stage
client = mlflow.tracking.MlflowClient()
latest_prod_model_detail = client.get_latest_versions(model_name, stages=["Production"])[0]
latest_prod_model =  mlflow.spark.load_model(f"runs:/{latest_prod_model_detail.run_id}/sentimentDL_model")

# Make prediction
sdf_preds = latest_prod_model.transform(sdf_twitter)

# Unnest result column
sdf_preds_trans = sdf_preds.select("id", "text", "QUERY", "yfinance_ticker", "user_name", "result_type", "favorite_count", "followers_count", "retweet_count", "created_at","day","hour","minute", F.col("class.result").getItem(0))

# Rename prediction column
sdf_preds_trans = sdf_preds_trans.withColumnRenamed(sdf_preds_trans.columns[-1], "sentiment")

# Write dataframe to a table
sdf_preds_trans.write.format('delta').mode("overwrite").saveAsTable("twitter_silver")

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pyspark.sql.functions import *

# Import data from gold table and transform it further
sdf_corr = spark.table('gold')
sdf_corr = sdf_corr \
            .withColumn('sentiment_encoded', regexp_replace('sentiment', 'positive', '1')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded', 'negative', '-1')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded','neutral', '0')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded','null', '0'))

df_corr = sdf_corr.toPandas()
df_corr = df_corr.dropna(axis = 0, how ='any')
df_corr['sentiment_encoded'] = df_corr['sentiment_encoded'].astype(int)
df_corr["QUERY"] = df_corr["QUERY"].str.lower()

# Group data by cryptocurrency and date, and add up the total sentiment score for each crptopcurreny in a 15-minute increments
df_corr_grouped = df_corr.groupby(['QUERY', 'date']).sum('sentiment_endcoded')
x = df_corr_grouped [['sentiment_encoded']]
y = df_corr_grouped ['delta'] # % change in closing price

# Select model for prediction
model_name = "best_model_corr"

# Assign MLflow client and get model for prediction from production stage
client = mlflow.tracking.MlflowClient()
latest_prod_model_detail = client.get_latest_versions(model_name, stages=["Production"])[0]
latest_prod_model =  mlflow.sklearn.load_model(f"runs:/{latest_prod_model_detail.run_id}/corr_model")

# Make prediction
preds = latest_prod_model.predict(x)

# Transform predictions in numpy to pandas DataFrame and then spark DataFrame, write spark DataFrame to a table
df_preds = pd.DataFrame(preds, columns=["preds"])
sdf_corr = spark.createDataFrame(df_preds, ["preds"])
sdf_corr.write.format('delta').mode("overwrite").saveAsTable("corr_test")
