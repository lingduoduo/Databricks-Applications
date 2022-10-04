# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Readin Training Data

# COMMAND ----------

# import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import max as sparkMax
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder, Normalizer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from sklearn.metrics import roc_curve, auc

import time
import datetime
import numpy as np
import pandas as pd

# COMMAND ----------

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("push Project") \
    .getOrCreate()
    
# fetch SparkContext context
sc = spark.sparkContext

# check Spark session
spark.sparkContext.getConf().getAll()

# COMMAND ----------

spark

# COMMAND ----------

df = spark.sql("select * from ml_push.daily_push_meetme")

# COMMAND ----------

def shape(data):
    rows, cols = data.count(), len(data.columns)
    shape = (rows, cols)
    return shape

# COMMAND ----------

shape(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Feature Engineering

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM ml_push.daily_push_meetme
# MAGIC LIMIT 3

# COMMAND ----------

df = df.withColumn('open_flag', F.col('open_flag').cast(BooleanType()))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

def category_cleaner(value):
    if value is None:
        return 'missing'
    else:
        return value

# COMMAND ----------

def process_data(dataframe):
    """
    Function to wrap specific processing for data tables
    Input and output is a pyspark.pandas dataframe
    """
    # concat user_network_id
    dataframe['from_user_id'] = dataframe['from_user_network'] + ':user:' + dataframe['from_user_id']
    dataframe['country_tier_cat'] = dataframe['country_tier'].astype('category')

    categorical_cols = ['gender', 'age_bucket', 'country_tier', 'device_type', 'from_user_id']

    # categorical column cleansing
    for column in categorical_cols:
        dataframe[column] = dataframe[column].apply(lambda value: category_cleaner(value))

    # Drop columns
    dataframe = dataframe.drop(['common__timestamp', 'network_user_id', 'event_type', 'event_status', 'notification_type', 'notification_name', 'from_user_network', 'send_ts'], axis = 1)

    return dataframe

# COMMAND ----------

df_pdf = df.to_pandas_on_spark()

# COMMAND ----------

train_pdf = process_data(df_pdf)

# COMMAND ----------

train_pdf.head(3)

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# Paths for various Delta tables
train_tbl_path = '/home/{}/ml_push/train_pdf/'.format(user)
train_tbl_name = 'train_pdf'

# COMMAND ----------

train_pdf.to_delta(train_tbl_path)
spark.sql('''
             CREATE TABLE {0}
             USING DELTA 
             LOCATION '{1}'
          '''.format(train_tbl_name, train_tbl_path)
         )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Readin Inference Data

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/e6cf3c38_c01b_4646_ba35_01df7eec7ac5.csv")

# COMMAND ----------

inference_df  = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/e6cf3c38_c01b_4646_ba35_01df7eec7ac5.csv")
           )
 
display(inference_df)

# COMMAND ----------

shape(inference_df)

# COMMAND ----------

# Save resulting standardized Dataframe as a Table
(inference_df
         .write
         .format("delta")
         .mode("overwrite")
         .saveAsTable("inference_df")
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE ml_push.daily_push_meetme_inference AS
# MAGIC SELECT 
# MAGIC send_data.*,
# MAGIC case when open_data.network_user_id is not null then 1 else 0 end open_flag
# MAGIC FROM
# MAGIC (
# MAGIC SELECT *, common__timestamp as send_ts
# MAGIC FROM inference_df
# MAGIC WHERE event_status = 'success' and event_type = 'send'
# MAGIC ) send_data
# MAGIC LEFT JOIN
# MAGIC (
# MAGIC SELECT network_user_id, common__timestamp as open_ts
# MAGIC FROM inference_df
# MAGIC WHERE event_status = 'success' and event_type = 'open'
# MAGIC ) open_data
# MAGIC on send_data.network_user_id = open_data.network_user_id and send_data.send_ts < open_data.open_ts

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select * from ml_push.daily_push_meetme_inference limit 10;

# COMMAND ----------

inference_df = spark.sql("select * from ml_push.daily_push_meetme_inference")
inference_df = inference_df.withColumn('open_flag', F.col('open_flag').cast(BooleanType()))

# COMMAND ----------

# Paths for various Delta tables
inference_tbl_path = '/home/{}/ml_push/inference_pdf_new/'.format(user)
inference_tbl_name = 'inference_pdf_new'

# COMMAND ----------

inference_pdf =inference_df.to_pandas_on_spark()
inference_pdf = process_data(inference_pdf)

inference_pdf.to_delta(inference_tbl_path, mode='overwrite')
# inference_pdf.write.format('delta').mode('overwrite').save(inference_tbl_path )
 
spark.sql('''
             CREATE TABLE {0}
             USING DELTA 
             LOCATION '{1}'
          '''.format(inference_tbl_name, inference_tbl_path)
         )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### AutoML

# COMMAND ----------

import databricks.automl
 
summary = databricks.automl.classify(train_pdf, target_col='open_flag', primary_metric="f1", data_dir='dbfs:/automl/ml_push', timeout_minutes=30)

# COMMAND ----------


