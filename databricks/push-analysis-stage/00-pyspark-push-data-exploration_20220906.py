# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Initial Data Table Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```
# MAGIC select
# MAGIC   n.common__timestamp,
# MAGIC   n.network_user_id,
# MAGIC   n.event_type,
# MAGIC   n.event_status,
# MAGIC   n.notification_type,
# MAGIC   n.notification_name,
# MAGIC   n.device_type,
# MAGIC   n.from_user_id,
# MAGIC   n.from_user_network,
# MAGIC         mups.gender,
# MAGIC         mups.age_bucket,
# MAGIC         mups.country_tier
# MAGIC from tmg.s_tmg_notification n
# MAGIC left outer join tmg.m_user_profile_snapshot mups
# MAGIC   on n.user_id = mups.user_id
# MAGIC  and n.network = mups.user_network
# MAGIC where n._partition_date >= date_add('day', -1, current_date)
# MAGIC and n._partition_date < current_date
# MAGIC and n.notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC and n.network = 'meetme'
# MAGIC group by 1,2,3,4,5,6,7,8,9,10, 11, 12;
# MAGIC ```

# COMMAND ----------

# display(dbutils.fs.ls("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/0bdc7eed_e83e_445d_88e0_4da185392a3f.csv"))
display(dbutils.fs.ls("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/dae634f4_63d7_44dc_b947_535057f5ec2b.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Exploratory Data Analysis

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
import seaborn as sns
from matplotlib import pyplot as plt

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

df = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/dae634f4_63d7_44dc_b947_535057f5ec2b.csv")
           )
 
display(df)

# COMMAND ----------

def shape(data):
    rows, cols = data.count(), len(data.columns)
    shape = (rows, cols)
    return shape

# COMMAND ----------

shape(df)

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Missing Data

# COMMAND ----------

df.select([F.count(F.when(F.col(c).contains('None') |
                          F.col(c).contains('NULL') |
                        (F.col(c) == '') |
                        F.col(c).isNull() |
                        F.isnan(c), c
                 )).alias(c)
                    for c in ['from_user_id', 'from_user_network', 'gender', 'age_bucket', 'country_tier']]).show(vertical=True)

# COMMAND ----------

df.filter(df.from_user_id.isNull()).show(10)

# COMMAND ----------

df.filter(df.gender.isNull()).show(10)

# COMMAND ----------

# Save resulting standardized Dataframe as a Table
(df
         .write
         .format("delta")
         .mode("overwrite")
         .saveAsTable("df")
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS ml_push.daily_push_meetme;

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE ml_push.daily_push_meetme AS
# MAGIC SELECT 
# MAGIC send_data.*,
# MAGIC case when open_data.network_user_id is not null then 1 else 0 end open_flag
# MAGIC FROM
# MAGIC (
# MAGIC SELECT *, common__timestamp as send_ts
# MAGIC FROM df
# MAGIC WHERE event_status = 'success' and event_type = 'send'
# MAGIC ) send_data
# MAGIC LEFT JOIN
# MAGIC (
# MAGIC SELECT network_user_id, common__timestamp as open_ts
# MAGIC FROM df
# MAGIC WHERE event_status = 'success' and event_type = 'open'
# MAGIC ) open_data
# MAGIC on send_data.network_user_id = open_data.network_user_id and send_data.send_ts < open_data.open_ts

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select * from ml_push.daily_push_meetme limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(*), sum(open_flag), sum(open_flag)/count(*) from ml_push.daily_push_meetme;

# COMMAND ----------

df = spark.sql("select * from ml_push.daily_push_meetme")

# COMMAND ----------

df.show(n=1, vertical=True)

# COMMAND ----------

df.take(1)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data Visualization

# COMMAND ----------

print("total records: {}".format(df.count()))

# COMMAND ----------

df.select('network_user_id').distinct().count()

# COMMAND ----------

df.select('from_user_id').distinct().count()

# COMMAND ----------

df.groupBy('device_type').count().sort("count", ascending=False).show()

# COMMAND ----------

df.groupby('network_user_id').count().sort('count', ascending=False).show(10)

# COMMAND ----------

df.groupby('notification_name').count().sort('count', ascending=False).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Open Rates by Device Type

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select device_type, count(*), sum(open_flag), sum(open_flag)/count(*) from ml_push.daily_push_meetme group by 1;

# COMMAND ----------

# convert our raw spark distributed dataframe into a distributed pandas dataframe
raw_df_pdf = df.to_pandas_on_spark()
 
# perform the same aggregation we did in SQL using familiar Pandas syntax
ctr_device = raw_df_pdf[['open_flag', 'device_type']].groupby('device_type').mean().reset_index()

ctr_device.plot(kind='bar', x='device_type', y='open_flag')

# COMMAND ----------

df.groupBy("open_flag", "device_type").agg(F.countDistinct("network_user_id")
                                             .alias('user_count')).sort('user_count', ascending=False).show()

# COMMAND ----------

df_dedupe = df.withColumn("last_record", F.row_number()\
                           .over(Window.partitionBy("network_user_id").orderBy(F.desc("common__timestamp")))\
                           .cast(IntegerType())).where("last_record == 1").drop("last_record");

# convert our raw spark distributed dataframe into a distributed pandas dataframe
df_dedupe_pdf = df_dedupe.to_pandas_on_spark()
 
# perform the same aggregation we did in SQL using familiar Pandas syntax
ctr_device = df_dedupe_pdf[['open_flag', 'device_type']].groupby('device_type').mean().reset_index()

ctr_device.plot(kind='bar', x='device_type', y='open_flag')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Open Rates by Demo

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select gender, age_bucket,  count(*), sum(open_flag), sum(open_flag)/count(*) from ml_push.daily_push_meetme group by 1, 2 

# COMMAND ----------

# perform the same aggregation we did in SQL using familiar Pandas syntax
ctr_demo = raw_df_pdf[['open_flag', 'age_bucket', 'gender']].groupby(['age_bucket', 'gender']).mean().reset_index()

ctr_demo.plot(kind='bar', x='age_bucket', y='open_flag', color = 'gender',  barmode='group')

# COMMAND ----------

ctr_demo = df_dedupe_pdf[['open_flag', 'age_bucket', 'gender']].groupby(['age_bucket', 'gender']).mean().reset_index()

ctr_demo.plot(kind='bar', x='age_bucket', y='open_flag', color = 'gender',  barmode='group')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Open Rates by Geo

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select country_tier,  count(*), sum(open_flag), sum(open_flag)/count(*) from ml_push.daily_push_meetme group by 1

# COMMAND ----------

ctr_geo = raw_df_pdf[['open_flag', 'country_tier']].groupby('country_tier').mean().reset_index()

ctr_geo.plot(kind='bar', x='country_tier', y='open_flag')

# COMMAND ----------

ctr_geo = df_dedupe_pdf[['open_flag', 'country_tier']].groupby('country_tier').mean().reset_index()

ctr_geo.plot(kind='bar', x='country_tier', y='open_flag')
