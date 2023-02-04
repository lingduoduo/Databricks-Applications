# Databricks notebook source
# MAGIC %md 
# MAGIC ### Setup Bronze Table Connection

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

DATA_OUTPUT_PATH = '/mnt/tmg-prod-datalake-outputs'
dbutils.fs.ls(DATA_OUTPUT_PATH)

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

# import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
from pyspark.sql.window import Window

# COMMAND ----------

# bronze_df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time > CURRENT_TIMESTAMP - INTERVAL '1' HOUR")
# bronze_df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time = (select max(calculated_time) from ml_push.l7_push_meetme_source_partitioned)")
# bronze_df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time = '2022-10-31 19:20:06.243'")
# bronze_df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time = '2022-11-04 19:19:55.364'")

# bronze_df = spark.sql("select * from ml_push.push_demographics")
# bronze_df = spark.sql("select * from ml_push.temp_push_demographics_v2")
bronze_df = spark.sql("select * from ml_push.push_demographics_v4")

# COMMAND ----------

bronze_df = bronze_df.withColumn('cal_utc_day_of_week', F.dayofweek(F.from_unixtime(F.col('send_ts')/1000).cast(DateType()))) \
    .withColumn('cal_utc_hour', F.hour(F.from_unixtime(F.col('send_ts')/1000))) \
    .withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id'))) \
    .withColumn('lang', F.substring(F.col('locale'), 1, 2)) \
    .withColumn('from_user_lang', F.substring(F.col('from_user_locale'), 1, 2))

# COMMAND ----------

bronze_df.count()

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

rows = bronze_df.groupBy('network_user_id').count()
rows.count()

# COMMAND ----------

rows = bronze_df.groupBy('broadcaster_id').count()
rows.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Sampling Methods

# COMMAND ----------

# 5. Proportional sampling using broadcaster_id
bronze_df_pos = bronze_df.filter(bronze_df.open_flag == 1)
bronze_df_neg = bronze_df.filter(bronze_df.open_flag == 0)               
tot = bronze_df_neg.count()
bronze_df_ratios = bronze_df_neg.withColumn('ratio', F.lit(100/tot)).groupBy('broadcaster_id').agg(F.sum('ratio').alias('ratio'))
d = {r['broadcaster_id']: min(1, r['ratio']) for r in bronze_df_ratios.collect()}
bronze_df_sample = bronze_df_neg.sampleBy('broadcaster_id', d, seed=28).union(bronze_df_pos)

# COMMAND ----------

bronze_df_sample.count()

# COMMAND ----------

bronze_df_sample.groupBy('open_flag').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Feature Engineering

# COMMAND ----------

silver_df = bronze_df_sample.withColumn('open_flag', F.col('open_flag').cast(BooleanType()))

# COMMAND ----------

categorical_features = [
    'device_type',
    'gender',
    'age',
    'country',
    'lang',
    'locale',
    'from_user_age',
    'from_user_gender',
    'from_user_country',
    'from_user_lang',
    'from_user_locale',
    'cal_utc_hour',
    'cal_utc_day_of_week'
]

# COMMAND ----------

for col_name in categorical_features:
    silver_df = silver_df.withColumn(col_name, F.col(col_name).cast(StringType()))

# COMMAND ----------

silver_df = silver_df.fillna(value='missing', subset=categorical_features)

# COMMAND ----------

numeric_features = [
    'broadcast_search_count_24h',
    'search_user_count_24h',
    'search_match_count_24h',
    'view_count_24h',
    'view_end_count_24h',
    'gift_count_24h',
]

# COMMAND ----------

columns = ['open_flag'] + categorical_features + numeric_features

# COMMAND ----------

df_train, df_val = silver_df.select(columns).randomSplit([0.8, 0.2], seed=12345)

# COMMAND ----------

display(df_train)

# COMMAND ----------

import databricks.automl
summary = databricks.automl.classify(df_train, target_col='open_flag', primary_metric="f1", data_dir='dbfs:/automl/ml_push', timeout_minutes=120)

# COMMAND ----------


