# Databricks notebook source
# MAGIC %md 
# MAGIC ### Initial Database Setup

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt')

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

# Paths for various Delta tables
# bronze_tbl_path = '/home/{}/push_data_exploration/bronze/'.format(user)
# silver_tbl_path = '/home/{}/push_data_exploration/silver/'.format(user)

# bronze_tbl_path = '/FileStore/shared_uploads/{}/push_data_exploration/bronze/'.format(user)

# COMMAND ----------

# Delete the old database and tables if needed (for demo purposes)
database_name = 'ml_push'
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))

# COMMAND ----------

# import shutil

# COMMAND ----------

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
# shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Creating Bronze Delta Table from Spark Dataframe

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml_push.s_tmg_notification;
# MAGIC CREATE EXTERNAL TABLE ml_push.s_tmg_notification(
# MAGIC   common__timestamp bigint, 
# MAGIC   common__subject string, 
# MAGIC   network_user_id string, 
# MAGIC   user_id string, 
# MAGIC   common__network string, 
# MAGIC   network string, 
# MAGIC   event_type string, 
# MAGIC   event_status string, 
# MAGIC   notification_type string, 
# MAGIC   notification_name string, 
# MAGIC   from_user_id string, 
# MAGIC   from_user_network string, 
# MAGIC   service_name string, 
# MAGIC   device_type string)
# MAGIC PARTITIONED BY ( 
# MAGIC   _processing_timestamp timestamp)
# MAGIC ROW FORMAT SERDE 
# MAGIC   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' 
# MAGIC STORED AS INPUTFORMAT 
# MAGIC   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' 
# MAGIC OUTPUTFORMAT 
# MAGIC   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
# MAGIC LOCATION
# MAGIC   '/mnt/tmg-stage-datalake/topics/s_tmg_notification/'

# COMMAND ----------

# MAGIC %sql
# MAGIC MSCK REPAIR TABLE  ml_push.s_tmg_notification;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ml_push.s_tmg_notification
# MAGIC limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Push historical data in last 7 days

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml_push.l7_push_meetme;
# MAGIC CREATE TABLE ml_push.l7_push_meetme AS
# MAGIC   SELECT 
# MAGIC   send_data.*,
# MAGIC   dayofweek(from_unixtime(send_data.send_ts, 'yyyy-MM-dd HH:mm:ss')) AS utc_day_of_week,
# MAGIC   hour(from_unixtime(send_data.send_ts)) AS utc_hour,
# MAGIC   case when open_data.network_user_id is not null then 1 else 0 end open_flag
# MAGIC   FROM
# MAGIC   (
# MAGIC     SELECT *, 
# MAGIC       common__timestamp as send_ts
# MAGIC     FROM ml_push.s_tmg_notification
# MAGIC     WHERE event_status = 'success' 
# MAGIC     AND network = 'meetme'
# MAGIC     AND event_type = 'send'
# MAGIC     AND FROM_UNIXTIME(common__timestamp) >= (CURRENT_TIMESTAMP - INTERVAL '7' DAY)
# MAGIC     AND notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC  ) send_data
# MAGIC LEFT JOIN
# MAGIC (
# MAGIC   SELECT network_user_id, common__timestamp as open_ts
# MAGIC   FROM ml_push.s_tmg_notification
# MAGIC   WHERE event_status = 'success' 
# MAGIC   AND network = 'meetme'
# MAGIC   AND event_type = 'open'
# MAGIC   AND FROM_UNIXTIME(common__timestamp) >= (CURRENT_TIMESTAMP - INTERVAL '7' DAY)
# MAGIC   AND notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC ) open_data
# MAGIC ON send_data.network_user_id = open_data.network_user_id 
# MAGIC AND send_data.send_ts < open_data.open_ts

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE EXTENDED ml_push.l7_push_meetme;

# COMMAND ----------

df = spark.sql("select * from ml_push.l7_push_meetme")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

def shape(data):
    rows, cols = data.count(), len(data.columns)
    shape = (rows, cols)
    return shape

shape(df)

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# Paths for various Delta tables
database_name = 'ml_push'
bronze_tbl_name = 'bronze_l7_push_meetme'
bronze_tbl_path = '/FileStore/shared_uploads/{}/push_data/bronze/'.format(user)

# COMMAND ----------

#To improve read performance when you load data back, Databricks recommends turning off compression when you save data loaded from binary files:
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

# Create a Delta Lake table from loaded 
df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

# COMMAND ----------

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path)
# shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)

# COMMAND ----------

# Create bronze table
spark.sql('DROP TABLE IF EXISTS {}'.format('ml_push.bronze_l7_push_meetme'))

_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,bronze_tbl_name,bronze_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

bronze_df = spark.table("ml_push.bronze_l7_push_meetme")

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

columns = [
 'network_user_id',
 'notification_type',
 'notification_name',
 'device_type',
 'utc_day_of_week',
 'utc_hour',
 'open_flag'   
]

# COMMAND ----------

def count_dist(df, field):
    return df.select(field).distinct().count()

# COMMAND ----------

bronze_df.select(columns).select([F.count(F.when(F.col(c).contains('None') |
                          F.col(c).contains('NULL') |
                        (F.col(c) == '') |
                        F.col(c).isNull() |
                        F.isnan(c), c
                 )).alias(c)
                    for c in columns]).show(vertical=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM ml_push.bronze_l7_push_meetme

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Reading from a Delta table is much faster than trying to load all the data from a directory! By saving the data in the Delta format, we can continue to process the data into a silver table, transforming it into a proper training set for our use case. This entails parsing out labels from the data paths and converting string labels into numerical values.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Send Traffics

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT notification_type, count(*)
# MAGIC FROM ml_push.bronze_l7_push_meetme
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

for c in columns:
    (bronze_df
    .groupBy(c)
    .count()
    .sort('count', ascending=False)
    .show())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Unique Users

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT notification_type, count(distinct network_user_id) AS user_cnt
# MAGIC FROM ml_push.bronze_l7_push_meetme
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

for c in columns:
    (bronze_df
    .groupBy(c)
    .agg(F.countDistinct('network_user_id').alias('user_count'))
    .sort('user_count', ascending=False)
    .show())

# COMMAND ----------

# MAGIC %md
# MAGIC ####Open Counts

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT notification_type, sum(open_flag) AS opens
# MAGIC FROM ml_push.bronze_l7_push_meetme
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

for c in columns:
    (bronze_df
    .groupBy(c)
    .agg(F.sum('open_flag').alias('opens'))
    .sort('opens', ascending=False)
    .show())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Open Rates

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT notification_type, mean(open_flag) AS opens
# MAGIC FROM ml_push.bronze_l7_push_meetme
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

for c in columns:
    (bronze_df
    .groupBy(c)
    .agg(F.mean('open_flag').alias('open rates'))
    .sort('open rates', ascending=False)
    .show())

# COMMAND ----------


