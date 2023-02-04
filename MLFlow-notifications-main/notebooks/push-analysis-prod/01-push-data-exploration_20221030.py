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

# MAGIC %md
# MAGIC 
# MAGIC ### Readin Push data

# COMMAND ----------

# MAGIC %sql
# MAGIC select CURRENT_TIMESTAMP - INTERVAL '1' HOUR

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW PARTITIONS ml_push.l7_push_meetme_source_partitioned;

# COMMAND ----------

# MAGIC %sql select calculated_time from ml_push.l7_push_meetme_source_partitioned group by 1 order by 1 desc limit 24;

# COMMAND ----------

# MAGIC %sql select max(calculated_time) from ml_push.l7_push_meetme_source_partitioned;

# COMMAND ----------

# df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time > CURRENT_TIMESTAMP - INTERVAL '1' HOUR")
# df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time = (select max(calculated_time) from ml_push.l7_push_meetme_source_partitioned)")
# df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time = '2022-10-31 19:20:06.243'")
df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time = '2022-11-04 19:19:55.364'")

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

# MAGIC %md
# MAGIC 
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df = df.withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id')))

# COMMAND ----------

columns = [
 'network_user_id',
 'broadcaster_id',
 'notification_type',
 'notification_name',
 'device_type',
 'open_flag'   
]

# COMMAND ----------

def count_dist(df, field):
    return df.select(field).distinct().count()

# COMMAND ----------

df.select(columns).select([F.count(F.when(F.col(c).contains('None') |
                          F.col(c).contains('NULL') |
                        (F.col(c) == '') |
                        F.col(c).isNull() |
                        F.isnan(c), c
                 )).alias(c)
                    for c in columns]).show(vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Send Traffics

# COMMAND ----------

for c in columns:
    (df
    .groupBy(c)
    .count()
    .sort('count', ascending=False)
    .show())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Unique Users

# COMMAND ----------

for c in columns:
    (df
    .groupBy(c)
    .agg(F.countDistinct('network_user_id').alias('user_count'))
    .sort('user_count', ascending=False)
    .show())

# COMMAND ----------

# MAGIC %md
# MAGIC ####Open Counts

# COMMAND ----------

for c in columns:
    (df
    .groupBy(c)
    .agg(F.sum('open_flag').alias('opens'))
    .sort('opens', ascending=False)
    .show())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Open Rates

# COMMAND ----------

for c in columns:
    (df
    .groupBy(c)
    .agg(F.mean('open_flag').alias('open rates'))
    .sort('open rates', ascending=False)
    .show())

# COMMAND ----------


