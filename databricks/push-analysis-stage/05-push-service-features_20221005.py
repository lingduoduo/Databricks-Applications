# Databricks notebook source
# MAGIC %md 
# MAGIC ### Load Data

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

display(bronze_df)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
from pyspark.sql.window import Window

# COMMAND ----------

most_popular_broadcasters = bronze_df.withColumn("popular_broadcaster"
    .groupBy("from_user_id")\
    .count()\
    .sort("count", ascending=False)

# COMMAND ----------

user = Window().partitionBy('user_id')

user_log = bronze_df.withColumn('total_push', F.count('*').over(user))\
    .withColumn('total_opens', F.sum('open_flag').over(user))\
    .withColumn('min_dt',  F.from_unixtime((F.min('send_ts').over(user))/1000).cast(DateType()))\
    .withColumn('max_dt', F.from_unixtime((F.max('send_ts').over(user))/1000).cast(DateType()))\
    .withColumn('max_unix', F.max('send_ts').over(user))\
    .select(*['user_id', 'total_push', 'open_flag', 'min_dt', 'max_dt', 'max_unix']).distinct()

# COMMAND ----------


