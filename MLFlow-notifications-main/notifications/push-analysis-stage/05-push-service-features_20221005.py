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

bronze_df = bronze_df.withColumn("n_visit", F.row_number()\
                           .over(Window.partitionBy("network_user_id").orderBy(F.desc("common__timestamp")))\
                           .cast(IntegerType()))

# COMMAND ----------


