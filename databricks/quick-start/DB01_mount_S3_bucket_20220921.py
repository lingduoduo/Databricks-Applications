# Databricks notebook source
from pyspark.sql.functions import *

import urllib

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Check FileStore Contents

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/tables"))

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/tables/65cb05a3_e45a_4a15_915b_90cf082dc203.csv"))

# COMMAND ----------

# Define file type
file_type = "csv"
# Whether the file has a header
first_row_is_header = "true"
# Delimiter used in the file
delimiter = ","

# Read the CSV file to spark dataframe
df = (spark.read.format(file_type)
.option("header", first_row_is_header)
.option("sep", delimiter)
.load("/FileStore/tables/65cb05a3_e45a_4a15_915b_90cf082dc203.csv"))

# COMMAND ----------

df = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/0bdc7eed_e83e_445d_88e0_4da185392a3f.csv")
           )
 
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Root dbfs:/

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/'))

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/databricks-datasets'))

# COMMAND ----------

# MAGIC %fs
# MAGIC 
# MAGIC ls /tmp

# COMMAND ----------

display(dbutils.fs.ls('/tmp/'))

# COMMAND ----------

import os

# COMMAND ----------

os.listdir('/dbfs/tmp')

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/tmp/

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Local filesystem on driver node

# COMMAND ----------

# MAGIC %fs
# MAGIC 
# MAGIC ls file:/tmp

# COMMAND ----------

display(dbutils.fs.ls('file:/tmp/'))

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /tmp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Mount S3 bucket

# COMMAND ----------

# https://s3.console.aws.amazon.com/s3/buckets/tmg-prod-ml-outputs?region=us-east-1

# COMMAND ----------

dbutils.fs.ls("s3a://tmg-prod-ml-artifacts")

# COMMAND ----------

aws_bucket_name = "tmg-prod-ml-outputs"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt/')

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-stage-datalake/')

# COMMAND ----------

aws_bucket_name = "tmg-stage-ml-outputs"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-stage-ml-outputs/')

# COMMAND ----------

aws_bucket_name = "tmg-stage-ml-artifacts"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-stage-ml-artifacts/')

# COMMAND ----------

aws_bucket_name = "tmg-prod-datalake"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-prod-datalake/')

# COMMAND ----------

aws_bucket_name = "tmg-prod-datalake-outputs"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-prod-datalake-outputs/')

# COMMAND ----------

aws_bucket_name = "tmg-prod-datalake-protected"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-prod-datalake-protected/')

# COMMAND ----------

aws_bucket_name = "tmg-prod-ml-artifacts"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-prod-ml-artifacts/')

# COMMAND ----------

aws_bucket_name = "tmg-prod-ml-outputs/"
dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % aws_bucket_name)

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-prod-ml-outputs')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Read a dataset in CSV format from S3 to Databricks

# COMMAND ----------

file_location = "/mnt/crypto-price-prediction/g-research-crypto-forecasting/crypto_100k_records.csv"
file_location = "/FileStore/tables/65cb05a3_e45a_4a15_915b_90cf082dc203.csv"
file_type = "csv"

# Define file type
file_type = "csv"
# Whether the file has a header
first_row_is_header = "true"
# Delimiter used in the file
delimiter = ","

# Read the CSV file to spark dataframe
df = (spark.read.format(file_type)
.option("header", first_row_is_header)
.option("sep", delimiter)
.load("/FileStore/tables/65cb05a3_e45a_4a15_915b_90cf082dc203.csv"))


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Save Spark Dataframe As Table

# COMMAND ----------

# Allow creating table using non-emply location 
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
 
# Save table
df.write.format("parquet").saveAsTable('train_test_ling')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Save Spark Dataframe To S3 Bucket

# COMMAND ----------

# Save to the mounted S3 bucket
df.write.save(f'/mnt/tmg-stage-ml-outputs/train_test_ling', format='csv')
 
# Check if the file was saved successfuly
display(dbutils.fs.ls("/mnt/tmg-stage-ml-outputs/train_test_ling"))

# COMMAND ----------

# Remove the file if it was saved before
dbutils.fs.rm('/mnt/tmg-stage-ml-outputs/train_test_ling', True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Unmount S3 Bucket

# COMMAND ----------

# Unmount S3 bucket
dbutils.fs.unmount("/mnt/tmg-stage-ml-outputs")
