# Databricks notebook source
from pyspark.sql.functions import *

import urllib

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
