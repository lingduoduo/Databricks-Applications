# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Managed vs. Unmanaged Tables

# COMMAND ----------

# MAGIC %md
# MAGIC In step 1, let's understand the difference between managed and external tables.
# MAGIC * Managed Tables
# MAGIC   * Data management: Spark manages both the metadata and the data
# MAGIC   * Data location: Data is saved in the Spark SQL warehouse directory `/user/hive/warehouse`. Metadata is saved in a meta-store of relational entities.
# MAGIC   * Data deletion: The metadata and the data will be deleted after deleting the table.
# MAGIC * Unmanaged/External Tables
# MAGIC   * Data management: Spark manages only the metadata, and the data itself is not controlled by spark. 
# MAGIC   * Data location: Source data location is required to create a table.
# MAGIC   * Data deletion: Only the metadata will be deleted. The tables saved in the external location.

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Mount S3 Bucket And Read CSV To Spark Dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC We first need to import libraries. `pyspark.sql.functions` has the functions for pyspark. `urllib` is the package for handling urls.

# COMMAND ----------

# pyspark functions
from pyspark.sql.functions import *

# URL processing
import urllib

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let’s read the csv file with AWS keys to Databricks. We specify the file type to be csv, indicate that the file has first row as the header and comma as the delimiter. Then the path of the csv file was passed in to load the file.

# COMMAND ----------

# Define file type
file_type = "csv"

# Whether the file has a header
first_row_is_header = "true"

# Delimiter used in the file
delimiter = ","

# Read the CSV file to spark dataframe
aws_keys_df = spark.read.format(file_type)\
.option("header", first_row_is_header)\
.option("sep", delimiter)\
.load("/FileStore/tables/65cb05a3_e45a_4a15_915b_90cf082dc203.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC After getting the permissions, it’s time to mount the S3 bucket! We can mount the bucket by passing in the S3 url and the desired mount name to `dbutils.fs.mount()`. It returns `Ture` if the bucket is mounted successfully.

# COMMAND ----------

# MAGIC %md
# MAGIC After checking the contents in the bucket using `%fs ls`, we can see that there are two folders in the bucket, data and output.

# COMMAND ----------

dbutils.fs.ls('/mnt/tmg-stage-ml-outputs/')

# COMMAND ----------

# MAGIC %md
# MAGIC The folder `data` has the dataset we need for this tutorial.

# COMMAND ----------

# MAGIC %fs ls "/mnt/demo4tutorial/data"

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's read the dataset from S3 bucket to spark dataframe. We set the delimiter to be a comma, indicate that the first row is the header, and ask spark to infer the schema.

# COMMAND ----------

# File location and type
file_location = "/mnt/demo4tutorial/data/crypto_100k_records.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Take a look at the data
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, we can use `.csv` to read CSV files.

# COMMAND ----------

# Use .csv to import CSV file
df = spark.read \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .csv(file_location)

# Take a look at the data
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC To manually define schema when reading in the data, we can use `StructType` to assign data types to columns.

# COMMAND ----------

# Import libraries
from pyspark.sql.types import LongType, StringType, FloatType, DoubleType, DecimalType, StructType, StructField

# User-defined schema
userDefinedSchema = StructType([
  StructField("timestamp", LongType(), True), # LongType: Represents 8-byte signed integer numbers. The range of numbers is from # -9223372036854775808 to 9223372036854775807.
  StructField("Asset_id", StringType(), True),
  StructField("Count", FloatType(), True),   # FloatType: Represents 4-byte single-precision floating point numbers.
  StructField("Open", FloatType(), True),   
  StructField("High", DoubleType(), True),   # DoubleType: Represents 8-byte double-precision floating point numbers.
  StructField("Low", DoubleType(), True),       
  StructField("Close", DoubleType(), True),   
  StructField("Volume", DoubleType(), True),     
  StructField("VWAP", DoubleType(), True),   
  StructField("Target", DoubleType(), True)   
])

# Create spark dataframe
df = spark.read.format(file_type) \
  .schema(userDefinedSchema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Take a look at the data
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC An alternative way to `StructType` is using a DDL formatted string.

# COMMAND ----------

# Define DDL schema
DDLSchema = 'timestamp long, Asset_id string, Count float, Open float, High double, Low double, Close double, Volume double, VWAP double, Target double'

# Create spark dataframe
df = spark.read.format(file_type) \
  .schema(DDLSchema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Take a look at the data
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Create Database In Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC In step 3, we will create a new database in Databricks. The tables will be created and saved in the new database.
# MAGIC Using the sql command `CREATE DATABASE IF NOT EXISTS`, a database called `demo` is created. `SHOW DATABASES` shows all the databased in Databricks. 
# MAGIC There are two databases available, the database named `demo` is what we just created, and Databricks automatically created the database named 'default'.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create database
# MAGIC -- CREATE DATABASE IF NOT EXISTS demo;
# MAGIC 
# MAGIC -- Show all available databases
# MAGIC SHOW DATABASES

# COMMAND ----------

# MAGIC %md
# MAGIC The SQL command `DESCRIBE DATABASE` shows that the database `demo` is saved under `dbfs:/user/hive/warehouse`, and the owner is `root`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Describe database information
# MAGIC DESCRIBE DATABASE demo;

# COMMAND ----------

# MAGIC %md
# MAGIC Using `SELECT CURRENT_DATABASE()`, we can see that the current database is `default`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check the current database
# MAGIC SELECT CURRENT_DATABASE();

# COMMAND ----------

# MAGIC %md
# MAGIC After changing the default database to `demo`, we can see that the current database shows `demo` now.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Change the current database
# MAGIC USE demo;
# MAGIC 
# MAGIC -- Check the current database
# MAGIC SELECT CURRENT_DATABASE();

# COMMAND ----------

# MAGIC %md
# MAGIC To check the tables in the database, we can use the `SHOW TABLES IN` SQL command. It does not show any results because we have not created any tables yet.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CHeck tables in a database
# MAGIC SHOW TABLES IN demo

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Create Mananged Table From Spark Dataframe Using pySpark (Method 1)

# COMMAND ----------

# MAGIC %md
# MAGIC In step 4, we will create a managed table using `pyspark`. The spark dataframe is saved as a table named `demo.crypto_1` in delta format. Using the table name without the database name `demo` will give us the same results because `demo` has been set as the default database.
# MAGIC 
# MAGIC The best practice is to write results to a Delta table. Data in Delta tables is stored in Parquet format, but has an additional layer on top of it with advanced features.

# COMMAND ----------

# MAGIC %md
# MAGIC If you got the error message `AnalysisException: Can not create the managed table('`demo`.`crypto_1`'). The associated location('dbfs:/user/hive/warehouse/demo.db/crypto_1') already exists.`, it's because the table was created in the same location before. We can use `spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")` before creating the table to handle the error message.

# COMMAND ----------

# Allow creating table using non-emply location if the table has been created before
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

# Create table
df.write.format("delta").mode("overwrite").saveAsTable("demo.crypto_1")

# COMMAND ----------

# MAGIC %md
# MAGIC After creating the table, we can see that the table `crypto_1` is in the database `demo`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Specify current database
# MAGIC USE demo;
# MAGIC 
# MAGIC -- Show tables in current database
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %md
# MAGIC The SQL code `DESCRIBE EXTENDED` provides information about the table. It has the column name, column data type, comments, as well as detailed table information. 
# MAGIC 
# MAGIC By default, the table is saved under `dbfs:/user/hive/warehouse/`, but we can change it to a different location.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Get table information
# MAGIC DESCRIBE EXTENDED crypto_1;

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the type of the table is `MANAGED`. The type of the table can also be checked using the command `spark.catalog.listTables()`. It shows that the table type is `MANAGED` as well.

# COMMAND ----------

# List table information
spark.catalog.listTables()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: Create Managed Table From Existing Table Using SQL (Method 2)

# COMMAND ----------

# MAGIC %md
# MAGIC In step 5, we will create a managed table from an existing managed table using SQL. SQL queries can be run direclty on the existing tables in the database.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SQL query to create table
# MAGIC CREATE OR REPLACE TABLE demo.crypto_2 AS
# MAGIC SELECT * 
# MAGIC FROM demo.crypto_1
# MAGIC WHERE Asset_ID = 1

# COMMAND ----------

# MAGIC %md
# MAGIC Now the database `demo` has two tables, crypto_1 and crypto_2.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Specify current database 
# MAGIC USE demo;
# MAGIC 
# MAGIC -- Show tables in the current database
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %md
# MAGIC The describing results show that crypto_2 is a managed table as well.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Describe table information
# MAGIC DESCRIBE EXTENDED crypto_2;

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 6: Create Managed Table From Spark Dataframe Using SQL (Method 3)

# COMMAND ----------

# MAGIC %md
# MAGIC In step 6, we will create a managed table from a spark dataframe using SQL. SQL code does not work on spark dataframe directly, so we need to create a view for the dataframe and run SQL code on the view.
# MAGIC 
# MAGIC Using the code `createOrReplaceTempView`, a temp view is created for the spark dataframe.

# COMMAND ----------

# Create a temp view
df.createOrReplaceTempView('df')

# COMMAND ----------

# MAGIC %md
# MAGIC Next, a table named crypto_3 is created by querying the temp view.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create table using SQL query
# MAGIC CREATE OR REPLACE TABLE crypto_3 AS
# MAGIC SELECT * FROM df

# COMMAND ----------

# MAGIC %md
# MAGIC Using the magic command `%sql` is equivalent to using the spark sql code.

# COMMAND ----------

# Use spark.sql to run SQL query
spark.sql(
'''
CREATE OR REPLACE TABLE crypto_3 AS
SELECT * FROM df
'''
)

# COMMAND ----------

# MAGIC %md
# MAGIC An alternative way is to define a table with column names and column types, then insert the data into the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create an empty table with column names and their types
# MAGIC CREATE OR REPLACE TABLE crypto_3 (
# MAGIC timestamp INT,
# MAGIC Asset_ID INT,
# MAGIC Count INT,
# MAGIC Open DOUBLE,
# MAGIC High DOUBLE,
# MAGIC Low DOUBLE,
# MAGIC Close DOUBLE,
# MAGIC Volume DOUBLE,
# MAGIC VWAP DOUBLE,
# MAGIC Target DOUBLE
# MAGIC );
# MAGIC 
# MAGIC -- Insert data into the table
# MAGIC INSERT INTO crypto_3
# MAGIC SELECT * FROM df

# COMMAND ----------

# MAGIC %md
# MAGIC From the output of `spark.catalog.listTables()`, we can see that the three tables we created so far are all managed tables. And `df` is a temporary table.

# COMMAND ----------

# List all tables and their information
spark.catalog.listTables()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 7: Create Unmananged Table From Spark Dataframe Using pySpark (Method 4)

# COMMAND ----------

# MAGIC %md
# MAGIC In step 7, we will create an unmanaged table from a spark dataframe using pyspark. The syntax is very similar to the syntax for creating a managed table. The only difference is that we specify the table location as an external folder in S3 bucket. 

# COMMAND ----------

# Create external table 
df.write.format("delta").mode("overwrite").option("path", "/mnt/demo4tutorial/output/crypto_4").saveAsTable("crypto_4")

# COMMAND ----------

# MAGIC %md
# MAGIC After the table is created, we can see that the S3 bucket has a new folder called crypto_4.

# COMMAND ----------

# MAGIC %fs ls "/mnt/demo4tutorial/output"

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 8: Create Unmanaged Table From Spark Dataframe Using SQL (Method 5)

# COMMAND ----------

# MAGIC %md
# MAGIC In step 8, we will create an external table from a spark dataframe using SQL. SQL code does not work on spark dataframe directly, so we need to create a view for the dataframe and run SQL code on the view.
# MAGIC 
# MAGIC Using the code `createOrReplaceTempView`, a temp view is created for the spark dataframe. There is no need to recreate the temp view if you have created it in step 6.

# COMMAND ----------

# Create a temp view
df.createOrReplaceTempView("df")

# COMMAND ----------

# MAGIC %md
# MAGIC The only difference between creating a managed and external table using SQL is the `LOCATION`. When an external location is specified in the SQL code, an unmanaged table will be created.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create an external table
# MAGIC DROP TABLE IF EXISTS demo.crypto_5; 
# MAGIC CREATE TABLE demo.crypto_5 
# MAGIC USING delta
# MAGIC LOCATION "/mnt/demo4tutorial/output/crypto_5/"
# MAGIC SELECT * FROM df

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use OPTIONS to specify the path.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create an external table
# MAGIC DROP TABLE IF EXISTS demo.crypto_5; 
# MAGIC CREATE TABLE demo.crypto_5 
# MAGIC USING delta
# MAGIC OPTIONS (path "/mnt/demo4tutorial/output/crypto_5/")
# MAGIC SELECT * FROM df

# COMMAND ----------

# MAGIC %md 
# MAGIC If defining the column types is needed when creating the table, we can create an empty table to define column types first, then insert data into the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create an external table using defined column types
# MAGIC DROP TABLE IF EXISTS demo.crypto_5; 
# MAGIC CREATE TABLE demo.crypto_5 (
# MAGIC timestamp INT,
# MAGIC Asset_id STRING,
# MAGIC Count INT,
# MAGIC Open DOUBLE,
# MAGIC High DOUBLE,
# MAGIC Low DOUBLE,
# MAGIC Close DOUBLE,
# MAGIC Volume DOUBLE,
# MAGIC VWAP DOUBLE,
# MAGIC Target DOUBLE
# MAGIC )
# MAGIC USING delta
# MAGIC LOCATION "/mnt/demo4tutorial/output/crypto_5/";
# MAGIC 
# MAGIC INSERT INTO demo.crypto_5
# MAGIC SELECT * FROM df

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 9: Delete Managed And Unmanaged Tables

# COMMAND ----------

# MAGIC %md
# MAGIC In step 9, we will talk about how to delete managed and unmanaged tables in Databricks.
# MAGIC 
# MAGIC Firstly, let's check the tables we created in the database called `demo`. We can see that all five tables are in the database. The temp view df is also saved in the demo database.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show tables
# MAGIC SHOW TABLES IN demo

# COMMAND ----------

# MAGIC %md
# MAGIC `spark.catalog.listTables` shows that the tables crypto_1, crypto_2, and crypto_3 are managed tables. And the table crypto_4 and crypto_5 are external tables.

# COMMAND ----------

# List table information
spark.catalog.listTables()

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's run `DROP TABLE` to drop the tables we created.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Drop tables
# MAGIC DROP TABLE IF EXISTS crypto_1;
# MAGIC DROP TABLE IF EXISTS crypto_2;
# MAGIC DROP TABLE IF EXISTS crypto_3;
# MAGIC DROP TABLE IF EXISTS crypto_4;
# MAGIC DROP TABLE IF EXISTS crypto_5;

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that all the tables are deleted from the database.

# COMMAND ----------

# List table information
spark.catalog.listTables()

# COMMAND ----------

# MAGIC %md
# MAGIC But the unmanaged tables saved on external locations such as the S3 bucket still exist.
# MAGIC 
# MAGIC To delete the external tables from S3, we can use `dbutils.fs.rm`. `recurse=True` deletes a folder and the files in the folder.

# COMMAND ----------

dbutils.fs.rm('/mnt/demo4tutorial/output/crypto_4/', recurse=True)
dbutils.fs.rm('/mnt/demo4tutorial/output/crypto_5/', recurse=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Summary

# COMMAND ----------

# MAGIC %md
# MAGIC This tutorial demonstrates five different ways to create tables in databricks. It covered:
# MAGIC * What's the difference between managed and external tables?
# MAGIC * How to mount S3 bucket to Databricks and read CSV to spark dataframe?
# MAGIC * How to create a database in Databricks?
# MAGIC * How to create a managed table from a spark datafram using pySpark?
# MAGIC * How to create a managed table from an existing table using SQL?
# MAGIC * How to create a managed table from a spark datafram using SQL?
# MAGIC * How to create an unmanaged table from a spark datafram using pySpark?
# MAGIC * How to create an unmanaged table from a spark datafram using SQL?
# MAGIC * How to delete managed and external tables? 
