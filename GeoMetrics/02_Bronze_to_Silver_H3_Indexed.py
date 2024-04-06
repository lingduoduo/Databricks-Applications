# Databricks notebook source
# MAGIC %md
# MAGIC ### Bronze-to-Silver: Blend Datasets Using H3 Index
# MAGIC
# MAGIC This notebook builds off the ```01_Raw_to_Bronze_Safegraph_Geometries``` notebook which indexes POI locations (from Safegraph). Now that we know how to transform the polygons which make up POIs into easily identifiable H3 indexes at a certain resolution, we can start relating them to foot traffic from our mobility data (from Veraset). Our mobility data consists of daily "pings" identified with latitude and longitude coordinates. With these coordinates we can derive a H3 hexagonal index at the same resolution as our POI data. Then we can then identify which pings occured in our POIs, and perform aggregations to extract meaningful metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Installing the Python H3 library

# COMMAND ----------

# MAGIC %pip install h3

# COMMAND ----------

import h3
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, pandas_udf, PandasUDFType, hex, from_unixtime, substring
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC #### Viewing our daily mobility data which is partitioned on the day
# MAGIC
# MAGIC We use Veraset data here. Typically Veraset will deliver data paritioned by day on a regular time cadence. 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Reading a single days worth of our mobility data. 
# MAGIC
# MAGIC The volume here is very large; we can limit the size as much as possible by performing these operations on a daily basis, persisting the outcome to our Geospatial Lakehouse as silver tables.

# COMMAND ----------

datalake = "dbfs:/ml/blogs/geospatial"
schema_path = "/mobility/raw/2021/09/03"
path = "/mobility/raw"
res=13

veraset_raw_data_df = spark.read.parquet(datalake+schema_path)

display(veraset_raw_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Defining a UDF to convert coordinates to H3 hexagons
# MAGIC
# MAGIC We can use the H3 python library to to get the hexagon that a particular coodinate falls into at a defined resolution. Wrapping the functionality into a regular Spark UDF, performs the operations row-at-a-time in the python kernal. Some options for imporved performance include writing the function in Scala, using the Java H3 library and then calling it from Python. Another full Python option includes utilizing Pandas UDFs for faster serialization rates with Apache Arrow and vectorized processing. For now we demonstrate with a simple implementation.

# COMMAND ----------

@udf
def geo_to_H3(lat: float, long: float) -> str: 
  h3_index = h3.geo_to_h3(lat, long, res)   
  return h3_index

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####Transform mobility data
# MAGIC
# MAGIC Addding human readable date and timestamp feilds, as well as hex values of the derived h3_indexes, at resolution 13.

# COMMAND ----------

veraset_extended_cols_df = (veraset_raw_data_df
                     .withColumn('utc_date_time',from_unixtime('utc_timestamp','yyyy-MM-dd HH:mm:ss'))
                     .withColumn('utc_date',from_unixtime('utc_timestamp','yyyy-MM-dd'))
                     .withColumn('geo_hash_region', substring('geo_hash', 1, 3))
                     .withColumn("h3_res13", geo_to_H3(col("latitude"), col("longitude")))
                     .withColumn("h3", hex(geo_to_H3(col("latitude"), col("longitude")))) )

display(veraset_extended_cols_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Persisting our processed dataset to a Delta table

# COMMAND ----------

# Perform some data quality control to assure we can use our ```h3``` data for joins downstream
from pyspark.sql.functions import col, lower 
veraset_extended_cols_df = veraset_extended_cols_df.withColumn("h3", lower(col("h3")))
display(veraset_extended_cols_df)

# COMMAND ----------

veraset_extended_cols_df.write.format("delta").mode("overwrite").partitionBy("utc_date").saveAsTable("geospatial_lakehouse_blog_db.us_pings_by_date")

# COMMAND ----------

# MAGIC %md #### Perform Delta Lake `OPTIMIZE`, `ZORDER BY` and `VACUUM`
# MAGIC
# MAGIC Here we `OPTIMIZE` the Delta table, ZORDERing by the h3 indices, and lat/lon, to improve downstream query performance

# COMMAND ----------

# MAGIC %sql OPTIMIZE geospatial_lakehouse_blog_db.us_pings_by_date ZORDER BY (h3_res13, h3, latitude, longitude)

# COMMAND ----------

# MAGIC %sql
# MAGIC VACUUM geospatial_lakehouse_blog_db.us_pings_by_date

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####Daily incremental load function
# MAGIC
# MAGIC This function is meant to perform the join with Safegraph and persist the data to a silver table for a days worth of pings data at a time. We would easily convert this function to handle a few days at a time as well.

# COMMAND ----------

# MAGIC %scala 
# MAGIC
# MAGIC def dailyIncrementalLoad(start_date: String, end_date: String): Unit = {
# MAGIC
# MAGIC //planning to make this run a few days at a time in the future
# MAGIC // val start_date: String = start_date
# MAGIC val end_date: String = start_date
# MAGIC   
# MAGIC println( "*" * 50) 
# MAGIC println(s"Beginning run for $start_date")
# MAGIC   
# MAGIC val predicate_str: String = s"utc_date >= to_date('$start_date') AND utc_date <= to_date('$end_date')"
# MAGIC
# MAGIC val sql_query = s"""
# MAGIC SELECT
# MAGIC   DISTINCT vera.utc_date_time,
# MAGIC   vera.ad_id,
# MAGIC   poi.safegraph_place_id,
# MAGIC   poi.brands,
# MAGIC   poi.location_name,
# MAGIC   poi.street_address,
# MAGIC   poi.city,
# MAGIC   poi.region,
# MAGIC   poi.postal_code,
# MAGIC   vera.utc_date,
# MAGIC   vera.ip_address,
# MAGIC   vera.horizontal_accuracy,
# MAGIC   vera.latitude as ping_latitude,
# MAGIC   vera.longitude as ping_longitude,
# MAGIC   vera.geo_hash,
# MAGIC   vera.geo_hash_region,
# MAGIC   poi.h3 as h3_res13,
# MAGIC   vera.h3 as h3_index,
# MAGIC   vera.iso_country_code
# MAGIC FROM
# MAGIC   geospatial_lakehouse_blog_db.h3_indexed_safegraph_poi poi
# MAGIC   INNER JOIN geospatial_lakehouse_blog_db.us_pings_by_date vera ON poi.h3_hex = vera.h3
# MAGIC WHERE
# MAGIC   $predicate_str
# MAGIC   
# MAGIC """
# MAGIC
# MAGIC val input_df = spark.sql(sql_query)
# MAGIC   
# MAGIC println(s"Loading data for $start_date") 
# MAGIC  
# MAGIC val silver_table_path = "/ml/blogs/geospatial/delta/silver_us_poi_pings_h3_indexed" 
# MAGIC
# MAGIC (input_df.write.format("delta")
# MAGIC  .mode("overwrite")
# MAGIC  .option("replaceWhere", s"utc_date >= '$start_date' AND utc_date <= '$end_date'")
# MAGIC  .option("mergeSchema", true)
# MAGIC  .partitionBy("utc_date", "region")
# MAGIC  .saveAsTable("geospatial_lakehouse_blog_db.silver_us_poi_pings_h3_indexed") )
# MAGIC   
# MAGIC println(s"Running OPTIMIZE and ZORDER for $start_date")
# MAGIC   
# MAGIC spark.sql(s"""OPTIMIZE geospatial_lakehouse_blog_db.silver_us_poi_pings_h3_indexed
# MAGIC WHERE $predicate_str
# MAGIC ZORDER BY (h3_index, utc_date_time)""")
# MAGIC   
# MAGIC   println(s"Completed cycle for $start_date")
# MAGIC   println( "*" * 50)
# MAGIC }

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####Define days that are needed to load and run the load.

# COMMAND ----------

# MAGIC %scala
# MAGIC // used for incremental loading (post initial table load)
# MAGIC val loaded_dates_list = spark.sql("SELECT DISTINCT utc_date FROM geospatial_lakehouse_blog_db.us_pings_by_date").collect.map(x=>x(0)).toList

# COMMAND ----------

# MAGIC %scala 
# MAGIC // use the below to initially load the table
# MAGIC val dates_to_load = spark.sql("SELECT DISTINCT utc_date FROM geospatial_lakehouse_blog_db.us_pings_by_date").collect.map(x=>x(0).toString).toList.sorted
# MAGIC // use the below once daily batch ingestion runs are set up 
# MAGIC // val dates_to_load = spark.sql("SELECT DISTINCT utc_date FROM geospatial_lakehouse_blog_db.us_pings_by_date").filter(!$"utc_date".isin(loaded_dates_list:_*)).collect.map(x=>x(0).toString).toList.sorted
# MAGIC

# COMMAND ----------

# MAGIC %scala 
# MAGIC
# MAGIC for (day <- dates_to_load)
# MAGIC {
# MAGIC   dailyIncrementalLoad(day, day)
# MAGIC }

# COMMAND ----------

# MAGIC %scala
# MAGIC val dates_to_load = spark.sql("select distinct utc_date from geospatial_lakehouse_blog_db.us_pings_by_date").collect.map(x=>x(0).toString).toList.sorted

# COMMAND ----------

# MAGIC %md #### Let's review our new Silver table

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM geospatial_lakehouse_blog_db.silver_us_poi_pings_h3_indexed

# COMMAND ----------

# MAGIC %sql
# MAGIC ANALYZE TABLE geospatial_lakehouse_blog_db.silver_us_poi_pings_h3_indexed COMPUTE STATISTICS;

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESCRIBE DETAIL geospatial_lakehouse_blog_db.silver_us_poi_pings_h3_indexed
