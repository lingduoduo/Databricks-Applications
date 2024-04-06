# Databricks notebook source
# MAGIC %md
# MAGIC ### Raw-to-Bronze Preprocessing Safegraph Geometries for H3
# MAGIC
# MAGIC This notebook handles raw Point Of Interest data from Safegraph and creates Delta tables with preprocessed H3 indexes. Each H3 index relates to a hexagon located on the map as defined by [Ubers H3 library.](https://h3geo.org/)
# MAGIC
# MAGIC We do this process ahead of time and persist the intermediate results with H3 indexes as it is extremely compute/memory intensive. Because this is POI data, once the data is processed, very little changes are made.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Install required python geospatial libraries
# MAGIC
# MAGIC The two main librarys required for handling geometry and polygon data in python are `H3` and `Geopandas`. For the Scala version of this, please see the notebook demonstrating the use of Geomesa libraries to complete the same process.

# COMMAND ----------

# MAGIC %pip install h3

# COMMAND ----------

# MAGIC %pip install geopandas

# COMMAND ----------

import h3
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, input_file_name, pandas_udf, PandasUDFType, split, explode, hex
import geopandas
import pandas as pd
from shapely.geometry import mapping


# COMMAND ----------

# MAGIC %md
# MAGIC ####Read Data in from the raw csv files
# MAGIC
# MAGIC Here we are reading from the most recent vendor delivery of POI data, which contains all kinds of interesting places such as shops, malls, buildings, even national parks. The key feild here is the `geometry_wkt` column which contains ploygons of the outline of the location using precise lattitudes and longtitudes.

# COMMAND ----------

schema = StructType([StructField("placekey",StringType(),True),StructField("safegraph_place_id",StringType(),True),StructField("parent_placekey",StringType(),True),StructField("parent_safegraph_place_id",StringType(),True),StructField("safegraph_brand_ids",StringType(),True),StructField("location_name",StringType(),True),StructField("brands",StringType(),True),StructField("top_category",StringType(),True),StructField("sub_category",StringType(),True),StructField("naics_code",StringType(),True),StructField("latitude",StringType(),True),StructField("longitude",StringType(),True),StructField("street_address",StringType(),True),StructField("city",StringType(),True),StructField("region",StringType(),True),StructField("postal_code",StringType(),True),StructField("open_hours",StringType(),True),StructField("category_tags",StringType(),True),StructField("opened_on",StringType(),True),StructField("closed_on",StringType(),True),StructField("tracking_opened_since",StringType(),True),StructField("tracking_closed_since",StringType(),True),StructField("polygon_wkt",StringType(),True),StructField("polygon_class",StringType(),True),StructField("building_height",StringType(),True),StructField("enclosed",StringType(),True),StructField("phone_number",StringType(),True),StructField("is_synthetic",StringType(),True),StructField("includes_parking_lot",StringType(),True),StructField("iso_country_code",StringType(),True)])


# COMMAND ----------

raw_df = spark.read.format("csv").schema(schema) \
.option("delimiter", ",") \
.option("quote", "\"") \
.option("escape", "\"")\
.option("header", "true")\
.load("dbfs:/ml/blogs/geospatial/poi/2021/09/03/22/*")

display(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create pyspark UDFs to utilize h3 and geopandas functionality
# MAGIC
# MAGIC Pandas UDFs are vectorized and use arrow serialization and should perform at similiar parities to scala UDFs

# COMMAND ----------

@pandas_udf('string')
def poly_to_H3(wkts: pd.Series) -> pd.Series:
    polys = geopandas.GeoSeries.from_wkt(wkts)
    h3_list = [str(h3.polyfill_geojson(mapping(poly), 13)).replace("{", "").replace("}", " ").replace("'", "") if poly is not None else '{0}' for poly in polys]
    return pd.Series(h3_list)
  
  
@pandas_udf('float')
def poly_area(wkts: pd.Series) -> pd.Series:
    polys = geopandas.GeoSeries.from_wkt(wkts)
    return polys.area

# COMMAND ----------

# MAGIC %md 
# MAGIC ####Write raw POI data to a Delta table
# MAGIC
# MAGIC It is always perferable to keep a copy of our raw source table in a clean Delta format. This is also key so that we can reparition the table and write out smaller files. While you typically try to aviod small files, we want to make sure we keep our read partitions small while consuming from this table. You will see why in the following sections.

# COMMAND ----------

raw_df.repartition(500).write.format("delta").mode("overwrite").saveAsTable("geospatial_lakehouse_blog_db.raw_safegraph_poi")

# COMMAND ----------

# MAGIC %md
# MAGIC ####limit the size of each read partition
# MAGIC
# MAGIC this helps to limit the amount of data on each core during the explosion operation. We also manage this by finding the areas of the largest POIs and filtering them out
# MAGIC
# MAGIC `1 byte * 1024  * 1024 = 1mb`

# COMMAND ----------

1 * 1024  * 1024

# COMMAND ----------

# MAGIC %sql
# MAGIC set spark.sql.files.maxPartitionBytes = 1048576

# COMMAND ----------

# MAGIC %md
# MAGIC ####Transform polygons by assigning each a series of h3_ids
# MAGIC
# MAGIC Here is the main transformation where we select our key columns of value going forward, filter out the types of records that we can't use in our UDFs due to corrupted or unexpected data, filter out polygons with areas that are too large and would produce too many h3_id references and crash the process. If we are filling up a starbucks location with 25 hexagons, just imagine how many hexagons in would take to fill something like a ski resort or a national park. In general, if you need fine grained granularity for extermely large POIs, there are ways to start with a more coarse resolution and and then traverse into the rest of the dataset by using heirarchical H3 functionality. We can display that another time.
# MAGIC
# MAGIC Next we need to convert our single rows for each location, where Polygons in WKT format portray the precise location, into several indexed rows of that are assigned to a singe fine grained H3 index. This allows for highly performant stardardardized joins to blend other datasets, which you will see in the next notebook.

# COMMAND ----------

h3_df = spark.table("geospatial_lakehouse_blog_db.raw_safegraph_poi").repartition(500)\
        .select("placekey", "safegraph_place_id", "parent_placekey", "parent_safegraph_place_id", "location_name", "brands", "latitude", "longitude", "street_address", "city", "region", "postal_code", "polygon_wkt") \
        .filter(col("polygon_wkt").isNotNull() & (~ col("polygon_wkt").like("MULTI%")) & (~ col("polygon_wkt").like("POLYGON((%"))) \
        .withColumn("area", poly_area(col("polygon_wkt")))\
        .filter(col("area") < 0.001)\
        .withColumn("h3", poly_to_H3(col("polygon_wkt"))) \
        .withColumn("h3_array", split(col("h3"), ","))\
        .drop("polygon_wkt")\
        .withColumn("h3", explode("h3_array"))\
        .drop("h3_array").withColumn("h3_hex", hex("h3"))

#finding the areas of each polygon
#filtering out the super large ones to limit data explosion
display(h3_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Write H3 indexed data to Delta table
# MAGIC
# MAGIC For this demonstration we are simply overwriting the results each time since we are dealing with a one time load. In production you would likely define a more elaborate loading pattern.

# COMMAND ----------

h3_df.write.format("delta").mode("overwrite").saveAsTable("geospatial_lakehouse_blog_db.h3_indexed_safegraph_poi")

# COMMAND ----------

# MAGIC %md #### Perform Delta Lake `OPTIMIZE`, `ZORDER BY` and `VACUUM`
# MAGIC
# MAGIC Here we `OPTIMIZE` the Delta table, ZORDERing by the h3 indices, and lat/lon 

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE geospatial_lakehouse_blog_db.h3_indexed_safegraph_poi ZORDER BY (h3, h3_hex, latitude, longitude)

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from geospatial_lakehouse_blog_db.h3_indexed_safegraph_poi
