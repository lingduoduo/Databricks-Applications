# Databricks notebook source
# MAGIC %md 
# MAGIC ### Silver-to-Gold: Window and Rollup pois-pings to reduce noise and provide clean datasets for downstream analyses
# MAGIC
# MAGIC This notebook builds off the ```02_Bronze_to_Silver_H3_Indexed``` notebook that generates indexed POI locations. Using Spark Window function, we demonstrate how to effectively aggregate visits by individual advertising IDs based on "pings" along time axis for all POIs.  We show how to estimate the overall traffic pattern for specific POIs during a given time window (e.g., two weeks period between 2019-03-01 and 2019-03-15).  These estimates are essentially one type of insights derived from raw "pings" data that can be queried directly for business intelligence and/or modeled with advanced machine learning algorithms. 

# COMMAND ----------

from pyspark.sql import Row, DataFrame

# COMMAND ----------

# MAGIC %md #### Sanity check Silver table

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTENDED geospatial_lakehouse_blog_db.silver_us_poi_pings_h3_indexed

# COMMAND ----------

# MAGIC %md #### Select relevant subset of POI + Mobility data from the Silver table(s)
# MAGIC
# MAGIC From Silver tables, we select the features that we want to aggregate on, including advertising IDs (deidentifiable device ids)

# COMMAND ----------

gold_h3_indexed_ad_ids_df = spark.sql("""
 SELECT ad_id, geo_hash_region, geo_hash, h3_index, utc_date_time FROM geospatial_lakehouse_blog_db.silver_us_poi_pings_h3_indexed
 ORDER BY geo_hash_region 
""")

# COMMAND ----------

# Sanity check our data set 
display(gold_h3_indexed_ad_ids_df)

# COMMAND ----------

# temp table/view for manipulation below
gold_h3_indexed_ad_ids_df.createOrReplaceTempView("gold_h3_indexed_ad_ids")

# COMMAND ----------

# MAGIC %md #### Window over datetimes paritioned by mobility-data advertising ids, processing geohashes for coalescing 
# MAGIC
# MAGIC Set up "pings" data with ordering on advertising IDs, time frame, and lag for coalescing

# COMMAND ----------

gold_h3_lag_df = spark.sql("""
  SELECT ad_id, geo_hash, h3_index, utc_date_time, row_number() OVER(PARTITION BY ad_id
    ORDER BY utc_date_time ASC) AS rn,
    lag(geo_hash, 1) over(partition by ad_id 
    ORDER BY utc_date_time ASC) AS prev_geo_hash
  FROM gold_h3_indexed_ad_ids
""")

# COMMAND ----------

# Sanity check our data set 
display(gold_h3_lag_df)

# COMMAND ----------

# temp table/view for manipulation below
gold_h3_lag_df.createOrReplaceTempView("gold_h3_lag")

# COMMAND ----------

# MAGIC %md #### Rollup pings to reduce noise within datetime windows by coalescing along them
# MAGIC
# MAGIC Coalescing "pings" by geometries

# COMMAND ----------

gold_h3_coalesced_df = spark.sql(""" 
  SELECT ad_id, geo_hash, h3_index, utc_date_time AS ts, rn, coalesce(prev_geo_hash, geo_hash) AS prev_geo_hash 
    FROM gold_h3_lag
  """)

# COMMAND ----------

# Sanity check our data set 
display(gold_h3_coalesced_df)

# COMMAND ----------

# temp table/view for manipulation below
gold_h3_coalesced_df.createOrReplaceTempView("gold_h3_coalesced")

# COMMAND ----------

# MAGIC %md #### Group points-in-polygons by rolled-up pings
# MAGIC
# MAGIC Generate a table of advertising IDs that are present in all geometries into temp tables.  This effectively groups a collection of geohashes if they belong to the same "pings".

# COMMAND ----------

gold_h3_cleansed_poi_df = spark.sql(""" 
  SELECT ad_id, geo_hash, h3_index, ts,
    SUM(CASE WHEN geo_hash = prev_geo_hash THEN 0 ELSE 1 END) OVER (ORDER BY ad_id, rn) AS group_id
    FROM gold_h3_coalesced
  """)
gold_h3_cleansed_poi_df.createOrReplaceTempView("gold_h3_cleansed_poi")

# COMMAND ----------

# Sanity check our data set 
display(gold_h3_cleansed_poi_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM gold_h3_cleansed_poi

# COMMAND ----------

# MAGIC %md #### Save the cleansed, grouped, rolledup pings-per-POIs to a persistent gold table 

# COMMAND ----------

# write this out into a gold table 
gold_h3_cleansed_poi_df.write.format("delta").save("/dbfs/ml/blogs/geospatial/delta/gold_h3_cleansed_poi")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE geospatial_lakehouse_blog_db.gold_h3_cleansed_poi 
# MAGIC USING DELTA LOCATION '/dbfs/ml/blogs/geospatial/delta/gold_h3_cleansed_poi'

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE geospatial_lakehouse_blog_db.gold_h3_cleansed_poi ZORDER BY h3_index;

# COMMAND ----------

# MAGIC %sql
# MAGIC ANALYZE TABLE geospatial_lakehouse_blog_db.gold_h3_cleansed_poi COMPUTE STATISTICS;

# COMMAND ----------

# MAGIC %md #### Further rollup pings by extracting the pings first and last timestamps to provide windowing intervals
# MAGIC
# MAGIC Estimate the first and last ping per advertising IDs for each specific geometry

# COMMAND ----------

gold_h3_coalesced_windowed_df = spark.sql(""" 
  SELECT ad_id, h3_index,         	
         first_value(ts) over (partition BY ad_id, group_id ORDER BY ts ASC) AS ts1,
         last_value(ts) over(partition BY ad_id, group_id ORDER BY ts ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING
                        AND UNBOUNDED FOLLOWING) AS ts2
         FROM gold_h3_cleansed_poi
  """)

# COMMAND ----------

# Sanity check our data set 
display(gold_h3_coalesced_windowed_df)

# COMMAND ----------

# temp table/view for manipulation below
gold_h3_coalesced_windowed_df.createOrReplaceTempView("gold_h3_coalesced_windowed")

# COMMAND ----------

# MAGIC %md #### Further window pois-pings by origin (advertising id), timestamp intervals and counts
# MAGIC
# MAGIC Generate frequency table of advertising IDs per geometry within a time span

# COMMAND ----------

gold_h3_windowed_with_count_df = spark.sql(""" 
  SELECT ad_id, h3_index, ts1, ts2, COUNT(*) AS ct 
  FROM gold_h3_coalesced_windowed GROUP BY 1, 2, 3, 4
""")

# COMMAND ----------

# Sanity check our data set 
display(gold_h3_windowed_with_count_df)

# COMMAND ----------

# temp table/view for manipulation below
gold_h3_windowed_with_count_df.createOrReplaceTempView("gold_h3_windowed_with_count")

# COMMAND ----------

# MAGIC %md #### Save the windowed-per-datetime and -per-count pois-pings as another persistent gold table

# COMMAND ----------

# write this out into a gold table 
gold_h3_windowed_with_count_df.write.format("delta").save("/dbfs/ml/blogs/geospatial/delta/gold_h3_windowed_with_count")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE geospatial_lakehouse_blog_db.gold_h3_windowed_with_count 
# MAGIC USING DELTA LOCATION '/dbfs/ml/blogs/geospatial/delta/gold_h3_windowed_with_count'

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE geospatial_lakehouse_blog_db.gold_h3_windowed_with_count ZORDER BY h3_index;

# COMMAND ----------

# MAGIC %sql
# MAGIC ANALYZE TABLE geospatial_lakehouse_blog_db.gold_h3_windowed_with_count COMPUTE STATISTICS;

# COMMAND ----------

# MAGIC %md #### Produce a cleansed dataset for pois-pings carrying a singular, rolled up geohash per datetime window together with a cleansed, statistically significant count of pings as a weighing feature

# COMMAND ----------

gold_h3_poi_cleansed_windowed_df = spark.sql("""
          SELECT ad_id, h3_index, ct,
            ts1 AS utc_start_ts,
            ts2 AS utc_end_ts
            FROM gold_h3_windowed_with_count
          """)
gold_h3_poi_cleansed_windowed_df.createOrReplaceTempView("gold_h3_poi_cleansed_windowed")

# COMMAND ----------

# MAGIC %md #### Run an example query off this cleansed, windowed dataset

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM gold_h3_poi_cleansed_windowed

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_h3_poi_cleansed_windowed WHERE utc_start_ts LIKE '2021-09-03%' AND utc_end_ts LIKE '2021-09-03%'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- test this query with, say, 4 days of data
# MAGIC SELECT * FROM gold_h3_poi_cleansed_windowed WHERE utc_start_ts >= '2021-08-31 00:00:00' AND utc_end_ts <= '2021-09-03 23:59:59'

# COMMAND ----------

# MAGIC %md ### Save this cleansed, windowed dataset as another gold table

# COMMAND ----------

# write this out into a gold table 
gold_h3_poi_cleansed_windowed_df.write.format("delta").save("/dbfs/ml/blogs/geospatial/delta/gold_h3_poi_cleansed_windowed")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE geospatial_lakehouse_blog_db.gold_h3_poi_cleansed_windowed 
# MAGIC USING DELTA LOCATION '/dbfs/ml/blogs/geospatial/delta/gold_h3_poi_cleansed_windowed'

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE geospatial_lakehouse_blog_db.gold_h3_poi_cleansed_windowed ZORDER BY h3_index;

# COMMAND ----------

# MAGIC %sql
# MAGIC ANALYZE TABLE geospatial_lakehouse_blog_db.gold_h3_poi_cleansed_windowed COMPUTE STATISTICS;
