# Databricks notebook source
# MAGIC %run ./Classroom-Setup

# COMMAND ----------

import pyspark.sql.functions as F

# Load dataset
cols = ["accommodates", "bedrooms", "beds", "minimum_nights", "number_of_reviews", "review_scores_rating", "price", "neighbourhood_cleansed", "property_type", "room_type"]
airbnb_df = spark.read.format("delta").load(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/").select(cols)
df1, df2 = airbnb_df.randomSplit([0.5, 0.5], seed=42)

# Simulate drift on second time period
df2 = df2.withColumn("neighbourhood_cleansed", F.when((F.col("neighbourhood_cleansed") == "Mission"), F.lit(None)).otherwise(F.col("neighbourhood_cleansed")))
df2 = df2.withColumn("price", 2 * F.col("price"))
df2 = df2.withColumn("review_scores_rating", F.col("review_scores_rating") / 5 )

None # Suppress Output

# COMMAND ----------

data_path1 = f"{DA.paths.working_dir}/driftexample/data1"
data_path2 = f"{DA.paths.working_dir}/driftexample/data2"

dbutils.fs.rm(data_path1, True)
dbutils.fs.mkdirs(data_path1)

dbutils.fs.rm(data_path2, True)
dbutils.fs.mkdirs(data_path2)

df1.write.format("delta").mode("overwrite").save(data_path1)
df2.write.format("delta").mode("overwrite").save(data_path2)

None # Suppress Output

