# Databricks notebook source
# MAGIC %pip install dython==0.7.1
# MAGIC %pip install databricks-feature-engineering
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import window as w 
from pyspark.sql.types import FloatType, IntegerType

# COMMAND ----------

new_tx = spark.sql("""select * from data_science_prod.default.transactions_users_silver
                     where t_instance_timestamp/1000 > (UNIX_TIMESTAMP(current_timestamp()) - 30 * 24 * 60 * 60); """)

# COMMAND ----------

display(new_tx)

# COMMAND ----------

# DBTITLE 1,Enhanced Features
win_asc = w.Window.partitionBy('t_caden_alias').orderBy(F.col('t_transaction_time'))
win_desc = w.Window.partitionBy('t_caden_alias').orderBy(F.col('t_transaction_time').desc())
new_tx_enhanced = new_tx.withColumn(
    'days_since_prior_transaction',
    F.coalesce(  # Replace nulls with a default value, e.g., 0
        F.datediff(
            F.col('t_transaction_time'),
            F.lag('t_transaction_time', 1).over(win_asc)
        ),
        F.lit(0)
    )
)

new_tx_enhanced = (
    new_tx_enhanced
      .withColumn(
        'days_prior_to_last_transaction', 
        F.sum('days_since_prior_transaction').over(win_desc) - F.coalesce(F.col('days_since_prior_transaction'), F.lit(0))
        )
)

# COMMAND ----------

display(new_tx_enhanced)

# COMMAND ----------

# DBTITLE 1,User Features
new_tx_users = (new_tx_enhanced.withColumn('row_num', F.row_number().over(win_desc))
                .filter(F.col('row_num') == 1)
                .select('u_caden_alias',
                        'u_age_group',
                        'u_gender_group',
                        'u_yearly_household_income',
                        'u_ethnicity',
                        'u_zipcode',
                        'u_state',
                        'days_since_prior_transaction',
                        'days_prior_to_last_transaction'
                        ))

# COMMAND ----------

display(new_tx_users)

# COMMAND ----------

# display(new_tx_transactions.select(
#     't_caden_alias', 
#     't_transaction_time', 
#     'days_since_prior_transaction',
#     'days_prior_to_last_transaction'
# )
# .filter(F.col('t_caden_alias') == '002ea8cb-e8a9-4e41-a541-e75dd01b61c4')
# )

# COMMAND ----------

enhanced_trans_features = (
    new_tx.groupBy(
        "t_caden_alias", F.window("t_transaction_time", "7 days")
    )  
    .agg(
        F.mean("t_price").alias("mean_transaction_window_7_day"),  
        F.count("*").alias("count_transaction_window_7_day"),
    )
    .select(
        "t_caden_alias",
        F.unix_timestamp(F.col("window.end")).cast("timestamp").alias("window_end_ts"),
        F.col("mean_transaction_window_7_day").cast(FloatType()),  
        F.col("count_transaction_window_7_day").cast(IntegerType()),
    )
)

# COMMAND ----------

display(enhanced_trans_features)

# COMMAND ----------

user_features = (
    new_tx.groupBy(
        "t_caden_alias", F.window("t_transaction_time", "168 hour", "24 hours")
    )  
    .agg(
        F.mean("t_price").alias("mean_transaction_window_7_day"),  # Aggregation alias
        F.count("*").alias("count_transaction_window_7_day"),
        F.min('u_gender_group').alias("u_gender_group")
    )
    .select(
        "t_caden_alias",
        F.unix_timestamp(F.col("window.end")).cast("timestamp").alias("window_end_ts"),
        F.col("mean_transaction_window_7_day").cast(FloatType()),  # Corrected alias name
        F.col("count_transaction_window_7_day").cast(IntegerType()),
        F.col("u_gender_group"),
    )
)

# COMMAND ----------

new_tx_transactions_ = new_tx_transactions.cache()
prior_order_details = new_tx_transactions_
prior_days = [7, 14, 21]
aggregations = []
for column in ['t_caden_alias']:
  for prior_day in prior_days:
    
    # count distinct instances in the field during this time-range
    aggregations += [
      F.countDistinct(
        F.expr(
          'CASE WHEN (days_prior_to_last_transaction <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('global_cnt_distinct_{1}_last_{0}_days'.format(prior_day, column))]
    
# execute metric definitions
global_metrics = (
  prior_order_details
  ).agg(*aggregations)

# COMMAND ----------



# COMMAND ----------

display(user_features)

# COMMAND ----------



# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Prepare feature DataFrame
def compute_customer_features(data):
  ''' Feature computation code returns a DataFrame with 'customer_id' as primary key'''
  pass

customer_features_df = compute_customer_features(df)

# Create feature table with `customer_id` as the primary key.
# Take schema from DataFrame output by compute_customer_features
customer_feature_table = fe.create_table(
  name='ml.recommender_system.customer_features',
  primary_keys='customer_id',
  schema=customer_features_df.schema,
  description='Customer features'
)

# An alternative is to use `create_table` and specify the `df` argument.
# This code automatically saves the features to the underlying Delta table.

# customer_feature_table = fe.create_table(
#  ...
#  df=customer_features_df,
#  ...
# )

# To use a composite primary key, pass all primary key columns in the create_table call

# customer_feature_table = fe.create_table(
#   ...
#   primary_keys=['customer_id', 'date'],
#   ...
# )

# To create a time series table, set the timeseries_columns argument

# customer_feature_table = fe.create_table(
#   ...
#   primary_keys=['customer_id', 'date'],
#   timeseries_columns='date',
#   ...
# )

def compute_additional_customer_features(data):
  ''' Returns Streaming DataFrame
  '''
  pass

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

customer_transactions = spark.readStream.load("dbfs:/events/customer_transactions")
stream_df = compute_additional_customer_features(customer_transactions)

fe.write_table(
  df=stream_df,
  name='ml.recommender_system.customer_features',
  mode='merge'
)


# COMMAND ----------

display(dbutils.fs.ls("dbfs:/demos/dlt/loans/ling"))

# COMMAND ----------

def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def renameColumns(df):
    """Rename columns to be compatible with Feature Engineering in UC"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# COMMAND ----------

# Run functions
renamed_df = renameColumns()
df = addIdColumn(renamed_df, 'wine_id')

# COMMAND ----------

display(new_tx)

# COMMAND ----------

# DBTITLE 1,Register Transformed Data as Spark DataFrame
from pyspark.sql.types import *

# expected structure of the file
data_schema = StructType([
  StructField('u_caden_alias', StringType()),
  StructField('t_transaction_date', StringType()),
  StructField('t_is_exp', BooleanType()),
  StructField('u_employment_status', StringType()),
  StructField('u_relationship_status', StringType()),
  StructField('u_num_children', StringType()),
  StructField('u_education_level', StringType()),
  StructField('u_age_group', StringType()),
  StructField('u_gender_group', StringType()),
  StructField('u_yearly_household_income_group', StringType()),
  StructField('u_zipcode', StringType()),
  StructField('u_city', StringType()),
  StructField('u_state', StringType()),
  StructField('u_ethnicities', StringType()),
  StructField('t_price', FloatType()),
  StructField('u_income', FloatType())
  ])

# assemble user side dataset with transformed features
trans_features_pd = df[columns]

# send dataset to spark as temp table
spark.createDataFrame(trans_features_pd, data_schema).createOrReplaceTempView('trans_features_pd')


# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

census_income = (spark.read.format("csv") 
    .option("header", "true") 
    .load("dbfs:/tmp/Income By State - Median househome income.csv"))

# COMMAND ----------


