# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Readin Training Data

# COMMAND ----------

# import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import max as sparkMax
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder, Normalizer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from sklearn.metrics import roc_curve, auc

import time
import datetime
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```
# MAGIC WITH push AS (
# MAGIC   SELECT
# MAGIC       send_data.*,
# MAGIC       CASE WHEN open_data.network_user_id IS NOT NULL THEN 1 ELSE 0 END open_flag
# MAGIC   FROM
# MAGIC   (
# MAGIC     SELECT DISTINCT
# MAGIC           n.common__timestamp  as send_ts,
# MAGIC           n.network_user_id,
# MAGIC           n.event_type,
# MAGIC           n.event_status,
# MAGIC           n.notification_type,
# MAGIC           n.notification_name,
# MAGIC           n.device_type,
# MAGIC           n.from_user_id,
# MAGIC           n.from_user_network,
# MAGIC             mups.gender,
# MAGIC             mups.age_bucket,
# MAGIC             mups.country_tier
# MAGIC FROM tmg.s_tmg_notification n
# MAGIC LEFT OUTER JOIN tmg.m_user_profile_snapshot mups
# MAGIC   ON n.user_id = mups.user_id
# MAGIC  AND n.network = mups.user_network
# MAGIC WHERE n._partition_date >= date_add('day', -7, current_date)
# MAGIC AND FROM_UNIXTIME(n.common__timestamp/1000) >= (CURRENT_TIMESTAMP - INTERVAL '1' DAY)
# MAGIC AND n.notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC AND n.event_status = 'success'
# MAGIC AND n.network = 'meetme'
# MAGIC AND n.event_type = 'send'
# MAGIC  ) send_data
# MAGIC LEFT JOIN
# MAGIC (
# MAGIC   SELECT DISTINCT
# MAGIC     network_user_id,
# MAGIC     common__timestamp as open_ts
# MAGIC   FROM tmg.s_tmg_notification
# MAGIC   WHERE _partition_date >= date_add('day', -7, current_date)
# MAGIC   AND FROM_UNIXTIME(common__timestamp/1000) >= (CURRENT_TIMESTAMP - INTERVAL '1' DAY)
# MAGIC   AND notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC   AND event_status = 'success'
# MAGIC   AND network = 'meetme'
# MAGIC   AND event_type = 'open'
# MAGIC ) open_data
# MAGIC ON send_data.network_user_id = open_data.network_user_id
# MAGIC AND send_data.send_ts < open_data.open_ts
# MAGIC )
# MAGIC 
# MAGIC select * from push;
# MAGIC 
# MAGIC ```

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/6a6709b4_6dd6_45e0_9cb8_c5ada6cc4a9a.csv"))

# COMMAND ----------

df = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/6a6709b4_6dd6_45e0_9cb8_c5ada6cc4a9a.csv")
           )
 
display(df)

# COMMAND ----------

def shape(data):
    rows, cols = data.count(), len(data.columns)
    shape = (rows, cols)
    return shape

# COMMAND ----------

shape(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Aggregated counts

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC WITH push AS (
# MAGIC   SELECT
# MAGIC       send_data.*,
# MAGIC       CASE WHEN open_data.network_user_id IS NOT NULL THEN 1 ELSE 0 END open_flag
# MAGIC   FROM
# MAGIC   (
# MAGIC     SELECT DISTINCT
# MAGIC           n.common__timestamp  as send_ts,
# MAGIC           n.network_user_id,
# MAGIC           n.event_type,
# MAGIC           n.event_status,
# MAGIC           n.notification_type,
# MAGIC           n.notification_name,
# MAGIC           n.device_type,
# MAGIC           n.from_user_id,
# MAGIC           n.from_user_network,
# MAGIC             mups.gender,
# MAGIC             mups.age_bucket,
# MAGIC             mups.country_tier
# MAGIC FROM tmg.s_tmg_notification n
# MAGIC LEFT OUTER JOIN tmg.m_user_profile_snapshot mups
# MAGIC   ON n.user_id = mups.user_id
# MAGIC  AND n.network = mups.user_network
# MAGIC WHERE n._partition_date >= date_add('day', -7, current_date)
# MAGIC AND FROM_UNIXTIME(n.common__timestamp/1000) >= (CURRENT_TIMESTAMP - INTERVAL '1' DAY)
# MAGIC AND n.notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC AND n.event_status = 'success'
# MAGIC AND n.network = 'meetme'
# MAGIC AND n.event_type = 'send'
# MAGIC  ) send_data
# MAGIC LEFT JOIN
# MAGIC (
# MAGIC   SELECT DISTINCT
# MAGIC     network_user_id,
# MAGIC     common__timestamp as open_ts
# MAGIC   FROM tmg.s_tmg_notification
# MAGIC   WHERE _partition_date >= date_add('day', -7, current_date)
# MAGIC   AND FROM_UNIXTIME(common__timestamp/1000) >= (CURRENT_TIMESTAMP - INTERVAL '1' DAY)
# MAGIC   AND notification_name in ('broadcastStartWithDescriptionPush', 'broadcastStartPush')
# MAGIC   AND event_status = 'success'
# MAGIC   AND network = 'meetme'
# MAGIC   AND event_type = 'open'
# MAGIC ) open_data
# MAGIC ON send_data.network_user_id = open_data.network_user_id
# MAGIC AND send_data.send_ts < open_data.open_ts
# MAGIC ),
# MAGIC 
# MAGIC 
# MAGIC search_browse as (
# MAGIC SELECT common__subject,
# MAGIC     common__timestamp AS search_ts
# MAGIC FROM tmg.s_tmg_search_user_browse
# MAGIC WHERE partition_date >= (current_date - INTERVAL '2' DAY)
# MAGIC     AND SPLIT_PART(common__subject, ':', 1) = 'meetme'
# MAGIC ),
# MAGIC 
# MAGIC join_search_browse  as (
# MAGIC SELECT
# MAGIC     network_user_id,
# MAGIC     send_ts,
# MAGIC     COUNT(*) search_browse
# MAGIC FROM (
# MAGIC     SELECT
# MAGIC         push.network_user_id,
# MAGIC         push.send_ts,
# MAGIC         search_browse.search_ts
# MAGIC     FROM push JOIN search_browse
# MAGIC     ON push.network_user_id = search_browse.common__subject
# MAGIC     WHERE date_diff('second', FROM_UNIXTIME(search_browse.search_ts/1000), FROM_UNIXTIME(push.send_ts/1000)) <= 86400
# MAGIC     AND search_browse.search_ts < push.send_ts
# MAGIC     GROUP BY 1, 2, 3
# MAGIC )
# MAGIC GROUP BY 1, 2
# MAGIC ),
# MAGIC 
# MAGIC match_searches AS (
# MAGIC SELECT common__subject,
# MAGIC     common__timestamp AS match_ts
# MAGIC FROM tmg.s_tmg_search_user_match
# MAGIC WHERE partition_date >= (current_date - INTERVAL '2' DAY)
# MAGIC     AND SPLIT_PART(common__subject, ':', 1) = 'meetme'
# MAGIC ),
# MAGIC 
# MAGIC join_match_searches  as (
# MAGIC SELECT
# MAGIC     network_user_id,
# MAGIC     send_ts,
# MAGIC     COUNT(*) match_searches
# MAGIC FROM (
# MAGIC     SELECT
# MAGIC         push.network_user_id,
# MAGIC         push.send_ts,
# MAGIC         match_searches.match_ts
# MAGIC     FROM push JOIN match_searches
# MAGIC     ON push.network_user_id = match_searches.common__subject
# MAGIC     WHERE date_diff('second', FROM_UNIXTIME(match_searches.match_ts/1000), FROM_UNIXTIME(push.send_ts/1000)) <= 86400
# MAGIC     AND match_searches.match_ts < push.send_ts
# MAGIC     GROUP BY 1, 2, 3
# MAGIC )
# MAGIC GROUP BY 1, 2
# MAGIC ),
# MAGIC 
# MAGIC vpaas_searches
# MAGIC AS (SELECT common__subject,
# MAGIC         common__timestamp as vpaas_searches_ts
# MAGIC    FROM tmg.s_tmg_search_broadcast
# MAGIC   WHERE partition_date >= (current_date - INTERVAL '2' DAY)
# MAGIC         AND SPLIT_PART(common__subject, ':', 1) = 'meetme'
# MAGIC ),
# MAGIC 
# MAGIC join_vpaas_searches  as (
# MAGIC SELECT
# MAGIC     network_user_id,
# MAGIC     send_ts,
# MAGIC     COUNT(*) vpaas_searches
# MAGIC FROM (
# MAGIC     SELECT
# MAGIC         push.network_user_id,
# MAGIC         push.send_ts,
# MAGIC         vpaas_searches.vpaas_searches_ts
# MAGIC     FROM push JOIN vpaas_searches
# MAGIC     ON push.network_user_id = vpaas_searches.common__subject
# MAGIC     WHERE date_diff('second', FROM_UNIXTIME(vpaas_searches.vpaas_searches_ts/1000), FROM_UNIXTIME(push.send_ts/1000)) <= 86400
# MAGIC     AND vpaas_searches.vpaas_searches_ts < push.send_ts
# MAGIC     GROUP BY 1, 2, 3
# MAGIC )
# MAGIC GROUP BY 1, 2
# MAGIC ),
# MAGIC 
# MAGIC vpaas_views
# MAGIC AS (SELECT network_user_id,
# MAGIC         body_publishedtime AS vpaas_views_ts
# MAGIC    FROM tmg.ml_live_broadcast_end_views
# MAGIC   WHERE partition_time  >= (current_date - INTERVAL '2' DAY)
# MAGIC        AND sns_network = 'meetme'
# MAGIC ),
# MAGIC 
# MAGIC join_vpaas_views  as (
# MAGIC SELECT
# MAGIC     network_user_id,
# MAGIC     send_ts,
# MAGIC     COUNT(*) vpaas_views
# MAGIC FROM (
# MAGIC     SELECT
# MAGIC         push.network_user_id,
# MAGIC         push.send_ts,
# MAGIC         vpaas_views.vpaas_views_ts
# MAGIC     FROM push JOIN vpaas_views
# MAGIC     ON push.network_user_id = vpaas_views.network_user_id
# MAGIC     WHERE date_diff('second', FROM_UNIXTIME(vpaas_views.vpaas_views_ts), FROM_UNIXTIME(push.send_ts/1000)) <= 86400
# MAGIC     AND date_diff('second', FROM_UNIXTIME(vpaas_views.vpaas_views_ts), FROM_UNIXTIME(push.send_ts/1000)) > 0
# MAGIC     GROUP BY 1, 2, 3
# MAGIC )
# MAGIC GROUP BY 1, 2
# MAGIC ),
# MAGIC 
# MAGIC econ as (
# MAGIC     SELECT
# MAGIC         body_order_purchaser,
# MAGIC 		body_order_orderdate
# MAGIC     FROM tmg.s_tmg_economy_order_fulfilled
# MAGIC     WHERE partition_time >= date_add('day', -2, CURRENT_DATE)
# MAGIC       AND split_part(body_order_reference, ':', 1) = 'broadcast'   -- broadcast dmds only
# MAGIC       AND data_body_order_products_product_exchange_currency = 'DMD' -- only DMDS
# MAGIC       AND split_part(body_order_purchaser, ':', 1) = 'meetme'
# MAGIC ),
# MAGIC 
# MAGIC join_econ  as (
# MAGIC SELECT
# MAGIC     network_user_id,
# MAGIC     send_ts,
# MAGIC     COUNT(*) gift_cnt
# MAGIC FROM (
# MAGIC     SELECT
# MAGIC         push.network_user_id,
# MAGIC         push.send_ts,
# MAGIC         econ.body_order_orderdate
# MAGIC     FROM push JOIN econ
# MAGIC     ON push.network_user_id = econ.body_order_purchaser
# MAGIC     WHERE date_diff('second', econ.body_order_orderdate, FROM_UNIXTIME(push.send_ts/1000)) <= 86400
# MAGIC     AND date_diff('second', econ.body_order_orderdate, FROM_UNIXTIME(push.send_ts/1000)) > 0
# MAGIC     GROUP BY 1, 2, 3
# MAGIC )
# MAGIC GROUP BY 1, 2
# MAGIC ),
# MAGIC 
# MAGIC push_agg_cnt AS (
# MAGIC SELECT push.*,
# MAGIC        CASE WHEN join_search_browse.search_browse IS NOT NULL THEN join_search_browse.search_browse ELSE 0 END search_browse,
# MAGIC        CASE WHEN join_match_searches.match_searches IS NOT NULL THEN join_match_searches.match_searches ELSE 0 END match_searches,
# MAGIC        CASE WHEN join_vpaas_searches.vpaas_searches IS NOT NULL THEN join_vpaas_searches.vpaas_searches ELSE 0 END vpaas_searches,
# MAGIC        CASE WHEN join_vpaas_views.vpaas_views IS NOT NULL THEN join_vpaas_views.vpaas_views ELSE 0 END vpaas_views,
# MAGIC        CASE WHEN join_econ.gift_cnt IS NOT NULL THEN join_econ.gift_cnt ELSE 0 END gift_cnt
# MAGIC FROM push
# MAGIC LEFT JOIN join_search_browse ON push.network_user_id = join_search_browse.network_user_id AND push.send_ts = join_search_browse.send_ts
# MAGIC LEFT JOIN join_match_searches ON push.network_user_id = join_match_searches.network_user_id AND push.send_ts = join_match_searches.send_ts
# MAGIC LEFT JOIN join_vpaas_searches ON push.network_user_id = join_vpaas_searches.network_user_id AND push.send_ts = join_vpaas_searches.send_ts
# MAGIC LEFT JOIN join_vpaas_views ON push.network_user_id = join_vpaas_views.network_user_id AND push.send_ts = join_vpaas_views.send_ts
# MAGIC LEFT JOIN join_econ ON push.network_user_id = join_econ.network_user_id AND push.send_ts = join_econ.send_ts
# MAGIC ),
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM push_agg_cnt
# MAGIC ```

# COMMAND ----------

df = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/0e246b70_3f7f_4279_85f2_38192f58c6ee__1_.csv")
           )

# COMMAND ----------

shape(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data Exploration

# COMMAND ----------

df = df.withColumn('cal_date', F.from_unixtime(F.col('send_ts')/1000).cast(DateType())) \
    .withColumn('cal_utc_day_of_week', F.dayofweek(F.from_unixtime(F.col('send_ts')/1000).cast(DateType()))) \
    .withColumn('cal_utc_hour', F.hour(F.from_unixtime(F.col('send_ts')/1000))) \
    .withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id')))

# COMMAND ----------

df.show(1, vertical=True)

# COMMAND ----------

def count_dist(df, field):
    return df.select(field).distinct().count()

# COMMAND ----------

count_dist(df, 'network_user_id')

# COMMAND ----------

df.groupBy('network_user_id').count().sort('count', ascending=False).show()

# COMMAND ----------

count_dist(df, 'broadcaster_id')

# COMMAND ----------

df.groupBy('broadcaster_id').count().sort('count', ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Categorical Features

# COMMAND ----------

categorical_features = [
'device_type',
'gender',
'age_bucket',
'country_tier',
'cal_date',
'cal_utc_hour',
'cal_utc_day_of_week'
]

# COMMAND ----------

# DBTITLE 1,Summary Stats
for c in categorical_features:
    df.groupBy(c).agg(
        F.count('network_user_id').alias('sends'),
        F.countDistinct('network_user_id').alias('recipients'),
        F.count('network_user_id')/F.countDistinct('network_user_id').alias('push per recipient'),
        F.sum('open_flag').alias('opens'),
        F.sum('open_flag')/F.count('network_user_id').alias('open %')
    ).show()

# COMMAND ----------

numeric_features = [
    'search_browse',
    'match_searches',
    'vpaas_searches',
    'paas_views',
    'gift_cnt'
]

# COMMAND ----------

for c in numeric_features:
    df.groupBy('open_flag').agg(
    F.avg(c).alias(c + '_avg'),
    F.min(c).alias(c + '_min'),
    F.max(c).alias(c + '_max'),
    F.percentile_approx(c, 0.25).alias(c + '_q_25'),
    F.percentile_approx(c, 0.5).alias(c + '_median'),
    F.percentile_approx(c, 0.75).alias(c + '_q_75')
).show()

# COMMAND ----------

# Bar chart
display(df.select('gender', 'age_bucket', 'network_user_id')
          .groupby('gender', 'age_bucket')
          .agg(F.count('network_user_id').alias('sends')))

# COMMAND ----------

df.filter((F.col('search_browse') == 0) & (F.col('match_searches') == 0) & (F.col('vpaas_searches') == 0) & (F.col('paas_views') == 0) & (F.col('gift_cnt') == 0)).count()

bronze_df.filter((F.col('search_browse') == 0) & (F.col('match_searches') == 0) & (F.col('vpaas_searches') == 0) & (F.col('paas_views') == 0) & (F.col('gift_cnt') == 0)) \
.groupBy('open_flag').count().show()
