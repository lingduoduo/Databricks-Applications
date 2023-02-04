# Databricks notebook source
# MAGIC %md 
# MAGIC ### Setup Bronze Table Connection

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt')

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
from pyspark.sql.window import Window

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
 
import pandas as pd
from typing import Tuple
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector

from sklearn.model_selection import train_test_split

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Inference

# COMMAND ----------

bronze_df = spark.sql("select * from ml_push.push_demographics_v4")

bronze_df = bronze_df.withColumn('cal_utc_day_of_week', F.dayofweek(F.from_unixtime(F.col('send_ts')/1000).cast(DateType()))) \
    .withColumn('cal_utc_hour', F.hour(F.from_unixtime(F.col('send_ts')/1000))) \
    .withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id'))) \
    .withColumn('lang', F.substring(F.col('locale'), 1, 2)) \
    .withColumn('from_user_lang', F.substring(F.col('from_user_locale'), 1, 2))

# COMMAND ----------

silver_df = bronze_df.withColumn(target_col, F.col(target_col).cast(BooleanType()))

categorical_features = [
    'device_type',
    'gender',
    'age',
    'country',
    'lang',
    'from_user_age',
    'from_user_gender',
    'from_user_country',
    'from_user_lang',
    'cal_utc_hour',
    'cal_utc_day_of_week'
]

for col_name in categorical_features:
    silver_df = silver_df.withColumn(col_name, F.col(col_name).cast(StringType()))

silver_df = silver_df.fillna(value='missing', subset=categorical_features)

numeric_features = [
    'broadcast_search_count_24h',
    'search_user_count_24h',
    'search_match_count_24h',
    'view_count_24h',
    'view_end_count_24h',
    'gift_count_24h',
]
columns = [target_col] +  ['network_user_id', 'broadcaster_id'] + categorical_features + numeric_features 

scoring_df = silver_df.select(columns)

# COMMAND ----------

cnt = scoring_df.groupby(['network_user_id', 'broadcaster_id']).count()
cnt.count()

# COMMAND ----------

scoring_pdf = scoring_df.toPandas()

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

model_name = "Test-Stage-Model"
model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(new_model_version)

# COMMAND ----------

latest_version_info = client.get_latest_versions(model_name, stages=['Production'])
latest_stage_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_stage_version))

# COMMAND ----------

### Load model using production model
model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model = mlflow.sklearn.load_model(model_production_uri)

# COMMAND ----------

predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=categorical_features + numeric_features))[:,1]

# COMMAND ----------

dbutils.fs.rm('dbfs:/mnt/mnt/tmg-prod-datalake-outputs/push_data/scoring_group_dow_v4', recurse=True)
dbutils.fs.rm('dbfs:/mnt/mnt/tmg-prod-datalake-outputs/push_data/scoring_group_dow_v4/_delta_log', recurse=True)

# COMMAND ----------

dbutils.fs.ls('dbfs:/mnt/mnt/tmg-prod-datalake-outputs/push_data/scoring_group_dow_v4')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml_push.scoring_group_dow_v4;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE EXTERNAL TABLE IF NOT EXISTS ml_push.scoring_group_dow_v4(
# MAGIC     open_flag   STRING,
# MAGIC     network_user_id STRING,
# MAGIC     broadcaster_id STRING,
# MAGIC     device_type  STRING,
# MAGIC     gender STRING,
# MAGIC     age STRING,
# MAGIC     country STRING,
# MAGIC     lang  STRING,
# MAGIC     from_user_age STRING,
# MAGIC     from_user_gender STRING,
# MAGIC     from_user_country  STRING,
# MAGIC     from_user_lang  STRING,
# MAGIC     cal_utc_hour  STRING,
# MAGIC     cal_utc_day_of_week  STRING,
# MAGIC     broadcast_search_count_24h BIGINT,
# MAGIC     search_user_count_24h BIGINT,
# MAGIC     search_match_count_24h BIGINT,
# MAGIC     view_count_24h BIGINT,
# MAGIC     view_end_count_24h BIGINT,
# MAGIC     gift_count_24h BIGINT,
# MAGIC     predicted_proba DOUBLE )
# MAGIC USING DELTA
# MAGIC   PARTITIONED BY (cal_utc_day_of_week)
# MAGIC LOCATION 'dbfs:/mnt/mnt/tmg-prod-datalake-outputs/push_data/scoring_group_dow_v4';

# COMMAND ----------

for group in scoring_pdf.cal_utc_day_of_week.unique():
    scoring_pdf_group = scoring_pdf[scoring_pdf.cal_utc_day_of_week==group]
    scoring_pdf_group["predicted_proba"]=predict(scoring_pdf_group)
    spark.createDataFrame(scoring_pdf_group) \
    .withColumn("open_flag", F.col("open_flag").cast(StringType())) \
    .write.format("delta").mode("append").option("overwriteSchema", "true").saveAsTable("ml_push.scoring_group_dow_v4") 
    print(group)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Derive decile from the training dataset

# COMMAND ----------

scoring_df = spark.sql("select * from ml_push.scoring_group_dow_v4");

# COMMAND ----------

df = spark.sql("select \
        percentile_approx(predicted_proba, 0.1) as p_1, \
        percentile_approx(predicted_proba, 0.2) as p_2, \
        percentile_approx(predicted_proba, 0.3) as p_3, \
        percentile_approx(predicted_proba, 0.4) as p_4, \
        percentile_approx(predicted_proba, 0.5) as p_5, \
        percentile_approx(predicted_proba, 0.6) as p_6, \
        percentile_approx(predicted_proba, 0.7) as p_7, \
        percentile_approx(predicted_proba, 0.8) as p_8, \
        percentile_approx(predicted_proba, 0.9) as p_9 \
        from ml_push.scoring_group_dow_v4");
dpf = df.toPandas().T
dpf.index = [i for i in range(1,10)]
d = dpf.to_dict()[0]

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select 
# MAGIC open_flag,
# MAGIC percentile_approx(predicted_proba, 0.1) as p_1,
# MAGIC percentile_approx(predicted_proba, 0.2) as p_2,
# MAGIC percentile_approx(predicted_proba, 0.3) as p_3,
# MAGIC percentile_approx(predicted_proba, 0.4) as p_4,
# MAGIC percentile_approx(predicted_proba, 0.5) as p_5,
# MAGIC percentile_approx(predicted_proba, 0.6) as p_6,
# MAGIC percentile_approx(predicted_proba, 0.7) as p_7,
# MAGIC percentile_approx(predicted_proba, 0.8) as p_8,
# MAGIC percentile_approx(predicted_proba, 0.9) as p_9
# MAGIC from ml_push.scoring_group_dow_v4
# MAGIC group by 1

# COMMAND ----------

 scoring_df = scoring_df.withColumn('decile', F.when(F.col('predicted_proba') > d[9], 9) \
                            .when(F.col('predicted_proba') > d[8], 8)\
                            .when(F.col('predicted_proba') > d[7], 7) \
                            .when(F.col('predicted_proba') > d[6], 6) \
                            .when(F.col('predicted_proba') > d[5], 5) \
                            .when(F.col('predicted_proba') > d[4], 4) \
                            .when(F.col('predicted_proba') > d[3], 3) \
                            .when(F.col('predicted_proba') > d[2], 2) \
                            .when(F.col('predicted_proba') > d[1], 1) \
                            .otherwise(0)
                           )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Profile training by decile

# COMMAND ----------

res = scoring_df.groupBy('decile') \
    .agg(
        F.round(F.min("predicted_proba"), 6).alias("min_score"),  \
        F.round(F.avg("predicted_proba"), 6).alias("avg_score"), \
        F.round(F.max("predicted_proba"),  6).alias("max_score"), \
        F.count("predicted_proba").alias("send_push"), \
        F.sum(F.when(F.col('open_flag') == True, 1)).alias("actual_open"), \
        F.round(F.sum("predicted_proba"),  1).alias("pred_open"), \
) 

# COMMAND ----------

res.withColumn("actual open %", F.round(res.actual_open/res.send_push, 6)) \
    .withColumn("pred open %", F.round(res.pred_open/res.send_push, 6)) \
     .sort("decile", ascending=False)

# COMMAND ----------

res = res.withColumn("cumsum_send_push", F.sum("send_push").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0))) \
    .withColumn("cumsum_actual_open", F.sum("actual_open").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0)))\
    .withColumn("cumsum_pred_open", F.sum("pred_open").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0)))

# COMMAND ----------

res = res.withColumn("cumsum_pred_open", F.round("cumsum_pred_open", 2)) 
res = res.withColumn("cumsum_actual_open %", F.round(res.cumsum_actual_open/res.cumsum_send_push, 4)) \
    .withColumn("cumsum_pred_open %", F.round(res.cumsum_pred_open/res.cumsum_send_push, 4)) 

# COMMAND ----------

categorical_features = [
    'device_type',
    'gender',
    'age',
    'country',
    'lang',
    'from_user_age',
    'from_user_gender',
    'from_user_country',
    'from_user_lang',
    'cal_utc_hour',
    'cal_utc_day_of_week'
]

# COMMAND ----------

for c in ['device_type', 'gender', 'cal_utc_day_of_week', 'cal_utc_hour']:
    res = scoring_df.groupBy('decile', c) \
    .agg(
        F.round(F.min("predicted_proba"), 6).alias("min_score"),  \
        F.round(F.avg("predicted_proba"), 6).alias("avg_score"), \
        F.round(F.max("predicted_proba"),  6).alias("max_score"), \
        F.count("predicted_proba").alias("send_push"), \
        F.sum(F.when(F.col('open_flag') == True, 1)).alias("actual_open"), \
        F.round(F.sum("predicted_proba"),  1).alias("pred_open"), \
    ) 
    res.withColumn("actual open %", F.round(res.actual_open/res.send_push, 6)) \
    .withColumn("pred open %", F.round(res.pred_open/res.send_push, 6)) \
    .sort(["decile",c], ascending=False).show()

# COMMAND ----------

numeric_features = [
    'broadcast_search_count_24h',
    'search_user_count_24h',
    'search_match_count_24h',
    'view_count_24h',
    'view_end_count_24h',
    'gift_count_24h',
]

# COMMAND ----------

res = scoring_df.groupBy('decile') \
    .agg(
        F.round(F.min("predicted_proba"), 6).alias("min_score"),  \
        F.round(F.avg("predicted_proba"), 6).alias("avg_score"), \
        F.round(F.max("predicted_proba"),  6).alias("max_score"), \
        F.count("predicted_proba").alias("send_push"), \
        F.sum(F.when(F.col('open_flag') == True, 1)).alias("actual_open"), \
        F.round(F.sum("predicted_proba"),  1).alias("pred_open"), \
        F.round(F.avg('broadcast_search_count_24h'),  1).alias('broadcast_search_count_24h'), \
        F.round(F.avg('search_user_count_24h'),  1).alias('search_user_count_24h'), \
        F.round(F.avg('search_match_count_24h'),  1).alias('search_match_count_24h'), \
        F.round(F.avg('view_count_24h'),  1).alias('view_count_24h'), \
        F.round(F.avg('view_end_count_24h'),  1).alias('view_end_count_24h'), \
        F.round(F.avg('gift_count_24h'),  1).alias('gift_count_24h'), \
) 

# COMMAND ----------

res.sort('decile', ascending=False)

# COMMAND ----------

def count_dist(df, field):
    return df.select(field).distinct().count()

# COMMAND ----------

# MAGIC %sql
# MAGIC select open_flag, count(*), count(distinct network_user_id) from ml_push.scoring_group_dow_v4 group by 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct network_user_id) from ml_push.scoring_group_dow_v4;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- retained users using threshold = 0.004;
# MAGIC select count(*), count(distinct network_user_id) from ml_push.scoring_group_dow_v4 where predicted_proba >=0.004;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- retained users
# MAGIC select count(*), count(distinct network_user_id) from ml_push.scoring_group_dow_v4 where predicted_proba < 0.004 and open_flag = True;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Profiling the new dataset

# COMMAND ----------

newdata_df = spark.sql("select * from ml_push.push_demographics_offline_eval")

# push_demographics_offline_eval
newdata_df = newdata_df.withColumn('cal_utc_day_of_week', F.dayofweek(F.from_unixtime(F.col('send_ts')/1000).cast(DateType()))) \
    .withColumn('cal_utc_hour', F.hour(F.from_unixtime(F.col('send_ts')/1000))) \
    .withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id'))) \
    .withColumn('lang', F.substring(F.col('locale'), 1, 2)) \
    .withColumn('from_user_lang', F.substring(F.col('from_user_locale'), 1, 2))

# COMMAND ----------

newdata_df=newdata_df.withColumn('send_date_1', F.to_date(F.from_unixtime(F.col('utc_hour')/1000)))
newdata_df=newdata_df.filter("send_date_1='2023-01-17'")
display(newdata_df)

# COMMAND ----------

target_col = 'open_flag'

categorical_features = [
    'device_type',
    'gender',
    'age',
    'country',
    'lang',
    'from_user_age',
    'from_user_gender',
    'from_user_country',
    'from_user_lang',
    'cal_utc_hour',
    'cal_utc_day_of_week'
]

for col_name in categorical_features:
    newdata_df = newdata_df.withColumn(col_name, F.col(col_name).cast(StringType()))

newdata_df = newdata_df.fillna(value='missing', subset=categorical_features)

numeric_features = [
    'broadcast_search_count_24h',
    'search_user_count_24h',
    'search_match_count_24h',
    'view_count_24h',
    'view_end_count_24h',
    'gift_count_24h',
]

columns = [target_col] + categorical_features + numeric_features 

# newdata_df = newdata_df.select(columns)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

model_name = "Test-Stage-Model"
model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(new_model_version)

# COMMAND ----------

# load input data table as a Spark DataFrame
model_production_uri = f"models:/{model_name}/production".format(model_name=model_name)
print(model_production_uri)

# COMMAND ----------

model = mlflow.sklearn.load_model(model_production_uri) 
predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=categorical_features + numeric_features))[:,1]
newdata_pdf = newdata_df.toPandas()

newdata_pdf["predicted_proba"] = predict(newdata_pdf)
newdata_df = spark.createDataFrame(newdata_pdf)

# COMMAND ----------

from sklearn.metrics import log_loss, roc_auc_score

# COMMAND ----------

 roc_auc_score(newdata_pdf['open_flag'], newdata_pdf['predicted_proba'])

# COMMAND ----------

log_loss(newdata_pdf['open_flag'], newdata_pdf['predicted_proba'])

# COMMAND ----------

newdata_df = newdata_df.withColumn('decile', F.when(F.col('predicted_proba') > d[9], 9) \
                            .when(F.col('predicted_proba') > d[8], 8)\
                            .when(F.col('predicted_proba') > d[7], 7) \
                            .when(F.col('predicted_proba') > d[6], 6) \
                            .when(F.col('predicted_proba') > d[5], 5) \
                            .when(F.col('predicted_proba') > d[4], 4) \
                            .when(F.col('predicted_proba') > d[3], 3) \
                            .when(F.col('predicted_proba') > d[2], 2) \
                            .when(F.col('predicted_proba') > d[1], 1) \
                            .otherwise(0)
                           )

# COMMAND ----------

res = newdata_df.groupBy('decile') \
    .agg(
        F.round(F.min("predicted_proba"), 6).alias("min_score"),  \
        F.round(F.avg("predicted_proba"), 6).alias("avg_score"), \
        F.round(F.max("predicted_proba"),  6).alias("max_score"), \
        F.count("predicted_proba").alias("send_push"), \
        F.sum(F.when(F.col('open_flag') == True, 1)).alias("actual_open"), \
        F.round(F.sum("predicted_proba"),  1).alias("pred_open"), \
) 

# COMMAND ----------

import sys
res = res.withColumn("cumsum_send_push", F.sum("send_push").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0))) \
    .withColumn("cumsum_actual_open", F.sum("actual_open").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0)))\
    .withColumn("cumsum_pred_open", F.sum("pred_open").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0)))

# COMMAND ----------

res = res.withColumn("cumsum_pred_open", F.round("cumsum_pred_open", 2)) 
res = res.withColumn("cumsum_actual_open %", F.round(res.cumsum_actual_open/res.cumsum_send_push, 4)) \
    .withColumn("cumsum_pred_open %", F.round(res.cumsum_pred_open/res.cumsum_send_push, 4)) 

# COMMAND ----------

display(res.sort('decile', ascending=False))

# COMMAND ----------

newdata_df.groupby('network_user_id').count().count()

# COMMAND ----------

newdata_df.groupby('open_flag').agg( \
  F.count('network_user_id').alias('push_count'), \
  F.countDistinct('network_user_id').alias('user_count')).show()

# COMMAND ----------

newdata_df.filter(newdata_df.predicted_proba >=0.004).groupby('open_flag').agg( \
  F.count('network_user_id').alias('push_count'), \
  F.countDistinct('network_user_id').alias('user_count')).show()

# COMMAND ----------

print("total unique open users: {}".format(newdata_df.filter(newdata_df.open_flag == 0).select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique open users: {}".format(newdata_df.filter(newdata_df.open_flag == 1).select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique users: {}".format(newdata_df.select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique users after dropped pushes: {}".format(newdata_df.filter((newdata_df.predicted_proba >= 0.004) & (newdata_df.open_flag == 0)).select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique open users after dropped pushes: {}".format(newdata_df.filter((newdata_df.predicted_proba >= 0.004) & (newdata_df.open_flag == 1)).select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique open users after dropped pushes: {}".format(newdata_df.filter(newdata_df.predicted_proba >= 0.004).select('network_user_id').distinct().count()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Dedupe push and use correlation_id

# COMMAND ----------

# drop device_type (android, ios, web)
status = newdata_df.withColumn('first_record', F.row_number()\
                              .over(Window.partitionBy('source_correlation_id', 'send_ts', 'network_user_id', 'broadcaster_id').orderBy('device_type')))

# COMMAND ----------

send_dedupe = status.filter(status.first_record == 1)

# COMMAND ----------

open_dedupe = status.filter((status.first_record == 1) & (F.col('open_ts').isNotNull())) \
    .select('network_user_id', 'source_correlation_id') \
    .dropDuplicates() \
    .withColumn('correlation_flag', F.lit(1))

# COMMAND ----------

send_dedupe_all = send_dedupe.join(open_dedupe, [send_dedupe.source_correlation_id == open_dedupe.source_correlation_id, send_dedupe.network_user_id == open_dedupe.network_user_id], how='left') \
    .select(send_dedupe['*'], 'correlation_flag') \
    .fillna(value=0, subset=['correlation_flag'])

# COMMAND ----------

send_dedupe_all.groupby('network_user_id').count().count()

# COMMAND ----------

send_dedupe_all.groupby('correlation_flag').agg( \
    F.count('network_user_id').alias('dedupe_push_count'), \
    F.countDistinct('network_user_id').alias('user_count')).show()

# COMMAND ----------

send_dedupe_all.filter(send_dedupe_all.predicted_proba >=0.004).groupby('correlation_flag').agg( \
  F.count('network_user_id').alias('dedupe_push_count'), \
  F.countDistinct('network_user_id').alias('user_count')).show()

# COMMAND ----------

print("total records: {}".format(send_dedupe_all.filter(send_dedupe_all.predicted_proba > 0.004).select('source_correlation_id', 'network_user_id').distinct().count()))

# COMMAND ----------

print("total unique open users: {}".format(send_dedupe_all.filter(send_dedupe_all.correlation_flag == 1).select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique users: {}".format(send_dedupe_all.select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique users after dropped pushes: {}".format(send_dedupe_all.filter((send_dedupe_all.predicted_proba >= 0.004) & (send_dedupe_all.correlation_flag == 0)).select('network_user_id').distinct().count()))

# COMMAND ----------

print("total unique open users after dropped pushes: {}".format(send_dedupe_all.filter(send_dedupe_all.predicted_proba >= 0.004).select('network_user_id').distinct().count()))
