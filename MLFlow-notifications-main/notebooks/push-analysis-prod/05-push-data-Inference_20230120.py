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

target_col = "open_flag"

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

# COMMAND ----------

bronze_df = spark.sql("select * from ml_push.temp_push_demographics_v2")

bronze_df = bronze_df.withColumn('cal_utc_day_of_week', F.dayofweek(F.from_unixtime(F.col('send_ts')/1000).cast(DateType()))) \
    .withColumn('cal_utc_hour', F.hour(F.from_unixtime(F.col('send_ts')/1000))) \
    .withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id'))) \
    .withColumn('lang', F.substring(F.col('locale'), 1, 2)) \
    .withColumn('from_user_lang', F.substring(F.col('from_user_locale'), 1, 2))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Setup Training Dataset

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

columns = [target_col] + categorical_features + numeric_features

# COMMAND ----------

df_train_test, df_val = silver_df.select(columns).randomSplit([0.9, 0.1], seed=12345)
df_train, df_test = df_train_test.randomSplit([0.8, 0.2], seed=12345)

df_train_agg = (df_train
.groupBy(columns)
.count())

# COMMAND ----------

def decompose(df: pd.DataFrame, label: str, weight: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """break down data into features, labels and weights"""
    return df.drop([label, weight], axis=1), df[label], df[weight]

# COMMAND ----------

df_train_pdf = df_train_agg.toPandas()
df_val_pdf = df_val.toPandas()
df_test_pdf = df_test.toPandas()

# COMMAND ----------

X_train, y_train, w_train = decompose(df_train_pdf, 'open_flag', 'count')

# COMMAND ----------

y_val = df_val_pdf["open_flag"]
X_val =  df_val_pdf.drop("open_flag", axis=1)

y_test = df_test_pdf["open_flag"]
X_test =  df_test_pdf.drop("open_flag", axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Proprocessors

# COMMAND ----------

supported_cols = categorical_features + numeric_features
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Numerical columns
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), numeric_features))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, numeric_features)]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, categorical_features)]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Medium-cardinality categoricals
# MAGIC Convert each medium-cardinality categorical column into a numerical representation.
# MAGIC Each string column is hashed to 1024 float columns.
# MAGIC Each numeric column is imputed with zeros.

# COMMAND ----------

from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

imputers = {
}

categorical_hash_transformers = []

# for col in ["broadcaster_id", "network_user_id"]:
for col in []:
    hasher = FeatureHasher(n_features=1024, input_type="string")
    if col in imputers:
        imputer_name, imputer = imputers[col]
    else:
        imputer_name, imputer = "impute_string_", SimpleImputer(fill_value='', missing_values=None, strategy='constant')
    hash_pipeline = Pipeline(steps=[
        (imputer_name, imputer),
        (f"{col}_hasher", hasher),
    ])
    categorical_hash_transformers.append((f"{col}_pipeline", hash_pipeline, [col]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers + categorical_one_hot_transformers + categorical_hash_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Build a baseline model

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
import pandas as pd

import lightgbm
from lightgbm import LGBMClassifier

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
mlflow.sklearn.autolog(disable=True)
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

def objective(params):
    with mlflow.start_run(experiment_id="3358613615173236", run_name="lightgbm_fulldata") as mlflow_run:
        lgbmc_classifier = LGBMClassifier(**params)

        model = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("classifier", lgbmc_classifier),
        ])

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(
            log_input_examples=True,
            silent=True)

        model.fit(X_train, y_train, classifier__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)], classifier__eval_set=[(X_val_processed,y_val)], classifier__sample_weight=w_train)

        # Log metrics for the training set
        lgbmc_training_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_train, y_train, prefix="training_", sample_weight=w_train)

        # Log metrics for the validation set
        lgbmc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

        # Log metrics for the test set
        lgbmc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

        loss = lgbmc_val_metrics["val_f1_score"]

        # Truncate metric key names so they can be displayed together
        lgbmc_val_metrics = {k.replace("val_", ""): v for k, v in lgbmc_val_metrics.items()}
        lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": lgbmc_val_metrics,
      "test_metrics": lgbmc_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

space = {
  "colsample_bytree": 0.5655989267963984,
  "lambda_l1": 31.125979447297205,
  "lambda_l2": 33.88612024657531,
  "learning_rate": 0.292445826919433,
  "max_bin": 428,
  "max_depth": 9,
  "min_child_samples": 182,
  "n_estimators": 724,
  "num_leaves": 811,
  "path_smooth": 90.97050554368093,
  "subsample": 0.7143732017757635,
  "random_state": 855876001,
}

# COMMAND ----------

trials = Trials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=1,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Register the model in MLflow Model Registry

# COMMAND ----------

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# COMMAND ----------

# model_uri = 'runs:/06986f73a319468e8199ad93cda64a72/model'

# COMMAND ----------

# Assign model name, i.e. best model coming out of pipeline
model_name = "Test-Stage-Model"

# COMMAND ----------

# Register models in the Model Registry
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
        name=model_name,
        version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

# COMMAND ----------

wait_until_ready(model_details.name, model_details.version)

# COMMAND ----------

# Archive the old model version
client.transition_model_version_stage(
  name=model_name,
  version=1,
  stage="Archived"
)

# COMMAND ----------

### Add model and model version descriptions.
client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description="This model predict push open probability."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using Sklearn pipeline with LightGBM."
)

# COMMAND ----------

### Transition a model version and retrieve details 
client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='production',
)
model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

latest_version_info = client.get_latest_versions(model_name, stages=["production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))

# COMMAND ----------

### Load model using registered model name and version
model_version_uri = f"models:/{model_name}/{latest_production_version}".format(model_name=model_name, latest_production_version=latest_production_version)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_latest_version = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

### Load model using production model
model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model_production = mlflow.pyfunc.load_model(model_production_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Inference

# COMMAND ----------

bronze_df = spark.sql("select * from ml_push.temp_push_demographics_v2")
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

# scoring_df_sampled_data = scoring_df.sample(False, 0.000005, 28)

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

dbutils.fs.rm('dbfs:/mnt/mnt/tmg-prod-datalake-outputs/push_data/scoring_by_utc_dow/_delta_log', recurse=True)

# COMMAND ----------

dbutils.fs.ls('dbfs:/mnt/mnt/tmg-prod-datalake-outputs/push_data/scoring_by_utc_dow')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml_push.scoring_group_dow;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE EXTERNAL TABLE IF NOT EXISTS ml_push.scoring_group_dow(
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
# MAGIC LOCATION 'dbfs:/mnt/mnt/tmg-prod-datalake-outputs/push_data/scoring_by_utc_dow';

# COMMAND ----------

for group in scoring_pdf.cal_utc_day_of_week.unique():
    scoring_pdf_group = scoring_pdf[scoring_pdf.cal_utc_day_of_week==group]
    scoring_pdf_group["predicted_proba"]=predict(scoring_pdf_group)
    spark.createDataFrame(scoring_pdf_group) \
    .withColumn("open_flag", F.col("open_flag").cast(StringType())) \
    .write.format("delta").mode("append").option("overwriteSchema", "true").saveAsTable("ml_push.scoring_group_dow") 
    print(group)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Profiling the dataset

# COMMAND ----------

scoring_df = spark.sql("select * from ml_push.scoring_group_dow")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from ml_push.scoring_group_dow;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select open_flag, count(*) from ml_push.scoring_group_dow group by 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select 
# MAGIC percentile_approx(predicted_proba, 0.1) as p_1,
# MAGIC percentile_approx(predicted_proba, 0.2) as p_2,
# MAGIC percentile_approx(predicted_proba, 0.3) as p_3,
# MAGIC percentile_approx(predicted_proba, 0.4) as p_4,
# MAGIC percentile_approx(predicted_proba, 0.5) as p_5,
# MAGIC percentile_approx(predicted_proba, 0.6) as p_6,
# MAGIC percentile_approx(predicted_proba, 0.7) as p_7,
# MAGIC percentile_approx(predicted_proba, 0.8) as p_8,
# MAGIC percentile_approx(predicted_proba, 0.9) as p_9
# MAGIC from ml_push.scoring_group_dow

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
# MAGIC from ml_push.scoring_group_dow
# MAGIC group by 1

# COMMAND ----------

d = {1: 0.0010196, 2: 0.001532, 3: 0.001780, 4: 0.002087, 5: 0.002761, 6: 0.003679, 7: 0.005161, 8: 0.023356, 9: 0.074450}

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
    .sort("decile", ascending=False).show()

# COMMAND ----------

import sys
res = res.withColumn("cumsum_send_push", F.sum("send_push").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0))) \
    .withColumn("cumsum_actual_open", F.sum("actual_open").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0)))\
    .withColumn("cumsum_pred_open", F.sum("pred_open").over(Window.partitionBy().orderBy(F.desc("decile")).rowsBetween(-sys.maxsize, 0)))

res = res.withColumn("cumsum_pred_open", F.round("cumsum_pred_open", 2)) 
res = res.withColumn("cumsum_actual_open %", F.round(res.cumsum_actual_open/res.cumsum_send, 4)) \
    .withColumn("cumsum_pred_open %", F.round(res.cumsum_pred_open/res.cumsum_send, 4)) 

# COMMAND ----------

res.sort('decile', ascending=False).show()

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


