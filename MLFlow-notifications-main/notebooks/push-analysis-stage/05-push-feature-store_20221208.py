# Databricks notebook source
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
# MAGIC ### Load Data

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt')

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

bronze_df = (spark.read 
                 .format("csv")
                 .option("header", True)
                 .option("inferSchema", True)
                 .load("dbfs:/FileStore/shared_uploads/lhuang@themeetgroup.com/0e246b70_3f7f_4279_85f2_38192f58c6ee__1_.csv")
           )

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

display(bronze_df)

# COMMAND ----------

# from pyspark.sql import functions as F
# from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
# from pyspark.sql.window import Window

# most_popular_broadcasters = bronze_df.withColumn("popular_broadcaster"
# .groupBy("from_user_id")\
# .count()\
# .sort("count", ascending=False)
# user = Window().partitionBy('user_id')

# user_log = bronze_df.withColumn('total_push', F.count('*').over(user))\
# .withColumn('total_opens', F.sum('open_flag').over(user))\
# .withColumn('min_dt',  F.from_unixtime((F.min('send_ts').over(user))/1000).cast(DateType()))\
# .withColumn('max_dt', F.from_unixtime((F.max('send_ts').over(user))/1000).cast(DateType()))\
# .withColumn('max_unix', F.max('send_ts').over(user))\
# .select(*['user_id', 'total_push', 'open_flag', 'min_dt', 'max_dt', 'max_unix']).distinct()

# COMMAND ----------

# bronze_df = bronze_df.withColumn('cal_send_ts', F.to_timestamp(F.from_unixtime(F.col('send_ts')/1000))) \
# bronze_df = bronze_df.withColumn('cal_send_ts', F.from_unixtime(F.col('send_ts')/1000, 'yyyy-MM-dd HH:mm:ss')) \
bronze_df = bronze_df.withColumn('cal_send_ts', F.from_unixtime(F.col('send_ts')/1000, 'yyyy-MM-dd-HH')) \
    .withColumn('cal_utc_day_of_week', F.dayofweek(F.from_unixtime(F.col('send_ts')/1000).cast(DateType()))) \
    .withColumn('cal_utc_hour', F.hour(F.from_unixtime(F.col('send_ts')/1000))) \
    .withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id')))

# COMMAND ----------

display(bronze_df)

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

bronze_df.count()

# COMMAND ----------

rows = bronze_df.groupby('open_flag').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Create Feature Table
# MAGIC 
# MAGIC Skip featue store at this point!!

# COMMAND ----------

import uuid

uid = uuid.uuid4().hex[:6]
table_name = f"ml_push._bronze_df_{uid}"

# COMMAND ----------

print(table_name)

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

help(fs.create_table)

# COMMAND ----------

# MAGIC %md <i18n value="52d53b6d-b1c0-4f8e-bb0c-72df4ad7b71f"/>
# MAGIC 
# MAGIC we can create the Feature Table using the **`create_table`** method.
# MAGIC 
# MAGIC This method takes a few parameters as inputs:
# MAGIC * **`name`** - A feature table name of the form **`<database_name>.<table_name>`**
# MAGIC * **`primary_keys`** - The primary key(s). If multiple columns are required, specify a list of column names.
# MAGIC * **`df`** - Data to insert into this feature table.  The schema of **`airbnb_df`** will be used as the feature table schema.
# MAGIC * **`schema`** - Feature table schema. Note that either **`schema`** or **`airbnb_df`** must be provided.
# MAGIC * **`description`** - Description of the feature table
# MAGIC * **`partition_columns`** - Column(s) used to partition the feature table.

# COMMAND ----------

## need to dedupe to generate feature store table
df = bronze_df.drop_duplicates(["network_user_id", "broadcaster_id", "send_ts"])

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["network_user_id", "broadcaster_id", "send_ts"],
    df=df,
    partition_columns="cal_send_ts",
    description="Test Feature Store Table"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch  metadata of the feature store

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

table_name = 'ml_push._bronze_dfb69f1b'
print(f"Feature table description : {fs.get_table(table_name).description}")
print(f"Feature table data source : {fs.get_table(table_name).path_data_sources}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Train a  model

# COMMAND ----------

from pyspark.sql import *
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta
import mlflow.pyfunc

# COMMAND ----------

from databricks.feature_store import FeatureLookup

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from ml_push._bronze_dfb69f1b limit 10

# COMMAND ----------

count_feature_lookups = [
   FeatureLookup( 
     table_name = "ml_push._bronze_dfb69f1b",
     feature_names = ["search_browse", "match_searches", "vpaas_searches", "paas_views", "gift_cnt", "gender", "age_bucket", "country_tier", "device_type"],
     lookup_key = ["network_user_id", "broadcaster_id", "send_ts"],
   ),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Training Dataset

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

bronze_df = bronze_df.select(
    "open_flag",
    "broadcaster_id",
    "cal_utc_day_of_week",
    "cal_utc_hour",
    "network_user_id", 
    "send_ts"
)

# COMMAND ----------

exclude_columns = ["send_ts", 
"event_type",
"event_status",
"notification_type",
"notification_name",
"from_user_id",
"from_user_network",
"cal_send_ts"
]

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
  bronze_df,
  feature_lookups = count_feature_lookups,
  label = "open_flag",
  exclude_columns = exclude_columns
)


# COMMAND ----------

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------

training_df = training_df.withColumn('open_flag', F.col('open_flag').cast(BooleanType()))

for col_name in ["cal_utc_day_of_week", "cal_utc_hour", "country_tier"]:
   training_df = training_df.withColumn(col_name, F.col(col_name).cast(StringType()))

# COMMAND ----------

n1 = training_df.filter(bronze_df.open_flag == 1).count()
n = training_df.count()
r = n1/n * 10
print(r)

# COMMAND ----------

training_df = training_df.sampleBy('open_flag', {False: r , True:1}, seed=28)

# COMMAND ----------

training_df.groupBy('open_flag').count()

# COMMAND ----------

import databricks.automl
summary = databricks.automl.classify(training_df, target_col='open_flag', primary_metric="f1", data_dir='dbfs:/automl/ml_push', timeout_minutes=120)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Skip Feature Store right now!

# COMMAND ----------

# MAGIC %md 
# MAGIC ### LightGBM Training

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

bronze_df = bronze_df.withColumn('open_flag', F.col('open_flag').cast(BooleanType()))

for col_name in ['network_user_id', 'broadcaster_id', 'device_type', 'cal_utc_day_of_week', 'cal_utc_hour', 'age_bucket', 'gender', 'country_tier', 'device_type']:
    bronze_df = bronze_df.withColumn(col_name, F.col(col_name).cast(StringType()))

# COMMAND ----------

display(bronze_df)

# COMMAND ----------

# def decompose(df: pd.DataFrame, label: str, weight: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
#     """break down data into features, labels and weights"""
#     return df.drop([label, weight], axis=1), df[label], df[weight]

# df_train_test, df_val = bronze_df.randomSplit([0.9, 0.1], seed=12345)
# df_train, df_test = df_train_test.randomSplit([0.8, 0.2], seed=12345)

# columns = ['open_flag'] + supported_cols

# df_train_agg = (df_train
# .groupBy(columns)
# .count())

# df_train_pdf = df_train_agg.toPandas()
# df_val_pdf = df_val.toPandas()
# df_test_pdf = df_test.toPandas()

# X_train, y_train, w_train = decompose(df_train_pdf, 'open_flag', 'count')

# y_val = df_val_pdf["open_flag"]
# X_val =  df_val_pdf.drop("open_flag", axis=1)

# y_test = df_test_pdf["open_flag"]
# X_test =  df_test_pdf.drop("open_flag", axis=1)

# categoricals = [
# 'network_user_id', 
# 'broadcaster_id', 
# 'device_type', 
# 'cal_utc_day_of_week', 
# 'cal_utc_hour', 
# 'age_bucket', 
# 'gender',
# 'country_tier', 
# ]
# for c in categoricals:
#     X_train[c] = X_train[c].astype('category')
#     X_val[c] = X_val[c].astype('category')
#     X_test[c] = X_test[c].astype('category')

# params = {
#   "colsample_bytree": 0.514110006702232,
#   "lambda_l1": 0.2519477297842174,
#   "lambda_l2": 97.75994735799596,
#   "learning_rate": 0.7238180156124475,
#   "max_bin": 411,
#    "max_depth": 5,
#   "min_child_samples": 106,
#   "n_estimators": 363,
#   "num_leaves": 459,
#   "path_smooth": 86.24468589325652,
#   "subsample": 0.7230826178445122,
#   "random_state": 750994329,
# }

# with mlflow.start_run(experiment_id="3530617088607082", run_name="lightgbm") as mlflow_run:

#     mlflow.lightgbm.autolog()

#     # Train a lightGBM model
#     model = lgb.LGBMClassifier(**params)
    
#     # model.fit(X=X_train, y=y_train, sample_weight=w_train, categorical_feature=categoricals, eval_set=[(X_test, y_test)], eval_names=["test"])
#     model.fit(X=X_train, y=y_train, sample_weight=w_train, categorical_feature=categoricals)

#     # Log metrics for the training set
#     lgbmc_training_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_train, y_train, prefix="training_")

#     # Log metrics for the test set
#     lgbmc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

#     # Log metrics for the validation set
#     lgbmc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

#     # Display the logged metrics
#     lgbmc_val_metrics = {k.replace("val_", ""): v for k, v in lgbmc_val_metrics.items()}
#     lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}

#     loss = -1 * lgbmc_val_metrics['roc_auc_score']
#     print(lgbmc_val_metrics)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = [
'network_user_id', 
'broadcaster_id', 
'device_type', 
'cal_utc_day_of_week', 
'cal_utc_hour', 
'age_bucket', 
'gender',
'country_tier', 
'search_browse', 
'match_searches', 
'vpaas_searches', 
'paas_views', 
'gift_cnt'
]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), [
'search_browse', 
'match_searches', 
'vpaas_searches', 
'paas_views', 
'gift_cnt'
]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, [
'search_browse', 
'match_searches', 
'vpaas_searches', 
'paas_views', 
'gift_cnt'
])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher


one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, [
'device_type', 
'cal_utc_day_of_week', 
'cal_utc_hour', 
'age_bucket', 
'gender',
'country_tier', 
])]

imputers = {}

categorical_hash_transformers = []

for col in ['broadcaster_id', 'network_user_id']:
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

transformers = categorical_one_hot_transformers + categorical_hash_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split

# COMMAND ----------

df_train_test, df_val = bronze_df.randomSplit([0.9, 0.1], seed=12345)

# COMMAND ----------

df_val.groupBy('open_flag').count().show()

# COMMAND ----------

df_train, df_test = df_train_test.randomSplit([0.8, 0.2], seed=12345)

# COMMAND ----------

columns = ['open_flag'] + supported_cols

df_train_agg = (df_train
.groupBy(columns)
.count())

# COMMAND ----------

df_train_agg.count()

# COMMAND ----------

display(df_train_agg)

# COMMAND ----------

import pandas as pd
from typing import Tuple
import mlflow
import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# COMMAND ----------

df_train_pdf = df_train_agg.toPandas()
df_val_pdf = df_val.toPandas()
df_test_pdf = df_test.toPandas()

# COMMAND ----------

def decompose(df: pd.DataFrame, label: str, weight: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """break down data into features, labels and weights"""
    return df.drop([label, weight], axis=1), df[label], df[weight]

# COMMAND ----------

X_train, y_train, w_train = decompose(df_train_pdf, 'open_flag', 'count')

# COMMAND ----------

y_val = df_val_pdf["open_flag"]
X_val =  df_val_pdf.drop("open_flag", axis=1)

y_test = df_test_pdf["open_flag"]
X_test =  df_test_pdf.drop("open_flag", axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment]
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

import lightgbm
from lightgbm import LGBMClassifier

help(LGBMClassifier)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

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
  with mlflow.start_run(experiment_id="3530617088607082", run_name="lightgbm") as mlflow_run:
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
    # model.fit(X=X_train, y=y_train, sample_weight=w_train, categorical_feature=categoricals)

    
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

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree classifier, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC 
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC 
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html
# MAGIC 
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

space = {
  "colsample_bytree": 0.6122777844378409,
  "lambda_l1": 349.4840449899885,
  "lambda_l2": 987.8208171387618,
  "learning_rate": 0.4619800375311295,
  "max_bin": 56,
  "max_depth": 7,
  "min_child_samples": 55,
  "n_estimators": 416,
  "num_leaves": 4,
  "path_smooth": 5.947790796882757,
  "subsample": 0.6006422492011244,
  "random_state": 275511465,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC 
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC 
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

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

display(pd.DataFrame([best_result["val_metrics"], best_result["test_metrics"]],index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that ML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

# COMMAND ----------

X_train.columns

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train[supported_cols].sample(n=min(100, X_train.shape[0]), random_state=275511465)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val[supported_cols].sample(n=min(100, X_val.shape[0]), random_state=275511465)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=supported_cols))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    # summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------


# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

X_train

# COMMAND ----------


