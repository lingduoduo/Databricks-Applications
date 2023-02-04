# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM training

# COMMAND ----------

dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt')

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "open_flag"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
from pyspark.sql.window import Window

import mlflow
import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

# COMMAND ----------

bronze_df = spark.sql("select * from ml_push.l7_push_meetme_source_partitioned where calculated_time = '2022-11-04 19:19:55.364'")
bronze_df = bronze_df.withColumn('utc_day_of_week', F.dayofweek(F.from_unixtime(F.col('send_ts')/1000).cast(DateType()))) \
    .withColumn('utc_hour', F.hour(F.from_unixtime(F.col('send_ts')/1000))) \
    .withColumn('broadcaster_id', F.concat(F.col('from_user_network'), F.lit(':user:'), F.col('from_user_id')))

silver_df = bronze_df.withColumn('open_flag', F.col('open_flag').cast(BooleanType()))
for col_name in ['utc_day_of_week', 'utc_hour']:
    silver_df = silver_df.withColumn(col_name, F.col(col_name).cast(StringType()))

# COMMAND ----------

columns = [
    'open_flag',
    'utc_day_of_week',
    'utc_hour',
    'broadcaster_id',
]

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

categoricals = [
'broadcaster_id',  
'utc_day_of_week', 
'utc_hour', 
]
for c in categoricals:
    X_train[c] = X_train[c].astype('category')
    X_val[c] = X_val[c].astype('category')
    X_test[c] = X_test[c].astype('category')

# COMMAND ----------

params = {
  "colsample_bytree": 0.514110006702232,
  "lambda_l1": 0.2519477297842174,
  "lambda_l2": 97.75994735799596,
  "learning_rate": 0.7238180156124475,
  "max_bin": 411,
  "max_depth": 2,
  "min_child_samples": 106,
  "n_estimators": 363,
  "num_leaves": 459,
  "path_smooth": 86.24468589325652,
  "subsample": 0.7230826178445122,
  "random_state": 750994329,
}

# COMMAND ----------

mlflow.end_run()

with mlflow.start_run(experiment_id="1660525252111638", run_name="lightgbm") as mlflow_run:

    mlflow.lightgbm.autolog()

    # Train a lightGBM model
    model = lgb.LGBMClassifier(**params)
    
    model.fit(X=X_train, y=y_train, sample_weight=w_train, categorical_feature=categoricals)
    
    # Log metrics for the training set
    lgbmc_training_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_train, y_train, prefix="training_")

    # Log metrics for the test set
    lgbmc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Log metrics for the validation set
    lgbmc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

#     loss = lgbmc_val_metrics['val_f1_score']
                             
    # Display the logged metrics
    lgbmc_val_metrics = {k.replace("val_", ""): v for k, v in lgbmc_val_metrics.items()}
    lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}
    print(lgbmc_val_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["broadcaster_id", "utc_day_of_week", "utc_hour"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
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

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["utc_day_of_week", "utc_hour"])]

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

for col in ["broadcaster_id"]:
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

display(X_train)

# COMMAND ----------

display(X_test)

# COMMAND ----------

display(X_val)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/1660525252111638)
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
    with mlflow.start_run(experiment_id="1660525252111638", run_name="lightgbm") as mlflow_run:
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
  "colsample_bytree": 0.514110006702232,
  "lambda_l1": 0.2519477297842174,
  "lambda_l2": 97.75994735799596,
  "learning_rate": 0.7238180156124475,
  "max_bin": 411,
  "max_depth": 2,
  "min_child_samples": 106,
  "n_estimators": 363,
  "num_leaves": 459,
  "path_smooth": 86.24468589325652,
  "subsample": 0.7230826178445122,
  "random_state": 750994329,
}

# COMMAND ----------

objective(space)

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

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
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

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(500, X_train.shape[0]), random_state=750994329)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(500, X_val.shape[0]), random_state=750994329)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC and Precision-Recall curves for validation data
# MAGIC 
# MAGIC We show the confusion matrix, ROC and Precision-Recall curves of the model on the validation data.
# MAGIC 
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Paste the entire output (%md ...) to an empty cell, and click the link to see the MLflow run page
print(f"%md [Link to model run page](#mlflow/experiments/1660525252111638/runs/{ mlflow_run.info.run_id }/artifactPath/model)")

# COMMAND ----------

import uuid
from IPython.display import Image
import os

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve.png")
display(Image(filename=eval_pr_curve_path))

# COMMAND ----------


