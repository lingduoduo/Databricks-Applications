# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Part 1. Train a classification model

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

df_train = spark.table("ml_push.silver_l7_push_meetme_train")
df_val = spark.table("ml_push.silver_l7_push_meetme_val")

# COMMAND ----------

df_train.printSchema()
df_val.printSchema()

# COMMAND ----------

df_train.show(5)

# COMMAND ----------

df_val.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Select supported columns

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["broadcaster_id", "device_type", "utc_day_of_week", "utc_hour"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, sparse_threshold=0, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers.append(("onehot", one_hot_pipeline, ["broadcaster_id", "device_type", "utc_day_of_week", "utc_hour"]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Feature standardization

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Convert spark Train and Eval datasets to pandas

# COMMAND ----------

target_col = "open_flag"

# COMMAND ----------

df_train = df_train.toPandas()
df_val = df_val.toPandas()

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_train.drop([target_col], axis=1)
split_y = df_train[target_col]

# Split out train data
X_train, X_test, y_train, y_test = train_test_split(split_X, split_y, train_size=0.8, random_state=149849802, stratify=split_y)

# COMMAND ----------

X_val = df_val.drop([target_col], axis=1)
y_val = df_val[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training and Testing
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under the MLflow experiment
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from imblearn.pipeline import make_pipeline

set_config(display="diagram")

# COMMAND ----------

xgbc_classifier = XGBClassifier(
  colsample_bytree=0.49562189236760895,
  learning_rate=0.1802793356958748,
  max_depth=9,
  min_child_weight=4,
  n_estimators=141,
  n_jobs=100,
  subsample=0.3617558654305153,
  verbosity=0,
  random_state=420440354,
)

# COMMAND ----------

model = make_pipeline(col_selector, preprocessor, standardizer, xgbc_classifier)

# COMMAND ----------

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="xgboost_push_l7_meetme") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    xgbc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}
    metrics_pdf = pd.DataFrame([xgbc_val_metrics, xgbc_test_metrics], index=["validation", "test"])
    metrics_pdf["dataset"] = ["ml_push.push_val", "ml_push.push_test"]
    metrics_df = spark.createDataFrame(metrics_pdf)
    display(metrics_df)
    

# COMMAND ----------

# Save metrics to a delta table
metrics_df.write.mode("overwrite").saveAsTable("ml_push.metric_push_data")

# COMMAND ----------

predicted_probs = model.predict_proba(X_val)
predicted_probs[:10, 0]

# COMMAND ----------

roc_auc = sklearn.metrics.roc_auc_score(y_val, predicted_probs[:,1])
print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Part 2. Hyperparameter Tuning - Parallel training with Hyperopt and SparkTrials
# MAGIC [Hyperopt](http://hyperopt.github.io/hyperopt/) is a Python library for hyperparameter tuning. For more information about using Hyperopt in Databricks, see the documentation ([AWS](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl-hyperparam-tuning/index#hyperparameter-tuning-with-hyperopt)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)).
# MAGIC 
# MAGIC You can use Hyperopt with SparkTrials to run hyperparameter sweeps and train multiple models in parallel. This reduces the time required to optimize model performance. MLflow tracking is integrated with Hyperopt to automatically log models and parameters.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

# COMMAND ----------

# Define the search space to explore
search_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 20, 800, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}


def train_model(params):
    # Enable autologging on each worker
    mlflow.autolog()
    with mlflow.start_run(nested = True):
        model_hp = XGBClassifier(
            random_state = 0,
            **params
        )
        # Chain indexer and dtc together into a single ML Pipeline.
        pipeline = make_pipeline(col_selector, preprocessor, standardizer, model_hp)
        model = pipeline.fit(X_train, y_train)

        predicted_probs = model.predict_proba(X_test)
        # Tune based on the test AUC
        # In production settings, you could use a separate validation set instead
        roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:, 1])
        mlflow.log_metric('test_auc', roc_auc)

        # Set the loss to -1*auc_score so fmin maximizes the auc_score
        return {'status': STATUS_OK, 'loss': -1 * roc_auc}


# SparkTrials distributes the tuning using Spark workers
# Greater parallelism speeds processing, but each hyperparameter trial has less information from other trials
# On smaller clusters or Databricks Community Edition try setting parallelism=2
spark_trials = SparkTrials(
    parallelism = 8
)

with mlflow.start_run(run_name = 'gb_hyperopt') as run:
    # Use hyperopt to find the parameters yielding the highest AUC
    best_params = fmin(
        fn = train_model,
        space = search_space,
        algo = tpe.suggest,
        max_evals = 32,
        trials = spark_trials
    )

# COMMAND ----------

best_params

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Retrain the model on the full training dataset
# MAGIC For tuning, this workflow split the training dataset into training and validation subsets. Now, retrain the model using the "best" hyperparameters on the full training dataset.

# COMMAND ----------

model_hp = XGBClassifier(
  colsample_bytree=0.49562189236760895,
  learning_rate=0.5120070791005492,
  max_depth=4,
  min_child_weight=4,
  n_estimators=490,
  n_jobs=100,
  subsample=0.3617558654305153,
  verbosity=0,
  random_state=420440354,
)

# COMMAND ----------

model = make_pipeline(col_selector, preprocessor, standardizer, model_hp)

# COMMAND ----------

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="xgboost_push_l7_meetme") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    xgbc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}
    metrics_pdf = pd.DataFrame([xgbc_val_metrics, xgbc_test_metrics], index=["validation", "test"])
    metrics_pdf["dataset"] = ["ml_push.push_val", "ml_push.push_test"]
    metrics_df = spark.createDataFrame(metrics_pdf)
    display(metrics_df)

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
    train_sample = X_train.sample(n=len(X_train.index))

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------


