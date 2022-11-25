# Databricks notebook source
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

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["broadcaster_id", "device_type", "utc_day_of_week", "utc_hour"]
col_selector = ColumnSelector(supported_cols)

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
# MAGIC ### Train - Test- Split

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Split the df_train data into 2 sets:
# MAGIC  - Train (80% of the dataset used to train the model)
# MAGIC  - Test(20% of the dataset used to tune the hyperparameters of the model)

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
# MAGIC ### Train classification model
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

# MAGIC %md ### View MLflow runs
# MAGIC To view the logged training runs, click the **Experiment** icon at the upper right of the notebook to display the experiment sidebar. If necessary, click the refresh icon to fetch and monitor the latest runs. 
# MAGIC 
# MAGIC <img width="350" src="https://docs.databricks.com/_static/images/mlflow/quickstart/experiment-sidebar-icons.png"/>
# MAGIC 
# MAGIC You can then click the experiment page icon to display the more detailed MLflow experiment page ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#notebook-experiments)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#notebook-experiments)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#notebook-experiments)). This page allows you to compare runs and view details for specific runs.
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/compare-runs.png"/>

# COMMAND ----------

# MAGIC %md ### Compare multiple runs in the UI
# MAGIC As in Part 1, you can view and compare the runs in the MLflow experiment details page, accessible via the external link icon at the top of the **Experiment** sidebar. 
# MAGIC 
# MAGIC On the experiment details page, click the "+" icon to expand the parent run, then select all runs except the parent, and click **Compare**. You can visualize the different runs using a parallel coordinates plot, which shows the impact of different parameter values on a metric. 
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/parallel-plot.png"/>

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

len(X_train)

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=len(X_train))

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

# COMMAND ----------

# Register to Model Registry
model_name = "Test-Stage-Model"

model_uri = f"runs:/{mlflow_run.info.run_id}/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# Load from Model Registry
loaded_model = mlflow.sklearn.load_model(model_uri)

model.predict(X_train)

# COMMAND ----------

model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### MLflow stats to Delta Lake Table

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

mlflow_run.info.run_id

# COMMAND ----------

expId = mlflow_run.info.experiment_id

# COMMAND ----------

mlflow_df = spark.read.format("mlflow-experiment").load(expId)

# COMMAND ----------

refined_mlflow_df = mlflow_df.select(col('run_id'), col("experiment_id"), explode(map_concat(col("metrics"), col("params"))), col('start_time'), col("end_time")) \
                .filter("key != 'model'") \
                .select("run_id", "experiment_id", "key", col("value").cast("float"), col('start_time'), col("end_time")) \
                .groupBy("run_id", "experiment_id", "start_time", "end_time") \
                .pivot("key") \
                .sum("value") \
                .withColumn("trainingDuration", col("end_time").cast("integer")-col("start_time").cast("integer")) # example of added column

# COMMAND ----------

refined_mlflow_df.write.mode("overwrite").saveAsTable(f"ml_push.experiment_data_20221018")

# COMMAND ----------

display(refined_mlflow_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Calculate Data Drift

# COMMAND ----------

# MAGIC %md
# MAGIC Understanding data drift is key to understanding when it is time to retrain your model. When you train a model, you are training it on a sample of data. While these training datasets are usually quite large, they don't represent changes that may happend to the data in the future. For instance, as new push data gets collected, new sends and opens could appear in the data coming into the model to be scored that the model does not know how to properly score.  
# MAGIC 
# MAGIC Monitoring for this drift is important so that you can retrain and refresh the model to allow for the model to adapt.  
# MAGIC 
# MAGIC The short example of this that we are showing today uses the [Kolmogorov-Smirnov test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) to compare the distribution of the training dataset with the incoming data that is being scored by the model.

# COMMAND ----------

# running Kolmogorov-Smirnov test for numerical columns
from scipy import stats
from pyspark.sql.types import *

from datetime import datetime

def calculate_numerical_drift(training_dataset, comparison_dataset, comparison_dataset_name, cols, p_value, date):
    drift_data = []
    for col in cols:
        passed = 1
        test = stats.ks_2samp(training_dataset[col], comparison_dataset[col])
        if test[1] < p_value:
            passed = 0
        drift_data.append((date, comparison_dataset_name, col, float(test[0]), float(test[1]), passed))
    return drift_data

# COMMAND ----------

p_value = 0.05
numerical_cols = ["utc_day_of_week", "utc_hour"]
dataset_name = "ml_push.silver_l7_push_meetme_val"
date = datetime.strptime("2022-10-05", '%Y-%m-%d').date() # simulated date for demo purpose

drift_data = calculate_numerical_drift(df_train, df_val, dataset_name, numerical_cols, p_value, date)

# COMMAND ----------

driftSchema = StructType([StructField("date", DateType(), True), \
                          StructField("dataset", StringType(), True), \
                          StructField("column", StringType(), True), \
                          StructField("statistic", FloatType(), True), \
                          StructField("pvalue", FloatType(), True), \
                          StructField("passed", IntegerType(), True)\
                      ])

# COMMAND ----------

numerical_data_drift_df = spark.createDataFrame(data=drift_data, schema=driftSchema)

# COMMAND ----------

display(numerical_data_drift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Patch requisite packages for Model serving

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

import xgboost
from mlflow.tracking import MlflowClient

xgbc_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(xgbc_temp_dir)

xgbc_client = MlflowClient()
xgbc_model_env_path = xgbc_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", xgbc_temp_dir)
xgbc_model_env_str = open(xgbc_model_env_path)
xgbc_parsed_model_env_str = yaml.load(xgbc_model_env_str, Loader=yaml.FullLoader)

xgbc_parsed_model_env_str["dependencies"][-1]["pip"].append(f"xgboost=={xgboost.__version__}")

with open(xgbc_model_env_path, "w") as f:
    f.write(yaml.dump(xgbc_parsed_model_env_str))
    
xgbc_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=xgbc_model_env_path, artifact_path="model")
shutil.rmtree(xgbc_temp_dir)

# COMMAND ----------


