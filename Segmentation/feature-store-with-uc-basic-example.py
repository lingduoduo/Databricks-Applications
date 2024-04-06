# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

import pandas as pd

from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

# MAGIC %md ## Load dataset
# MAGIC

# COMMAND ----------

raw_data = spark.read.format('delta').load('dbfs:/tmp/training_gender')

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

features_df = raw_data.drop('u_yearly_household_income')
display(features_df)


# COMMAND ----------

# Create a new catalog with:
# spark.sql("CREATE CATALOG IF NOT EXISTS ling_test_demo")
# spark.sql("USE CATALOG ling_test_demo")

# Or reuse existing catalog:
spark.sql("USE CATALOG ling_test_demo")

# COMMAND ----------

# spark.sql("CREATE SCHEMA IF NOT EXISTS wine_db")
# spark.sql("USE SCHEMA wine_db")

# Create a unique table name for each run. This prevents errors if you run the notebook multiple times.
table_name = f"ling_test_demo.default.imputation_db_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

fe = FeatureEngineeringClient()

# You can get help in the notebook for feature engineering client API functions:
# help(fe.<function_name>)

# For example:
# help(fe.create_table)

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["u_caden_alias"],
    df=features_df,
    schema=features_df.schema,
    description="user features"
)

# COMMAND ----------

# MAGIC %md You can also use `create_table` without providing a dataframe, and then later populate the feature table using `fe.write_table`.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```
# MAGIC fe.create_table(
# MAGIC     name=table_name,
# MAGIC     primary_keys=["wine_id"],
# MAGIC     schema=features_df.schema,
# MAGIC     description="wine features"
# MAGIC )
# MAGIC
# MAGIC fe.write_table(
# MAGIC     name=table_name,
# MAGIC     df=features_df,
# MAGIC     mode="merge"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Train a model with Feature Engineering in Unity Catalog

# COMMAND ----------

# MAGIC %md The feature table does not include the prediction target. However, the training dataset needs the prediction target values. There may also be features that are not available until the time the model is used for inference.
# MAGIC
# MAGIC This example uses the feature **`real_time_measurement`** to represent a characteristic of the wine that can only be observed at inference time. This feature is used in training and the feature value for a wine is provided at inference time.

# COMMAND ----------

inference_data_df = raw_data.select("u_caden_alias", "u_yearly_household_income", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md Use a `FeatureLookup` to build a training dataset that uses the specified `lookup_key` to lookup features from the feature table and the online feature `real_time_measurement`. If you do not specify the `feature_names` parameter, all features except the primary key are returned.

# COMMAND ----------

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fe.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fe.create_training_set(df=inference_data_df, feature_lookups=model_feature_lookups, label="u_yearly_household_income", exclude_columns="u_caden_alias")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("u_yearly_household_income", axis=1)
    y = training_pd["u_yearly_household_income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

# Create the train and test datasets
X_train, X_test, y_train, y_test, training_set = load_data(table_name, "u_caden_alias")
X_train.head()

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = "ling_test_demo.default.imputation_model"

client = MlflowClient()

try:
    client.delete_registered_model(model_name) # Delete the model if already created
except:
    None

# COMMAND ----------

# MAGIC %md
# MAGIC The code in the next cell trains a scikit-learn RandomForestRegressor model and logs the model with the Feature Engineering in UC.
# MAGIC
# MAGIC The code starts an MLflow experiment to track training parameters and results. Note that model autologging is disabled (`mlflow.sklearn.autolog(log_models=False)`); this is because the model is logged using `fe.log_model`.

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Engineering in UC
mlflow.sklearn.autolog(log_models=False)

def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="impute_quality_prediction",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )

train_model(X_train, X_test, y_train, y_test, training_set, fe)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC To view the logged model, navigate to the MLflow Experiments page for this notebook. To access the Experiments page, click the Experiments icon on the left navigation bar:  <img src="https://docs.databricks.com/_static/images/icons/experiments-icon.png"/>
# MAGIC
# MAGIC Find the notebook experiment in the list. It has the same name as the notebook, in this case, "Basic example for Feature Engineering in Unity Catalog".
# MAGIC
# MAGIC Click the experiment name to display the experiment page. The packaged Feature Engineering in UC model, created when you called `fe.log_model` appears in the **Artifacts** section of this page. You can use this model for batch scoring.
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/basic-fs-nb-artifact.png"/>
# MAGIC
# MAGIC The model is also automatically registered in the Unity Catalog.

# COMMAND ----------

# MAGIC %md ## Batch scoring
# MAGIC Use `score_batch` to apply a packaged Feature Engineering in UC model to new data for inference. The input data only needs the primary key column `wine_id` and the realtime feature `real_time_measurement`. The model automatically looks up all of the other feature values from the feature tables.

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

## For simplicity, this example uses inference_data_df as input data for prediction
batch_input_df = inference_data_df.drop("u_yearly_household_income") # Drop the label column

latest_model_version = get_latest_model_version(model_name)

predictions_df = fe.score_batch(model_uri=f"models:/{model_name}/{latest_model_version}", df=batch_input_df)

display(predictions_df["", "prediction"])

# COMMAND ----------

# MAGIC %md ## Modify feature table
# MAGIC Suppose you modify the dataframe by adding a new feature. You can use `fe.write_table` with `mode="merge"` to update the feature table.

# COMMAND ----------

## Modify the dataframe containing the features
so2_cols = ["free_sulfur_dioxide", "total_sulfur_dioxide"]
new_features_df = (features_df.withColumn("average_so2", expr("+".join(so2_cols)) / 2))

display(new_features_df)

# COMMAND ----------

# MAGIC %md Update the feature table using `fe.write_table` with `mode="merge"`.

# COMMAND ----------

fe.write_table(
    name=table_name,
    df=new_features_df,
    mode="merge"
)

# COMMAND ----------

# MAGIC %md To read feature data from the feature tables, use `fe.read_table()`.

# COMMAND ----------

# Displays most recent version of the feature table
# Note that features that were deleted in the current version still appear in the table but with value = null.
display(fe.read_table(name=table_name))

# COMMAND ----------

# MAGIC %md ## Train a new model version using the updated feature table

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fe.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fe.create_training_set(df=inference_data_df, feature_lookups=model_feature_lookups, label="u_yearly_household_income", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("u_yearly_household_income", axis=1)
    y = training_pd["u_yearly_household_income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Build a training dataset that will use the indicated `key` to lookup features.

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )

train_model(X_train, X_test, y_train, y_test, training_set, fe)

# COMMAND ----------

# MAGIC %md Apply the latest version of the registered MLflow model to features using **`score_batch`**.

# COMMAND ----------

## For simplicity, this example uses inference_data_df as input data for prediction
batch_input_df = inference_data_df.drop("u_yearly_household_income") # Drop the label column
latest_model_version = get_latest_model_version(model_name)
predictions_df = fe.score_batch(model_uri=f"models:/{model_name}/{latest_model_version}", df=batch_input_df)
display(predictions_df["wine_id","prediction"])

# COMMAND ----------

# MAGIC %md ## Control permissions for and delete feature tables
# MAGIC - To control who has access to a Unity Catalog feature table, use the **Permissions** button on the Catalog Explorer table details page.
# MAGIC - To delete a Unity Catalog feature table, click the kebab menu on the Catalog Explorer table details page and select **Delete**. When you delete a Unity Catalog feature table using the UI, the corresponding Delta table is also deleted.
