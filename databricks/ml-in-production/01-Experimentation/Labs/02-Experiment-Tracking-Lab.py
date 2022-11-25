# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="8ce98e48-d7f7-430d-ace6-a69a610a1eb6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Lab: Grid Search with MLflow
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Perform grid search using scikit-learn
# MAGIC  - Log the best model on MLflow
# MAGIC  - Load the saved model

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="39a1bcda-628a-4751-b34c-59c375fffdef"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Data Import
# MAGIC 
# MAGIC Load in same Airbnb data and create train/test split.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

# COMMAND ----------

# MAGIC %md <i18n value="103dd475-d82f-4966-9efc-ae45d27125ae"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Perform Grid Search using scikit-learn
# MAGIC 
# MAGIC We want to know which combination of hyperparameter values is the most effective. Fill in the code below to perform <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV" target="_blank"> grid search using **`sklearn`**</a>.
# MAGIC 
# MAGIC Set **`n_estimators`** to **`[50, 100]`** and **`max_depth`** to **`[3, 5]`**.

# COMMAND ----------

# TODO
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# dictionary containing hyperparameter names and list of values we want to try
parameters = {"n_estimators": #FILL_IN , 
              "max_depth": #FILL_IN }

rf = RandomForestRegressor()
grid_rf_model = GridSearchCV(rf, parameters, cv=3)
grid_rf_model.fit(X_train, y_train)

best_rf = grid_rf_model.best_estimator_
for p in parameters:
    print(f"Best '{p}': {best_rf.get_params()[p]}")

# COMMAND ----------

# MAGIC %md <i18n value="5de38ed5-17a5-421c-8ad1-893be59b062f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Log Best Model with MLflow
# MAGIC 
# MAGIC Log the best model as **`grid-random-forest-model`**, its parameters, and its MSE metric under a run with name **`RF-Grid-Search`** in our new MLflow experiment.

# COMMAND ----------

# TODO
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name= FILL_IN) as run:
    # Create predictions of X_test using best model
    # FILL_IN

    # Log model with name
    # FILL_IN

    # Log params
    # FILL_IN

    # Create and log MSE metrics using predictions of X_test and its actual value y_test
    # FILL_IN

    run_id = run.info.run_id
    print(f"Inside MLflow Run with id {run_id}")

# COMMAND ----------

# MAGIC %md <i18n value="ac401810-3cb2-4f6b-8d41-5a723769fe41"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Load the Saved Model
# MAGIC 
# MAGIC Load the trained and tuned model we just saved. Check that the hyperparameters of this model matches that of the best model we found earlier.

# COMMAND ----------

# TODO
model = < FILL_IN >

# COMMAND ----------

# MAGIC %md <i18n value="0350f335-9b0d-4e5c-a8bd-41865144ff43"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Time permitting, use the `MlflowClient` to interact programatically with your run.

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="2c011665-8444-441c-b830-e0e74886684f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Advanced Experiment Tracking]($../03-Advanced-Experiment-Tracking).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
