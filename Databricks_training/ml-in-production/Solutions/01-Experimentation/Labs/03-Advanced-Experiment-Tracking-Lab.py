# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="d4db55e1-8962-4f27-b122-eeedd7089ae7"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Lab: Advanced Experiment Tracking
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Manually log nested runs for hyperparameter tuning with <a href="https://www.mlflow.org/docs/latest/tracking.html" target="_blank">MLflow Tracking</a>
# MAGIC  - Autolog nested runs using <a href="http://hyperopt.github.io/hyperopt/" target="_blank">hyperopt</a>

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="b0833c9c-68a1-4193-9740-2f4e02d299a5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Manual Hyperparameter Tuning
# MAGIC 
# MAGIC Create an mlflow run structured in the following way:
# MAGIC 
# MAGIC * Create a parent run named **`parent`**
# MAGIC * In this parent run:
# MAGIC   * Train a sklearn RandomForestRegressor on **`X_train`** and **`y_train`**
# MAGIC   * Get the signature and an input example. (Get the signature with **`infer_signature`**)
# MAGIC * Created a nested run named **`child_1`**
# MAGIC   * In **`child_1`**:
# MAGIC     * Train a sklearn RandomForestRegressor on **`X_train`** and **`y_train`** with a max_depth of 5
# MAGIC     * Log the "max_depth" parameter
# MAGIC     * Log the mse
# MAGIC     * Log the model with input example and signature 
# MAGIC * Create another nested run named **`child_2`**
# MAGIC   * In **`child_2`**:
# MAGIC     * Train a sklearn RandomForestRegressor on **`X_train`** and **`y_train`** with a max_depth of 10
# MAGIC     * Log the "max_depth" parameter
# MAGIC     * Log the mse
# MAGIC     * Log the model with input example and signature 
# MAGIC     * Generate and log the feature importance plot for the model (review the demo if you need a hint)

# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

pdf = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X = pdf.drop("price", axis=1)
y = pdf["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# ANSWER
import mlflow 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import numpy as np 
import matplotlib.pyplot as plt

with mlflow.start_run(run_name="parent") as run:
  
    signature = infer_signature(X_train, pd.DataFrame(y_train))
    input_example = X_train.head(3)

    with mlflow.start_run(run_name="child_1", nested=True):
        max_depth = 5
        rf = RandomForestRegressor(random_state=42, max_depth=max_depth)
        rf_model = rf.fit(X_train, y_train)
        mse = mean_squared_error(rf_model.predict(X_test), y_test)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(rf_model, "model", signature=signature, input_example=input_example)

    with mlflow.start_run(run_name="child_2", nested=True):
        max_depth = 10
        rf = RandomForestRegressor(random_state=42, max_depth=max_depth)
        rf_model = rf.fit(X_train, y_train)
        mse = mean_squared_error(rf_model.predict(X_test), y_test)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(rf_model, "model", signature=signature, input_example=input_example)

        # Generate feature importance plot
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        feature_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")

        # Log figure
        mlflow.log_figure(fig, "feature_importances_rf.png")

# COMMAND ----------

# MAGIC %md <i18n value="ef7fafd4-ac6f-4e20-8ae0-e427e372ce92"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Autologging with Hyperopt
# MAGIC 
# MAGIC In this exercise, you will use HyperOpt to tune the hyperparameters of an sklearn random forest model. 
# MAGIC 
# MAGIC For this exercise:
# MAGIC 
# MAGIC 1. Use HyperOpt to tune **`n_estimators`** and **`max_depth`** in a sklearn random forest.
# MAGIC   * **`n_estimators`**: Use 50-500, in steps of 10
# MAGIC     * Take a look at the <a href="http://hyperopt.github.io/hyperopt/getting-started/search_spaces/" target="_blank">docs</a> to find the correct distribution to use
# MAGIC   * **`max_depth`**: Use 5-15, in steps of 1 
# MAGIC     * Take a look at the <a href="http://hyperopt.github.io/hyperopt/getting-started/search_spaces/" target="_blank">docs</a> to find the correct distribution to use
# MAGIC   * **`parallelism`**: 2
# MAGIC   * **`max_evals`**: 16
# MAGIC 2. Find the nested runs in the MLflow UI
# MAGIC 3. Generate the Parallel Coordinates Plot as shown in the lesson on your nested runs. 
# MAGIC 
# MAGIC **Note:** You will need to select all nested runs and hit compare in the MLflow UI. If you select the bottom-most nested run and then shift-click the top-most nested run, you will select all of them.

# COMMAND ----------

# ANSWER 
from hyperopt import fmin, tpe, hp, SparkTrials 

# Define objective function
def objective(params):
    # build a Random Forest Regressor with hyperparameters
    model = RandomForestRegressor(n_estimators=int(params["n_estimators"]), 
                                  max_depth=int(params["max_depth"]))

    # fit model with training data
    model.fit(X_train, y_train)

    # predict on testing data
    pred = model.predict(X_test)

    # compute mean squared error
    score = mean_squared_error(pred, y_test)
    return score

# COMMAND ----------

# ANSWER
# Define search space
search_space = {"n_estimators": hp.quniform("n_estimators", 50, 500, 10),
                "max_depth": hp.quniform("max_depth", 5, 15, 1)
               }

# Set algorithm type
algo = tpe.suggest
# Create SparkTrials object
spark_trials = SparkTrials(parallelism=2)

# start run
with mlflow.start_run(run_name="Hyperopt"):
    argmin = fmin(fn=objective,
                  space=search_space,
                  algo=algo,
                  max_evals=16,
                  trials=spark_trials)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
