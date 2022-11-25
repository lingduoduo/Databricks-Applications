# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="0bcc02e5-87e7-4dd9-8973-84babb1f8652"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Experiment Tracking
# MAGIC 
# MAGIC The machine learning life cycle involves training multiple algorithms using different hyperparameters and libraries, all with different performance results and trained models.  This lesson explores tracking those experiments to organize the machine learning life cycle.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Introduce tracking ML experiments in MLflow
# MAGIC  - Log an experiment and explore the results in the UI
# MAGIC  - Record parameters, metrics, and a model
# MAGIC  - Query past runs programatically

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="f567a231-9c54-4417-8c26-a2079e38d4a5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Over the course of the machine learning life cycle...<br><br>
# MAGIC 
# MAGIC * Data scientists test many different models
# MAGIC * Using various libraries
# MAGIC * Each with different hyperparameters
# MAGIC 
# MAGIC Tracking these various results poses an organizational challenge, including... <br><br>
# MAGIC 
# MAGIC * Storing experiments
# MAGIC * Results
# MAGIC * Models
# MAGIC * Supplementary artifacts
# MAGIC * Code
# MAGIC * Data snapshots

# COMMAND ----------

# MAGIC %md <i18n value="8379bb78-bbb6-48c4-910b-37ce5b17030b"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Tracking Experiments with MLflow
# MAGIC 
# MAGIC MLflow Tracking is...<br>
# MAGIC 
# MAGIC * a logging API specific for machine learning
# MAGIC * agnostic to libraries and environments that do the training
# MAGIC * organized around the concept of **runs**, which are executions of data science code
# MAGIC * runs are aggregated into **experiments** where many runs can be a part of a given experiment
# MAGIC * An MLflow server can host many experiments.
# MAGIC 
# MAGIC Each run can record the following information:<br>
# MAGIC 
# MAGIC * **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC * **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC * **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and data files
# MAGIC * **Source:** The code that originally ran the experiment
# MAGIC 
# MAGIC Experiments can be tracked using libraries in Python, R, and Java as well as by using the CLI and REST calls

# COMMAND ----------

# MAGIC %md <i18n value="4f094143-9696-4f76-8368-e249b0ff22c6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="5b6379eb-3ea4-41c3-b320-158c98cb75ef"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Experiment Logging and UI

# COMMAND ----------

# MAGIC %md <i18n value="924c3ca2-26cd-4a67-98f3-3976f61165d6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Load the San Francisco Airbnb listings - we'll use this to train a model.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="eaf7bc15-c588-4fd1-bd8e-534c31a2192b"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Navigate to the MLflow UI by clicking on the `Experiment` button on the top of the screen.**
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Every Python notebook in a Databricks Workspace has its own experiment. When you use MLflow in a notebook, it records runs in the notebook experiment. A notebook experiment shares the same name and ID as its corresponding notebook.

# COMMAND ----------

# MAGIC %md <i18n value="18498d46-c500-4939-95c8-f17df6f913ca"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Log a basic experiment by doing the following:<br><br>
# MAGIC 
# MAGIC 1. Start an experiment using **`mlflow.start_run()`** and passing it a name for the run
# MAGIC 2. Train your model
# MAGIC 3. Log the model using **`mlflow.sklearn.log_model()`**
# MAGIC 4. Log the model error using **`mlflow.log_metric()`**
# MAGIC 5. Print out the run id using **`run.info.run_id`**

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Basic RF Run") as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random_forest_model")

    # Log metrics
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)

    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="5350e1fc-0670-4e81-b747-524ca465bae7"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Examine the results in the UI.  Look for the following:<br><br>
# MAGIC 
# MAGIC 1. The `Experiment ID`
# MAGIC 2. The time the run was executed.  **Click this timestamp to see more information on the run.**
# MAGIC 3. The code that executed the run.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/mlflow/mlflow_exp_ui.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="18464172-8be4-4c26-85c7-3d9674253f98"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC After clicking on the time of the run, take a look at the following:<br><br>
# MAGIC 
# MAGIC 1. The Run ID will match what we printed above
# MAGIC 2. The model that we saved, included a picked version of the model as well as the Conda environment and the **`MLmodel`** file, which will be discussed in the next lesson.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/mlflow/mlflow_model_page.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="7c2e26f3-1fb1-4208-bbdd-a9127dd4619c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Parameters, Metrics, and Artifacts
# MAGIC 
# MAGIC But wait, there's more!  In the last example, you logged the run name, an evaluation metric, and your model itself as an artifact.  Now let's log parameters, multiple metrics, and other artifacts including the feature importances.
# MAGIC 
# MAGIC First, create a function to perform this.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> To log artifacts, we have to save them somewhere before MLflow can log them.  This code accomplishes that by using a temporary file that it then deletes.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def log_rf(experiment_id, run_name, params, X_train, X_test, y_train, y_test):
  
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Create model, train it, and create predictions
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model")

        # Log params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            "mse": mean_squared_error(y_test, predictions), 
            "mae": mean_absolute_error(y_test, predictions), 
            "r2": r2_score(y_test, predictions)
        })

        # Log feature importance
        importance = (pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), columns=["Feature", "Importance"])
                      .sort_values("Importance", ascending=False))
        importance_path = f"{DA.paths.working_path}/importance.csv"
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, "feature-importance.csv")

        # Log plot
        fig, ax = plt.subplots()
        importance.plot.bar(ax=ax)
        plt.title("Feature Importances")
        mlflow.log_figure(fig, "feature_importances.png")
        display(fig)

        return run.info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="8a40693f-5a0f-4a2f-90ed-660fa0a46ad5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Run with new parameters.

# COMMAND ----------

params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

log_rf(experiment_id, "Second Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md <i18n value="e54b78e2-f3a2-4ce4-ba44-dd466c1e2f6a"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Check the UI to see how this appears.  Take a look at the artifact to see where the plot was saved.
# MAGIC 
# MAGIC Now, run a third run.

# COMMAND ----------

params_1000_trees = {
    "n_estimators": 1000,
    "max_depth": 10,
    "random_state": 42
}

log_rf(experiment_id, "Third Run", params_1000_trees, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="0da95562-a3c3-406c-9142-ea6045b19d6b"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Querying Past Runs
# MAGIC 
# MAGIC You can query past runs programatically in order to use this data back in Python.  The pathway to doing this is an **`MlflowClient`** object.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> You can also set tags for runs using `client.set_tag(run.info.run_id, "tag_key", "tag_value")`

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

# MAGIC %md <i18n value="4e9d2c32-cbe0-4536-8aaa-3ad8611f2b03"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now list all the runs for your experiment using **`.list_run_infos()`**, which takes your **`experiment_id`** as a parameter.

# COMMAND ----------

display(client.list_run_infos(experiment_id))

# COMMAND ----------

# MAGIC %md <i18n value="354881c3-880b-4315-a3e7-f2c84bc812a9"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Pull out a few fields and create a spark DataFrame with it.

# COMMAND ----------

runs = spark.read.format("mlflow-experiment").load(experiment_id)
display(runs)

# COMMAND ----------

# MAGIC %md <i18n value="dc942bf8-e27f-45e3-8c28-cef25862046d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Pull the last run and take a look at the associated artifacts.

# COMMAND ----------

run_rf = runs.orderBy("start_time", ascending=False).first()

client.list_artifacts(run_rf.run_id)

# COMMAND ----------

# MAGIC %md <i18n value="1c34346a-8431-44da-9eae-ea2acebb6a9b"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Return the evaluation metrics for the last run.

# COMMAND ----------

client.get_run(run_rf.run_id).data.metrics

# COMMAND ----------

# MAGIC %md <i18n value="6476766c-5cc4-44c9-91c9-0a2304e50457"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Reload the model and take a look at the feature importance.

# COMMAND ----------

model = mlflow.sklearn.load_model(f"runs:/{run_rf.run_id}/random_forest_model")
model.feature_importances_

# COMMAND ----------

# MAGIC %md <i18n value="6d9be940-9705-4f7a-b453-a29df8a62cad"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Review
# MAGIC **Question:** What can MLflow Tracking log?
# MAGIC **Answer:** MLflow can log the following:
# MAGIC - **Parameters:** inputs to a model
# MAGIC - **Metrics:** the performance of the model
# MAGIC - **Artifacts:** any object including data, models, and images
# MAGIC - **Source:** the original code, including the commit hash if linked to git
# MAGIC 
# MAGIC **Question:** How do you log experiments?
# MAGIC **Answer:** Experiments are logged by first creating a run and using the logging methods on that run object (e.g. **`run.log_param("MSE", .2)`**).
# MAGIC 
# MAGIC **Question:** Where do logged artifacts get saved?
# MAGIC **Answer:** Logged artifacts are saved in a directory of your choosing.  On Databricks, this would be DBFS (Databricks File System).
# MAGIC 
# MAGIC **Question:** How can I query past runs?
# MAGIC **Answer:** This can be done using an **`MlflowClient`** object.  This allows you do everything you can within the UI programatically so you never have to step outside of your programming environment.

# COMMAND ----------

# MAGIC %md <i18n value="b9bcc0d6-110a-456a-a950-a7404782cf7f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the labs for this lesson, [Experiment Tracking Lab]($./Labs/02-Experiment-Tracking-Lab)

# COMMAND ----------

# MAGIC %md <i18n value="2fc3311c-4fc1-43ec-a673-b5842f1f05f3"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** What is MLflow at a high level?
# MAGIC **A:** <a href="https://databricks.com/session/accelerating-the-machine-learning-lifecycle-with-mlflow-1-0" target="_blank">Listen to Spark and MLflow creator Matei Zaharia's talk at Spark Summit in 2019.</a>
# MAGIC 
# MAGIC **Q:** What is a good source for the larger context of machine learning tools?
# MAGIC **A:** <a href="https://roaringelephant.org/2019/06/18/episode-145-alex-zeltov-on-mlops-with-mlflow-kubeflow-and-other-tools-part-1/#more-1958" target="_blank">Check out this episode of the podcast Roaring Elephant.</a>
# MAGIC 
# MAGIC **Q:** Where can I find the MLflow docs?
# MAGIC **A:** <a href="https://www.mlflow.org/docs/latest/index.html" target="_blank">You can find the docs here.</a>
# MAGIC 
# MAGIC **Q:** What is a good general resource for machine learning?
# MAGIC **A:** <a href="https://www-bcf.usc.edu/~gareth/ISL/" target="_blank">_An Introduction to Statistical Learning_</a> is a good starting point for the themes and basic approaches to machine learning.
# MAGIC 
# MAGIC **Q:** Where can I find out more information on machine learning with Spark?
# MAGIC **A:** Check out the Databricks blog <a href="https://databricks.com/blog/category/engineering/machine-learning" target="_blank">dedicated to machine learning</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
