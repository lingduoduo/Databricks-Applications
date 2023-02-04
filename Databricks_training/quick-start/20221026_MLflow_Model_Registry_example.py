# Databricks notebook source
# MAGIC %md # Overview
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions, annotations, and deployment management.
# MAGIC 
# MAGIC In this notebook, you use each of the MLflow Model Registry's components to develop and manage a production machine learning application. This notebook covers the following topics:
# MAGIC 
# MAGIC - Track and log models with MLflow
# MAGIC - Register models with the Model Registry
# MAGIC - Describe models and make model version stage transitions
# MAGIC - Integrate registered models with production applications
# MAGIC - Search and discover models in the Model Registry
# MAGIC - Archive and delete models
# MAGIC 
# MAGIC ## Requirements
# MAGIC - A cluster running Databricks Runtime 6.4 ML or above. Note that if your cluster is running Databricks Runtime 6.4 ML, you must upgrade the installed version of MLflow to 1.7.0. You can install this version from PyPI. See ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html#cluster-installed-library)|[Azure](https://docs.microsoft.com/azure/databricks/libraries/cluster-libraries#cluster-installed-library)) for instructions. 

# COMMAND ----------

# MAGIC %md # Machine learning application: Forecasting wind power
# MAGIC 
# MAGIC In this notebook, you use the MLflow Model Registry to build a machine learning application that forecasts the daily power output of a [wind farm](https://en.wikipedia.org/wiki/Wind_farm). Wind farm power output depends on weather conditions: generally, more energy is produced at higher wind speeds. Accordingly, the machine learning models used in the notebook predict power output based on weather forecasts with three features: `wind direction`, `wind speed`, and `air temperature`.

# COMMAND ----------

# MAGIC %md *This notebook uses altered data from the [National WIND Toolkit dataset](https://www.nrel.gov/grid/wind-toolkit.html) provided by NREL, which is publicly available and cited as follows:*
# MAGIC 
# MAGIC *Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Renewable Energy Laboratory.*
# MAGIC 
# MAGIC *Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366.*
# MAGIC 
# MAGIC *Lieberman-Cribbin, W., C. Draxl, and A. Clifton. 2014. Guide to Using the WIND Toolkit Validation Code (Technical Report, NREL/TP-5000-62595). Golden, CO: National Renewable Energy Laboratory.*
# MAGIC 
# MAGIC *King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Renewable Energy Laboratory.*

# COMMAND ----------

# MAGIC %md ## Load the dataset
# MAGIC 
# MAGIC The following cells load a dataset containing weather data and power output information for a wind farm in the United States. The dataset contains `wind direction`, `wind speed`, and `air temperature` features sampled every eight hours (once at `00:00`, once at `08:00`, and once at `16:00`), as well as daily aggregate power output (`power`), over several years.

# COMMAND ----------

import pandas as pd
wind_farm_data = pd.read_csv("https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv", index_col=0)

def get_training_data():
  training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
  X = training_data.drop(columns="power")
  y = training_data["power"]
  return X, y

def get_validation_data():
  validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
  X = validation_data.drop(columns="power")
  y = validation_data["power"]
  return X, y

def get_weather_and_forecast():
  format_date = lambda pd_date : pd_date.date().strftime("%Y-%m-%d")
  today = pd.Timestamp('today').normalize()
  week_ago = today - pd.Timedelta(days=5)
  week_later = today + pd.Timedelta(days=5)
  
  past_power_output = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(today)]
  weather_and_forecast = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(week_later)]
  if len(weather_and_forecast) < 10:
    past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
    weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10:]

  return weather_and_forecast.drop(columns="power"), past_power_output["power"]

# COMMAND ----------

# MAGIC %md Display a sample of the data for reference.

# COMMAND ----------

wind_farm_data["2019-01-01":"2019-01-14"]

# COMMAND ----------

# MAGIC %md # Train a power forecasting model and track it with MLflow
# MAGIC 
# MAGIC The following cells train a neural network to predict power output based on the weather features in the dataset. MLflow is used to track the model's hyperparameters, performance metrics, source code, and artifacts.

# COMMAND ----------

# MAGIC %md Define a power forecasting model using TensorFlow Keras.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# COMMAND ----------

def train_keras_model(X, y):
  
  model = Sequential()
  model.add(Dense(100, input_shape=(X_train.shape[-1],), activation="relu", name="hidden_layer"))
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")

  model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=.2)
  return model

# COMMAND ----------

# MAGIC %md Train the model and use MLflow to track its parameters, metrics, artifacts, and source code.

# COMMAND ----------

import mlflow
import mlflow.keras
import mlflow.tensorflow

X_train, y_train = get_training_data()

with mlflow.start_run():
  # Automatically capture the model's parameters, metrics, artifacts,
  # and source code with the `autolog()` function
  mlflow.tensorflow.autolog()
  
  train_keras_model(X_train, y_train)
  run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md # Register the model with the MLflow Model Registry API
# MAGIC 
# MAGIC Now that a forecasting model has been trained and tracked with MLflow, the next step is to register it with the MLflow Model Registry. You can register and manage models using the MLflow UI or the MLflow API .
# MAGIC 
# MAGIC The following cells use the API to register your forecasting model, add rich model descriptions, and perform stage transitions. See the documentation for the UI workflow.

# COMMAND ----------

model_name = "power-forecasting-model" # Replace this with the name of your registered model, if necessary.

# COMMAND ----------

# MAGIC %md ### Create a new registered model using the API
# MAGIC 
# MAGIC The following cells use the `mlflow.register_model()` function to create a new registered model whose name begins with the string `power-forecasting-model`. This also creates a new model version (for example, `Version 1` of `power-forecasting-model`).

# COMMAND ----------

import mlflow

# The default path where the MLflow autologging function stores the model
artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md After creating a model version, it may take a short period of time to become ready. Certain operations, such as model stage transitions, require the model to be in the `READY` state. Other operations, such as adding a description or fetching model details, can be performed before the model version is ready (for example, while it is in the `PENDING_REGISTRATION` state).
# MAGIC 
# MAGIC The following cell uses the `MlflowClient.get_model_version()` function to wait until the model is ready.

# COMMAND ----------

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

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
  
wait_until_ready(model_details.name, model_details.version)

# COMMAND ----------

# MAGIC %md ### Add model descriptions
# MAGIC 
# MAGIC You can add descriptions to registered models as well as model versions: 
# MAGIC * Model version descriptions are useful for detailing the unique attributes of a particular model version (such as the methodology and algorithm used to develop the model). 
# MAGIC * Registered model descriptions are useful for recording information that applies to multiple model versions (such as a general overview of the modeling problem and dataset).

# COMMAND ----------

# MAGIC %md Add a high-level description to the registered model, including the machine learning problem and dataset.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description="This model forecasts the power output of a wind farm based on weather data. The weather data consists of three features: wind speed, wind direction, and air temperature."
)

# COMMAND ----------

# MAGIC %md Add a model version description with information about the model architecture and machine learning framework.

# COMMAND ----------

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using TensorFlow Keras. It is a feed-forward neural network with one hidden layer."
)

# COMMAND ----------

# MAGIC %md ### Perform a model stage transition
# MAGIC 
# MAGIC The MLflow Model Registry defines several model stages: **None**, **Staging**, **Production**, and **Archived**. Each stage has a unique meaning. For example, **Staging** is meant for model testing, while **Production** is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages. In private preview, any user can transition a model to any stage. In the near future, administrators in your organization will be able to control these permissions on a per-user and per-model basis.
# MAGIC 
# MAGIC If you have permission to transition a model to a particular stage, you can make the transition directly by using the `MlflowClient.update_model_version()` function. If you do not have permission, you can request a stage transition using the REST API; for example:
# MAGIC 
# MAGIC ```
# MAGIC %sh curl -i -X POST -H "X-Databricks-Org-Id: <YOUR_ORG_ID>" -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" https://<YOUR_DATABRICKS_WORKSPACE_URL>/api/2.0/preview/mlflow/transition-requests/create -d '{"comment": "Please move this model into production!", "model_version": {"version": 1, "registered_model": {"name": "power-forecasting-model"}}, "stage": "Production"}'
# MAGIC ```

# COMMAND ----------

# MAGIC %md Now that you've learned about stage transitions, transition the model to the `Production` stage.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production',
)

# COMMAND ----------

# MAGIC %md Use the `MlflowClient.get_model_version()` function to fetch the model's current stage.

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------

# MAGIC %md The MLflow Model Registry allows multiple model versions to share the same stage. When referencing a model by stage, the Model Registry will use the latest model version (the model version with the largest version ID). The `MlflowClient.get_latest_versions()` function fetches the latest model version for a given stage or set of stages. The following cell uses this function to print the latest version of the power forecasting model that is in the `Production` stage.

# COMMAND ----------

latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))

# COMMAND ----------

# MAGIC %md # Integrate the model with the forecasting application
# MAGIC 
# MAGIC Now that you have trained and registered a power forecasting model with the MLflow Model Registry, the next step is to integrate it with an application. This application fetches a weather forecast for the wind farm over the next five days and uses the model to produce power forecasts. For example purposes, the application consists of a simple `forecast_power()` function (defined below) that is executed within this notebook. In practice, you may want to execute this function as a recurring batch inference job using the Databricks Jobs service.
# MAGIC 
# MAGIC The following section demonstrates how to load model versions from the MLflow Model Registry for use in applications. The **Forecast power output with the production model** section uses the **Production** model to forecast power output for the next five days.

# COMMAND ----------

# MAGIC %md ## Load versions of the registered model
# MAGIC 
# MAGIC The MLflow Models component defines functions for loading models from several machine learning frameworks. For example, `mlflow.tensorflow.load_model()` is used to load Tensorflow Keras models that were saved in MLflow format, and `mlflow.sklearn.load_model()` is used to load scikit-learn models that were saved in MLflow format.
# MAGIC 
# MAGIC These functions can load models from the MLflow Model Registry.

# COMMAND ----------

# MAGIC %md You can load a model by specifying its name (for example, `power-forecast-model`) and version number (in this case, `1`). The following cell uses the `mlflow.pyfunc.load_model()` API to load `Version 1` of the registered power forecasting model as a generic Python function.

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = "models:/{model_name}/1".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md You can also load a specific model stage. The following cell loads the `Production` stage of the power forecasting model.

# COMMAND ----------

model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model_production = mlflow.pyfunc.load_model(model_production_uri)

# COMMAND ----------

# MAGIC %md ## Forecast power output with the production model
# MAGIC 
# MAGIC In this section, the production model is used to evaluate weather forecast data for the wind farm. The `forecast_power()` application loads the latest version of the forecasting model from the specified stage and uses it to forecast power production over the next five days.

# COMMAND ----------

def plot(model_name, model_stage, model_version, power_predictions, past_power_output):
  import pandas as pd
  import matplotlib.dates as mdates
  from matplotlib import pyplot as plt
  index = power_predictions.index
  fig = plt.figure(figsize=(11, 7))
  ax = fig.add_subplot(111)
  ax.set_xlabel("Date", size=20, labelpad=20)
  ax.set_ylabel("Power\noutput\n(MW)", size=20, labelpad=60, rotation=0)
  ax.tick_params(axis='both', which='major', labelsize=17)
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
  ax.plot(index[:len(past_power_output)], past_power_output, label="True", color="red", alpha=0.5, linewidth=4)
  ax.plot(index, power_predictions.squeeze(), "--", label="Predicted by '%s'\nin stage '%s' (Version %d)" % (model_name, model_stage, model_version), color="blue", linewidth=3)
  ax.set_ylim(ymin=0, ymax=max(3500, int(max(power_predictions.values) * 1.3)))
  ax.legend(fontsize=14)
  plt.title("Wind farm power output and projections", size=24, pad=20)
  plt.tight_layout()
  display(plt.show())
  
def forecast_power(model_name, model_stage):
  from mlflow.tracking.client import MlflowClient
  client = MlflowClient()
  model_version = client.get_latest_versions(model_name, stages=[model_stage])[0].version
  model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
  model = mlflow.pyfunc.load_model(model_uri)
  weather_data, past_power_output = get_weather_and_forecast()
  power_predictions = pd.DataFrame(model.predict(weather_data))
  power_predictions.index = pd.to_datetime(weather_data.index)
  print(power_predictions)
  plot(model_name, model_stage, int(model_version), power_predictions, past_power_output)

# COMMAND ----------

forecast_power(model_name, "Production")

# COMMAND ----------

# MAGIC %md # Create and deploy a new model version
# MAGIC 
# MAGIC The MLflow Model Registry enables you to create multiple model versions corresponding to a single registered model. By performing stage transitions, you can seamlessly integrate new model versions into your staging or production environments. Model versions can be trained in different machine learning frameworks (such as `scikit-learn` and `tensorflow`); MLflow's `python_function` provides a consistent inference API across machine learning frameworks, ensuring that the same application code continues to work when a new model version is introduced.
# MAGIC 
# MAGIC The following sections create a new version of the power forecasting model using scikit-learn, perform model testing in **Staging**, and update the production application by transitioning the new model version to **Production**.

# COMMAND ----------

# MAGIC %md ## Create a new model version
# MAGIC 
# MAGIC Classical machine learning techniques are also effective for power forecasting. The following cell trains a random forest model using scikit-learn and registers it with the MLflow Model Registry via the `mlflow.sklearn.log_model()` function.

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run():
  n_estimators = 300
  mlflow.log_param("n_estimators", n_estimators)
  
  rand_forest = RandomForestRegressor(n_estimators=n_estimators)
  rand_forest.fit(X_train, y_train)

  val_x, val_y = get_validation_data()
  mse = mean_squared_error(rand_forest.predict(val_x), val_y)
  print("Validation MSE: %d" % mse)
  mlflow.log_metric("mse", mse)
  
  # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
  # function to register the model with the MLflow Model Registry. This automatically
  # creates a new model version
  mlflow.sklearn.log_model(
    sk_model=rand_forest,
    artifact_path="sklearn-model",
    registered_model_name=model_name,
  )

# COMMAND ----------

# MAGIC %md ### Fetch the new model version ID using MLflow Model Registry Search
# MAGIC 
# MAGIC The `MlflowClient.search_model_versions()` function searches for model versions by model name, MLflow run ID, or artifact source location. All model versions satisfying a particular filter query are returned.
# MAGIC 
# MAGIC The following cell uses this search function to fetch the version ID of the new model. It searches for the maximum value of the version ID (that is, the most recent version).

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md Wait for the new model version to become ready.

# COMMAND ----------

wait_until_ready(model_name, new_model_version)

# COMMAND ----------

# MAGIC %md ## Add a description to the new model version

# COMMAND ----------

client.update_model_version(
  name=model_name,
  version=new_model_version,
  description="This model version is a random forest containing 100 decision trees that was trained in scikit-learn."
)

# COMMAND ----------

# MAGIC %md ## Transition the new model version to Staging
# MAGIC 
# MAGIC Before deploying a model to a production application, it is often best practice to test it in a staging environment. The following cells transition the new model version to **Staging** and evaluate its performance.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Staging",
)

# COMMAND ----------

# MAGIC %md Evaluate the new model's forecasting performance in **Staging**

# COMMAND ----------

forecast_power(model_name, "Staging")

# COMMAND ----------

# MAGIC %md ## Transition the new model version to **Production**
# MAGIC 
# MAGIC After verifying that the new model version performs well in staging, the following cells transition the model version to **Production** and use the exact same application code from the **Forecast power output with the production model** section to produce a power forecast.
# MAGIC 
# MAGIC There are now two model versions of the forecasting model in the **Production** stage: the model version trained in Tensorflow Keras and the version trained in scikit-learn. 
# MAGIC 
# MAGIC *When referencing a model by stage, the MLflow Model Model Registry automatically uses the latest production version. This enables you to update your production models without changing any application code*. 
# MAGIC 
# MAGIC See the documentation for how to transition the model to **Production** using the UI.

# COMMAND ----------

# MAGIC %md ### Transition the new model version to Production using the API

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Production",
)

# COMMAND ----------

forecast_power(model_name, "Production")

# COMMAND ----------

# MAGIC %md # Archive and delete models
# MAGIC 
# MAGIC When a model version is no longer being used, you can archive it or delete it. You can also delete an entire registered model; this removes all of its associated model versions.

# COMMAND ----------

# MAGIC %md ## Archive `Version 1` of the power forecasting model
# MAGIC 
# MAGIC Archive `Version 1` of the power forecasting model because it is no longer being used. You can archive models in the MLflow Model Registry UI or via the MLflow API. See the documentation for the UI workflow.

# COMMAND ----------

# MAGIC %md ### Archive `Version 1` using the MLflow API
# MAGIC 
# MAGIC The following cell uses the `MlflowClient.update_model_version()` function to archive `Version 1` of the power forecasting model.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=1,
  stage="Archived",
)

# COMMAND ----------

# MAGIC %md ## Delete `Version 1` of the power forecasting model
# MAGIC 
# MAGIC You can also use the MLflow UI or MLflow API to delete model versions. **Model version deletion is permanent and cannot be undone.**
# MAGIC 
# MAGIC The following cells provide a reference for deleting `Version 1` of the power forecasting model using the MLflow API. See the documentation for how to delete a model version using the UI.

# COMMAND ----------

# MAGIC %md ### Delete `Version 1` using the MLflow API
# MAGIC 
# MAGIC The following cell permanently deletes `Version 1` of the power forecasting model.

# COMMAND ----------

client.delete_model_version(
 name=model_name,
 version=1,
)

# COMMAND ----------

# MAGIC %md ## Delete the power forecasting model
# MAGIC 
# MAGIC If you want to delete an entire registered model, including all of its model versions, you can use the `MlflowClient.delete_registered_model()` to do so. This action cannot be undone. You must first transition all model version stages to **None** or **Archived**.
# MAGIC 
# MAGIC **Warning: The following cell permanently deletes the power forecasting model, including all of its versions.**

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=2,
  stage="Archived"
)

client.delete_registered_model(name=model_name)

# COMMAND ----------

import mlflow
import shutil
from sklearn import linear_model

mlflow.autolog(disable=True)

model = linear_model.LinearRegression()
model.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
model_path = '/dbfs/Users/{0}/ml-artifacts/model-{1}-{2}'.format(user, 'a', 'b')

# clear any models in the path
shutil.rmtree(model_path, ignore_errors=True)

mlflow.sklearn.save_model(sk_model=model, path=model_path)

display(dbutils.fs.ls(model_path.replace('/dbfs', 'dbfs:')))

# COMMAND ----------

model_path

# COMMAND ----------


