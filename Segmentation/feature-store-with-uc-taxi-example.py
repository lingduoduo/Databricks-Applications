# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced example for Feature Engineering in Unity Catalog
# MAGIC
# MAGIC This notebook illustrates the use of Feature Engineering in Unity Catalog to create a model that predicts NYC Yellow Taxi fares. It includes these steps:
# MAGIC
# MAGIC - Compute and write time series features directly in Unity Catalog.
# MAGIC - Train a model using these features to predict fares.
# MAGIC - Evaluate that model on a fresh batch of data using existing features.
# MAGIC
# MAGIC ## Requirements
# MAGIC - Databricks Runtime 13.3 LTS for Machine Learning or above
# MAGIC   - If you do not have access to Databricks Runtime for Machine Learning, you can run this notebook on Databricks Runtime 13.3 LTS or above. To do so, run `%pip install databricks-feature-engineering` at the start of this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_flow_v3.png"/>

# COMMAND ----------

# MAGIC %md ## Compute features

# COMMAND ----------

# MAGIC %md #### Load the raw data used to compute features
# MAGIC
# MAGIC Load the `nyc-taxi-tiny` dataset.  This was generated from the full [NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) which can be found at `dbfs:/databricks-datasets/nyctaxi` by applying the following transformations:
# MAGIC
# MAGIC 1. Apply a UDF to convert latitude and longitude coordinates into ZIP codes, and add a ZIP code column to the DataFrame.
# MAGIC 1. Subsample the dataset into a smaller dataset based on a date range query using the `.sample()` method of the Spark `DataFrame` API.
# MAGIC 1. Rename certain columns and drop unnecessary columns.
# MAGIC
# MAGIC If you want to create this dataset from the raw data yourself, follow these steps:
# MAGIC 1. Run the Workspace Feature Store taxi example dataset notebook ([AWS](https://docs.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[Azure](https://docs.microsoft.com/azure/databricks/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[GCP](https://docs.gcp.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)) to generate the Delta table.
# MAGIC 1. In this notebook, replace the following `spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")` with: `spark.read.table("feature_store_taxi_example.nyc_yellow_taxi_with_zips")`

# COMMAND ----------

raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC From the taxi fares transactional data, we will compute two groups of features based on trip pickup and drop off zip codes.
# MAGIC
# MAGIC #### Pickup features
# MAGIC 1. Count of trips (time window = 1 hour, sliding window = 15 minutes)
# MAGIC 1. Mean fare amount (time window = 1 hour, sliding window = 15 minutes)
# MAGIC
# MAGIC #### Drop off features
# MAGIC 1. Count of trips (time window = 30 minutes)
# MAGIC 1. Does trip end on the weekend (custom feature using python code)
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_computation_v5.png"/>

# COMMAND ----------

# MAGIC %md ### Helper functions

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
from pytz import timezone


@udf(returnType=IntegerType())
def is_weekend(dt):
    tz = "America/New_York"
    return int(dt.astimezone(timezone(tz)).weekday() >= 5)  # 5 = Saturday, 6 = Sunday


def filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(col(ts_column) < end_date)
    return df

# COMMAND ----------

# MAGIC %md ### Data scientist's custom code to compute features

# COMMAND ----------

def pickup_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the pickup_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(df, ts_column, start_date, end_date)
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")
        )  # 1 hour window, sliding every 15 minutes
        .agg(
            mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        .select(
            col("pickup_zip").alias("zip"),
            unix_timestamp(col("window.end")).cast("timestamp").alias("ts"),
            col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features


def dropoff_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the dropoff_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(df, ts_column, start_date, end_date)
    dropoffzip_features = (
        df.groupBy("dropoff_zip", window("tpep_dropoff_datetime", "30 minute"))
        .agg(count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            col("dropoff_zip").alias("zip"),
            unix_timestamp(col("window.end")).cast("timestamp").alias("ts"),
            col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            is_weekend(col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features

# COMMAND ----------

from datetime import datetime

pickup_features = pickup_features_fn(
    df=raw_data,
    ts_column="tpep_pickup_datetime",
    start_date=datetime(2016, 1, 1),
    end_date=datetime(2016, 1, 31),
)
dropoff_features = dropoff_features_fn(
    df=raw_data,
    ts_column="tpep_dropoff_datetime",
    start_date=datetime(2016, 1, 1),
    end_date=datetime(2016, 1, 31),
)

# COMMAND ----------

display(pickup_features)

# COMMAND ----------

display(dropoff_features)

# COMMAND ----------

# MAGIC %md ### Create new time series feature tables in Unity Catalog

# COMMAND ----------

# MAGIC %md First, create a new catalog or reuse an existing one and create the schema where the feature tables will be stored.
# MAGIC - To create a new catalog, you must have the `CREATE CATALOG` privilege on the metastore.
# MAGIC - To use an existing catalog, you must have the `USE CATALOG` privilege on the catalog.
# MAGIC - To create a new schema in the catalog, you must have the `CREATE SCHEMA` privilege on the catalog.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a new catalog with:
# MAGIC -- CREATE CATALOG IF NOT EXISTS ml;
# MAGIC -- USE CATALOG ml;
# MAGIC
# MAGIC -- Or reuse existing catalog:
# MAGIC USE CATALOG ml;
# MAGIC
# MAGIC -- Create a new schema
# MAGIC CREATE SCHEMA IF NOT EXISTS taxi_example;
# MAGIC USE SCHEMA taxi_example;

# COMMAND ----------

# MAGIC %md Next, create time series feature tables in Unity Catalog with Primary Key Constraints.
# MAGIC
# MAGIC You can directly create a table in Unity Catalog using `CREATE TABLE` SQL syntax. Use the primary key constraint to specify primary key columns. For time series tables, use `TIMESERIES` to annotate the timeseries column ([AWS](https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-table-constraint.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/sql-ref-syntax-ddl-create-table-constraint)|[GCP](https://docs.gcp.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-table-constraint.html)).
# MAGIC
# MAGIC The timeseries column must be of `TimestampType` or `DateType`.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS ml.taxi_example.trip_pickup_time_series_features(
# MAGIC   zip INT NOT NULL,
# MAGIC   ts TIMESTAMP NOT NULL,
# MAGIC   mean_fare_window_1h_pickup_zip FLOAT,
# MAGIC   count_trips_window_1h_pickup_zip INT,
# MAGIC   CONSTRAINT trip_pickup_time_series_features_pk PRIMARY KEY (zip, ts TIMESERIES)
# MAGIC )
# MAGIC COMMENT "Taxi Fares. Pickup Time Series Features";

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS ml.taxi_example.trip_dropoff_time_series_features(
# MAGIC   zip INT NOT NULL,
# MAGIC   ts TIMESTAMP NOT NULL,
# MAGIC   count_trips_window_30m_dropoff_zip INT,
# MAGIC   dropoff_is_weekend INT,
# MAGIC   CONSTRAINT trip_dropoff_time_series_features_pk PRIMARY KEY (zip, ts TIMESERIES)
# MAGIC )
# MAGIC COMMENT "Taxi Fares. Dropoff Time Series Features";

# COMMAND ----------

# MAGIC %md ### Write initial features to feature tables in Unity Catalog

# COMMAND ----------

# MAGIC %md Create an instance of the Feature Engineering client.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Use the `write_table` API to write features to the feature tables in Unity Catalog.
# MAGIC
# MAGIC To write to a time series feature table, the DataFrame must contain a column that you designate as the timeseries column.

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "5")
fe.write_table(
    name="ml.taxi_example.trip_pickup_time_series_features",
    df=pickup_features
)
fe.write_table(
    name="ml.taxi_example.trip_dropoff_time_series_features",
    df=dropoff_features
)

# COMMAND ----------

# MAGIC %md ## Update features
# MAGIC
# MAGIC Use the `write_table` function to update the feature table values.

# COMMAND ----------

display(raw_data)

# COMMAND ----------

# Compute a newer batch of pickup_features feature group.
new_pickup_features = pickup_features_fn(
    df=raw_data,
    ts_column="tpep_pickup_datetime",
    start_date=datetime(2016, 2, 1),
    end_date=datetime(2016, 2, 29),
)
# Write the new pickup features DataFrame to the feature table
fe.write_table(
    name="ml.taxi_example.trip_pickup_time_series_features",
    df=new_pickup_features,
    mode="merge",
)

# Compute a newer batch of dropoff_features feature group.
new_dropoff_features = dropoff_features_fn(
    df=raw_data,
    ts_column="tpep_dropoff_datetime",
    start_date=datetime(2016, 2, 1),
    end_date=datetime(2016, 2, 29),
)
# Write the new dropoff features DataFrame to the feature table
fe.write_table(
    name="ml.taxi_example.trip_dropoff_time_series_features",
    df=new_dropoff_features,
    mode="merge",
)

# COMMAND ----------

# MAGIC %md When writing, `merge` mode is supported.
# MAGIC
# MAGIC     fe.write_table(
# MAGIC       name="ml.taxi_example.trip_pickup_time_series_features",
# MAGIC       df=new_pickup_features,
# MAGIC       mode="merge",
# MAGIC     )
# MAGIC
# MAGIC Data can also be streamed into feature tables by passing a dataframe where `df.isStreaming` is set to `True`:
# MAGIC
# MAGIC     fe.write_table(
# MAGIC       name="ml.taxi_example.trip_pickup_time_series_features",
# MAGIC       df=streaming_pickup_features,
# MAGIC       mode="merge",
# MAGIC     )
# MAGIC
# MAGIC You can schedule a notebook to periodically update features using Databricks Jobs ([AWS](https://docs.databricks.com/jobs.html)|[Azure](https://docs.microsoft.com/azure/databricks/jobs)|[GCP](https://docs.gcp.databricks.com/jobs.html)).

# COMMAND ----------

# MAGIC %md Analysts can interact with feature tables in Unity Catalog using SQL, for example:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   SUM(count_trips_window_30m_dropoff_zip) AS num_rides,
# MAGIC   dropoff_is_weekend
# MAGIC FROM
# MAGIC   ml.taxi_example.trip_dropoff_time_series_features
# MAGIC WHERE
# MAGIC   dropoff_is_weekend IS NOT NULL
# MAGIC GROUP BY
# MAGIC   dropoff_is_weekend;

# COMMAND ----------

# MAGIC %md ## Feature search and discovery

# COMMAND ----------

# MAGIC %md
# MAGIC You can now discover your feature tables in Unity Catalog in the <a href="#feature-store/feature-store" target="_blank">Features UI</a>. Search by "ml.taxi_example.trip_pickup_time_series_features" or "ml.taxi_example.trip_dropoff_time_series_features" and click the table name to see details such as table schema, metadata, and lineage in the Catalog Explorer UI. You can also edit the description for the feature table. For more information about feature discovery and tracking feature lineage, see ([AWS](https://docs.databricks.com/machine-learning/feature-store/uc/ui-uc.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/uc/ui-uc)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/uc/ui-uc.html)).
# MAGIC
# MAGIC You can also set feature table permissions in the Catalog Explorer UI. For details, see ([AWS](https://docs.databricks.com/data-governance/unity-catalog/manage-privileges/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/manage-privileges/)|[GCP](https://docs.gcp.databricks.com/data-governance/unity-catalog/manage-privileges/index.html)).

# COMMAND ----------

# MAGIC %md ## Train a model
# MAGIC
# MAGIC This section illustrates how to create a training set with the time series pickup and dropoff feature tables using point-in-time lookup and train a model using the training set. It trains a LightGBM model to predict taxi fare.

# COMMAND ----------

# MAGIC %md ### Helper functions

# COMMAND ----------

import mlflow.pyfunc

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# MAGIC %md ### Understanding how a training dataset is created
# MAGIC
# MAGIC In order to train a model, you need to create a training dataset that is used to train the model.  The training dataset is comprised of:
# MAGIC
# MAGIC 1. Raw input data
# MAGIC 1. Features from the feature tables in Unity Catalog
# MAGIC
# MAGIC The raw input data is needed because it contains:
# MAGIC
# MAGIC 1. Primary keys and timeseries columns are used to join with features with point-in-time correctness ([AWS](https://docs.databricks.com/machine-learning/feature-store/time-series.html#create-a-training-set-with-a-time-series-feature-table)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/time-series#create-a-training-set-with-a-time-series-feature-table)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/train-models-with-feature-store.html#create-a-training-dataset)).
# MAGIC 1. Raw features like `trip_distance` that are not in the feature tables.
# MAGIC 1. Prediction targets like `fare` that are required for model training.
# MAGIC
# MAGIC Here's a visual overview that shows the raw input data being combined with the features in the Unity Catalog to produce the training dataset:
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_feature_lookup_with_pit.png"/>
# MAGIC
# MAGIC
# MAGIC These concepts are described further in the Creating a Training Dataset documentation ([AWS](https://docs.databricks.com/machine-learning/feature-store/train-models-with-feature-store.html#create-a-training-dataset)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/train-models-with-feature-store#create-a-training-dataset)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/train-models-with-feature-store.html#create-a-training-dataset)).
# MAGIC
# MAGIC The next cell loads features from Unity Catalog for model training by creating a `FeatureLookup` for each needed feature.
# MAGIC
# MAGIC To perform a point-in-time lookup for feature values from a time series feature table, you must specify a `timestamp_lookup_key` in the feature’s `FeatureLookup`, which indicates the name of the DataFrame column that contains timestamps against which to lookup time series features. For each row in the DataFrame, the feature values retrieved are the latest feature values prior to the timestamps specified in the DataFrame’s `timestamp_lookup_key` column and whose primary keys match the values in the DataFrame’s `lookup_key` columns, or `null` if no such feature value exists.

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup
import mlflow

pickup_features_table = "ml.taxi_example.trip_pickup_time_series_features"
dropoff_features_table = "ml.taxi_example.trip_dropoff_time_series_features"

pickup_feature_lookups = [
    FeatureLookup(
        table_name=pickup_features_table,
        feature_names=[
            "mean_fare_window_1h_pickup_zip",
            "count_trips_window_1h_pickup_zip",
        ],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key="tpep_pickup_datetime",
    ),
]

dropoff_feature_lookups = [
    FeatureLookup(
        table_name=dropoff_features_table,
        feature_names=["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
        lookup_key=["dropoff_zip"],
        timestamp_lookup_key="tpep_dropoff_datetime",
    ),
]

# COMMAND ----------

# MAGIC %md ### Configure MLflow client to access models in Unity Catalog

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC When `fe.create_training_set(..)` is invoked, the following steps take place:
# MAGIC
# MAGIC 1. A `TrainingSet` object is created, which selects specific features from feature tables to use in training your model. Each feature is specified by the `FeatureLookup`'s created previously.
# MAGIC
# MAGIC 1. Features are joined with the raw input data according to each `FeatureLookup`'s `lookup_key`.
# MAGIC
# MAGIC 1. Point-in-Time lookup is applied to avoid data leakage problems. Only the most recent feature values, based on `timestamp_lookup_key`, are joined.
# MAGIC
# MAGIC The `TrainingSet` is then transformed into a DataFrame for training. This DataFrame includes the columns of taxi_data, as well as the features specified in the `FeatureLookups`.

# COMMAND ----------

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed to log the model
mlflow.start_run()

# Since the timestamp columns would likely cause the model to overfit the data
# unless additional feature engineering was performed, exclude them to avoid training on them.
exclude_columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fe.create_training_set(
    df=raw_data,
    feature_lookups=pickup_feature_lookups + dropoff_feature_lookups,
    label="fare_amount",
    exclude_columns=exclude_columns,
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the feature tables, like `dropoff_is_weekend`
display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Train a LightGBM model on the data returned by `TrainingSet.to_df`, then log the model with `FeatureEngineeringClient.log_model`. The model will be packaged with feature metadata.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

mlflow.lightgbm.autolog()
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
num_rounds = 100

# Train a lightGBM model
model = lgb.train(param, train_lgb_dataset, num_rounds)

# COMMAND ----------

# Log the trained model with MLflow and package it with feature lookup information.
fe.log_model(
    model=model,
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name="ml.taxi_example.taxi_example_fare_time_series_packaged",
)

# COMMAND ----------

# MAGIC %md ### See the model lineage in Catalog Explorer
# MAGIC Visit the table details page in Catalog Explorer. Go to "Lineage" tab and click "See lineage graph". You can see that the feature table now has a downstream model lineage.

# COMMAND ----------

# MAGIC %md ### Build and log a custom PyFunc model
# MAGIC
# MAGIC To add preprocessing or post-processing code to the model and generate processed predictions with batch inference, you can build a custom PyFunc MLflow model that encapusulates these methods. The following cell shows an example that returns a string output based on the numeric prediction from the model.

# COMMAND ----------

class fareClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model):
        self.model = trained_model

    def preprocess_result(self, model_input):
        return model_input

    def postprocess_result(self, results):
        """Return post-processed results.
        Creates a set of fare ranges
        and returns the predicted range."""

        return [
            "$0 - $9.99" if result < 10 else "$10 - $19.99" if result < 20 else " > $20"
            for result in results
        ]

    def predict(self, context, model_input):
        processed_df = self.preprocess_result(model_input.copy())
        results = self.model.predict(processed_df)
        return self.postprocess_result(results)


pyfunc_model = fareClassifier(model)

# End the current MLflow run and start a new one to log the new pyfunc model
mlflow.end_run()

with mlflow.start_run() as run:
    fe.log_model(
        model=pyfunc_model,
        artifact_path="pyfunc_packaged_model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name="ml.taxi_example.pyfunc_taxi_fare_time_series_packaged",
    )

# COMMAND ----------

# MAGIC %md ## Scoring: batch inference

# COMMAND ----------

# MAGIC %md Suppose another data scientist now wants to apply this model to a different batch of data.

# COMMAND ----------

# MAGIC %md Display the data to use for inference, reordered to highlight the `fare_amount` column, which is the prediction target.

# COMMAND ----------

cols = [
    "fare_amount",
    "trip_distance",
    "pickup_zip",
    "dropoff_zip",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]
new_taxi_data = raw_data.select(cols)
display(new_taxi_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the `score_batch` API to evaluate the model on the batch of data, retrieving needed features from Feature Engineering in Unity Catalog.
# MAGIC
# MAGIC When you score a model trained with features from time series feature tables, the appropriate features are retrieved using point-in-time lookups with metadata packaged with the model during training. The DataFrame you provide to `FeatureEngineeringClient.score_batch` must contain a timestamp column with the same name and DataType as the `timestamp_lookup_key` of the FeatureLookup provided to `FeatureEngineeringClient.create_training_set`.

# COMMAND ----------

# Get the model URI
latest_model_version = get_latest_model_version(
    "ml.taxi_example.taxi_example_fare_time_series_packaged"
)
model_uri = f"models:/ml.taxi_example.taxi_example_fare_time_series_packaged/{latest_model_version}"

# Call score_batch to get the predictions from the model
with_predictions = fe.score_batch(model_uri=model_uri, df=new_taxi_data)

# COMMAND ----------

# MAGIC %md To score using the logged PyFunc model:

# COMMAND ----------

latest_pyfunc_version = get_latest_model_version(
    "ml.taxi_example.pyfunc_taxi_fare_time_series_packaged"
)
pyfunc_model_uri = (
    f"models:/ml.taxi_example.pyfunc_taxi_fare_time_series_packaged/{latest_pyfunc_version}"
)

pyfunc_predictions = fe.score_batch(
    model_uri=pyfunc_model_uri, df=new_taxi_data, result_type="string"
)

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_score_batch_with_pit.png"/>

# COMMAND ----------

# MAGIC %md ### View the taxi fare predictions
# MAGIC
# MAGIC This code reorders the columns to show the taxi fare predictions in the first column.  Note that the `predicted_fare_amount` roughly lines up with the actual `fare_amount`, although more data and feature engineering would be required to improve the model accuracy.

# COMMAND ----------

import pyspark.sql.functions as func

cols = [
    "prediction",
    "fare_amount",
    "trip_distance",
    "pickup_zip",
    "dropoff_zip",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "mean_fare_window_1h_pickup_zip",
    "count_trips_window_1h_pickup_zip",
    "count_trips_window_30m_dropoff_zip",
    "dropoff_is_weekend",
]

with_predictions_reordered = (
    with_predictions.select(
        cols,
    )
    .withColumnRenamed(
        "prediction",
        "predicted_fare_amount",
    )
    .withColumn(
        "predicted_fare_amount",
        func.round("predicted_fare_amount", 2),
    )
)

display(with_predictions_reordered)

# COMMAND ----------

# MAGIC %md ### View the PyFunc predictions

# COMMAND ----------

display(pyfunc_predictions.select("fare_amount", "prediction"))

# COMMAND ----------

# MAGIC %md ## Next steps
# MAGIC
# MAGIC 1. Explore the feature tables created in this example in the <a href="#feature-store/feature-store">Features UI</a>.
# MAGIC 1. Publish the feature tables to online stores ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/publish-features.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/publish-features)).
# MAGIC 1. Deploy the model in Unity Catalog to Model Serving ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)).
# MAGIC 1. Adapt this notebook to your own data and create your own feature tables.
