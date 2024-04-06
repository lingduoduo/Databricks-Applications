# Databricks notebook source
# MAGIC %md
# MAGIC # Synthesizing Data with Generative Models for Better MLOps
# MAGIC
# MAGIC https://www.databricks.com/wp-content/uploads/notebooks/sdv-data-synthesis.html
# MAGIC
# MAGIC Generative models are all the rage, and flashy examples have dominated headlines recently -- DALL-E, ChatGPT, diffusion models. But does your business problem require concocting weird art or off-kilter poetry? Unlikely, unfortunately. Yet this new class of approaches, which generates _more data_ from data, has valuable and more prosaic applications.
# MAGIC
# MAGIC Given real business data, GANs (Generative Adversarial Networks) and VAEs (variational autoencoders) can produce synthetic data that resembles real data. Is fake data useful? It could be in cases where source data is sensitive and not readily shareable, yet something like the real data is needed for development or testing of pipelines on that data. Perhaps a third-party data science team will develop a new modeling pipeline, but, sharing sensitive data is not possible. Develop on synthetic data! 
# MAGIC
# MAGIC It's even possible that a bit of synthetic data alongside real data improves modeling outcomes.
# MAGIC
# MAGIC This example explores use of the Python library [SDV](https://sdv.dev/SDV/) to generate synthetic data resembling a dataset, and then uses [Auto ML](https://www.databricks.com/product/automl) to assess the quality of models built with synthetic data. SDV uses deep learning, GANs in particular, via its [TVAE](https://sdv.dev/SDV/user_guides/single_table/tvae.html) module. 
# MAGIC
# MAGIC With this example, you too can exploit generative AI!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC This notebook was run on DBR 12.2 ML, but should work on other recent versions.
# MAGIC Use a single-GPU instance if desired (in which case, use 12.2 ML GPU); if not, set `use_gpu=False` below.
# MAGIC
# MAGIC Install SDV along with supporting libraries for visualization. (`pandas-profiling` needs a small update to pick up a bug fix.)

# COMMAND ----------

# MAGIC %pip install "sdv==0.18.0" kaleido "pandas-profiling>=3.6.3"

# COMMAND ----------

use_gpu = True

username = spark.sql("select current_user()").first()['current_user()']
tmp_dir = f"/tmp/{username}"
tmp_experiment_dir = f"/Users/{username}/SDV"

print(f"Using tmp_dir: {tmp_dir}")
print(f"Using tmp_experiment_dir: {tmp_experiment_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC This example will turn again to our old friend, one of several NYC Taxi-related datasets like the one used in a [Kaggle competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data). This data is already available in `/databricks-datasets` in your workspace. It describes a huge number of taxi rides, including their pickup and drop-off time and place, distance, tolls, vendor, etc.
# MAGIC
# MAGIC Imagine we wish to predict the tip that the rider will add after the trip, using this data set. (For this reason, `total_amount` is redacted, as it would be a target leak.) Who knows? maybe this might be used to intelligently suggest a tip amount
# MAGIC
# MAGIC It's a huge dataset, so only a small sample will be used. This is, generally, a straightforward regression problem.

# COMMAND ----------

# Stick to reliable data in 2009-2016 years
train_df, test_df = spark.read.format("delta").load("/databricks-datasets/nyctaxi/tables/nyctaxi_yellow").\
  filter("YEAR(pickup_datetime) >= 2009 AND YEAR(pickup_datetime) <= 2016").\
  drop("total_amount").\
  sample(0.0005, seed=42).\
  randomSplit([0.9, 0.1], seed=42)
  
table_nyctaxi = train_df.toPandas().sample(frac=1, random_state=42, ignore_index=True) # random shuffle for good measure
table_nyctaxi.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a Synthetic Dataset
# MAGIC
# MAGIC Of course, it'd be easy to get started from here and build a regressor, with [Auto ML](https://www.databricks.com/product/automl) or any standard open-source library, if one had access to the data above.
# MAGIC
# MAGIC Imagine that this dataset is _sensitive_ or otherwise not shareable with data science practitioners, yet, these practitioners need to produce a model that accurately predicts tip. It's not as crazy as it sounds; even within an organization, it's possible that important data is tightly controlled and otherwise unavailable for data science experimentation, which is unfortunate for the experimenters.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A First Pass with SDV
# MAGIC
# MAGIC This is where a library like [SDV](https://sdv.dev/SDV/) (Synthetic Data Vault) come in. SDV is a toolkit for synthesizing data that looks like a given source. It can handle multi-table data architectures with foreign keys, anonymization of PII, and implements sophisticated synthesis techniques based on copulas, GANs and VAEs.
# MAGIC
# MAGIC However it also provides a fast and easy-to-use "preset" for simple single-table setups (built on [Gaussian copulas](https://bochang.me/blog/posts/copula/), for the interested). Give it a try:

# COMMAND ----------

from sdv.metadata.dataset import Metadata
from sdv.lite import TabularPreset

metadata = Metadata()
metadata.add_table(name="nyctaxi_yellow", data=table_nyctaxi)

model = TabularPreset(name='FAST_ML', metadata=metadata.get_table_meta("nyctaxi_yellow"))
model.fit(table_nyctaxi)

model.sample(num_rows=5, randomize_samples=False)

# COMMAND ----------

display(spark.createDataFrame(model.sample(num_rows=1000, randomize_samples=False)))

# COMMAND ----------

# MAGIC %md
# MAGIC At a glance, that looks like plausible real data. A deeper glance reveals some problems, though:
# MAGIC
# MAGIC - Some monetary amounts are negative, like MTA tax or tip
# MAGIC - Passenger count and distance are 0 sometimes
# MAGIC - Distance is occasionally impossibly shorter than straight line distance
# MAGIC - Longitude and latitude are sometimes nowhere near New York City (in some cases entirely invalid, like >90 degrees latitude)
# MAGIC - Monetary amounts have more than two decimal places
# MAGIC - Pickup time is occasionally after dropoff time, probably due to daylight savings issues, or sometimes more than a 12-hour shift long
# MAGIC
# MAGIC Many of these are actually problems found in the source data too!
# MAGIC We can nevertheless proceed to get a report on the data quality, from SDV:

# COMMAND ----------

df = spark.read.format("delta").load("/databricks-datasets/nyctaxi/tables/nyctaxi_yellow")

# COMMAND ----------

table_nyctaxi = df.sample(0.000005, seed=42)

# COMMAND ----------

# MAGIC %pip install "sdv==0.18.0" kaleido "pandas-profiling>=3.6.3"

# COMMAND ----------

from sdv.metadata.dataset import Metadata
from sdv.lite import TabularPreset

metadata = Metadata()
metadata.add_table(name="nyctaxi_yellow", data=table_nyctaxi)

# COMMAND ----------

from sdmetrics.reports.single_table import QualityReport

report = QualityReport()
report.generate(table_nyctaxi, 
                model.sample(num_rows=10000, randomize_samples=False), 
                metadata.get_table_meta("nyctaxi_yellow"))

report.get_visualization("Column Pair Trends")

# COMMAND ----------

# MAGIC %md
# MAGIC The quality is "OK". Pairwise correlations look similar between real and synthetic data; there is some issue with the synthesized `store_and_fwd_flag` column. Clearly, the synthetic data issues need fixing. 
# MAGIC
# MAGIC "Quality" is about 75%. This metric is just the average of two quality scores, based on individual column shapes, and column pairs - more or less how much columns, and pairs of columns, in the synthetic data look like real data.
# MAGIC
# MAGIC Those data engineers should fix the data problems, really, before data scientists seriously consider synthesizing data from it. Here, these can just be filtered from the input. Later, these observations will also translate into `Constraint`s that SDV can apply to its output as well, such as enforcing that a number is positive or in a range.
# MAGIC
# MAGIC Start over and fix the input data by filtering:

# COMMAND ----------

from pyspark.sql.functions import col, pandas_udf
import numpy as np

df = spark.read.format("delta").load("/databricks-datasets/nyctaxi/tables/nyctaxi_yellow").\
  filter("YEAR(pickup_datetime) >= 2009 AND YEAR(pickup_datetime) <= 2016").\
  drop("total_amount")

for c in ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount"]:
  df = df.filter(f"{c} >= 0")
df = df.filter("passenger_count > 0")
df = df.filter("trip_distance > 0 AND trip_distance < 100")
df = df.filter("dropoff_datetime > pickup_datetime")
df = df.filter("CAST(dropoff_datetime AS long) < CAST(pickup_datetime AS long) + 12 * 60 * 60")
for c in ["pickup_longitude", "dropoff_longitude"]:
  df = df.filter(f"{c} > -76 AND {c} < -72")
for c in ["pickup_latitude", "dropoff_latitude"]:
  df = df.filter(f"{c} > 39 AND {c} < 43")

# Define this as a standalone function for reuse later
def haversine_dist_miles(from_lat_deg, from_lon_deg, to_lat_deg, to_lon_deg):
  to_lat = np.deg2rad(to_lat_deg)
  to_lon = np.deg2rad(to_lon_deg)
  from_lat = np.deg2rad(from_lat_deg)
  from_lon = np.deg2rad(from_lon_deg)
  # 3958.8 is avg earth radius in miles
  return 3958.8 * 2 * np.arcsin(np.sqrt(
    np.square(np.sin((to_lat - from_lat) / 2)) + np.cos(to_lat) * np.cos(from_lat) * np.square(np.sin((to_lon - from_lon) / 2)))) 
  
@pandas_udf('double')
def haversine_dist_miles_udf(from_lat_deg, from_lon_deg, to_lat_deg, to_lon_deg):
  return haversine_dist_miles(from_lat_deg, from_lon_deg, to_lat_deg, to_lon_deg)

# Allow 90% of min theoretical distance to account for rounding, inaccuracy
df = df.filter(col("trip_distance") >= 
               0.9 * haversine_dist_miles_udf("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"))
  
train_df, test_df = df.sample(0.0005, seed=42).randomSplit([0.9, 0.1], seed=42)
train_df.cache()

table_nyctaxi = train_df.toPandas().sample(frac=1, random_state=42, ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Try again with `TabularPreset` on the fixed-up data:

# COMMAND ----------

metadata = Metadata()
metadata.add_table(name="nyctaxi_yellow", data=table_nyctaxi)

model = TabularPreset(name='FAST_ML', metadata=metadata.get_table_meta("nyctaxi_yellow"))
model.fit(table_nyctaxi)

report = QualityReport()
report.generate(table_nyctaxi, 
                model.sample(num_rows=10000, randomize_samples=False), 
                metadata.get_table_meta("nyctaxi_yellow"))

report.get_visualization("Column Pair Trends")

# COMMAND ----------

# MAGIC %md
# MAGIC Quality went from 75% to 82%, as it's easier to imitate data that doesn't have odd outliers. Can we do better with some advanced techniques?
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding Constraints, Variational Autoencoders, and MLflow
# MAGIC
# MAGIC Let's jump ahead to add several improvements. Below, SDV's [TVAE](https://sdv.dev/SDV/api_reference/tabular/api/sdv.tabular.ctgan.TVAE.html) (Triplet-based Variational Autoencoder) module is used for more sophisticated (and compute-intensive) synthesis of data. This part can and should be accelerated with a GPU. SDV also has `CTGAN` and `CopulaGAN`, though these turn out to be less effective on this data. Deep learning, GPUs - this is real-deal AI!
# MAGIC
# MAGIC Constraints are also added, per above, to improve the realism of the output. This includes a custom `Constraint` that checks `trip_distance` against the straight-line (Haversine) distance between the pickup and dropoff lat/lon, and a custom `Constraint` limiting the duration of the trip.

# COMMAND ----------

from sdv.constraints import FixedIncrements, Inequality, Positive, ScalarInequality, ScalarRange, create_custom_constraint
import numpy as np

# Add constraints mirroring those above

constraints = []

# Distance shouldn't be (too) much less than straight line distance
def is_trip_distance_valid(column_names, data):
  dist_col, from_lat, from_lon, to_lat, to_lon = column_names
  return data[dist_col] >= 0.9 * haversine_dist_miles(data[from_lat], data[from_lon], data[to_lat], data[to_lon])

TripDistanceValid = create_custom_constraint(is_valid_fn=is_trip_distance_valid)
constraints += [TripDistanceValid(column_names=["trip_distance", "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])]

# Dropoff shouldn't be more than 12 hours after pickup, or before pickup
def is_duration_valid(column_names, data):
  pickup_col, dropoff_col = column_names
  return (data[dropoff_col] - data[pickup_col]) < np.timedelta64(12, 'h')

DurationValid = create_custom_constraint(is_valid_fn=is_duration_valid)
constraints += [DurationValid(column_names=["pickup_datetime", "dropoff_datetime"])]
constraints += [Inequality(low_column_name="pickup_datetime", high_column_name="dropoff_datetime")]

# Monetary amounts should be positive
constraints += [ScalarInequality(column_name=c, relation=">=", value=0) for c in 
                 ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount"]]
# Passengers should be a positive integer
constraints += [FixedIncrements(column_name="passenger_count", increment_value=1)]
constraints += [Positive(column_name="passenger_count")]
# Distance should be positive and not (say) more than 100 miles
constraints += [ScalarRange(column_name="trip_distance", low_value=0, high_value=100)]
# Lat/lon should be in some credible range around New York City
constraints += [ScalarRange(column_name=c, low_value=-76, high_value=-72) for c in ["pickup_longitude", "dropoff_longitude"]]
constraints += [ScalarRange(column_name=c, low_value=39, high_value=43) for c in ["pickup_latitude", "dropoff_latitude"]]

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, the whole modeling process is also tracked via MLflow - data quality metrics, plots, and even a "model" based on SDV that can be loaded or deployed to generate synthetic data as its output.

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from sdv.metadata.dataset import Metadata
from sdv.tabular import TVAE
from sdmetrics.reports.single_table import QualityReport
import pandas as pd

# Won't log 'internal' sklearn models fit by SDV
mlflow.autolog(disable=True) 

# Wrapper convenience model that lets the SDV model "predict" new synthetic data
class SynthesizeModel(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model

  def predict(self, context, model_input):
    return self.model.sample(num_rows=len(model_input))

with mlflow.start_run():
  metadata = Metadata()
  metadata.add_table(name="nyctaxi_yellow", data=table_nyctaxi)

  model = TVAE(constraints=constraints, batch_size=1000, epochs=500, cuda=use_gpu)
  model.fit(table_nyctaxi)
  
  sample = model.sample(num_rows=10000, randomize_samples=False)
  report = QualityReport()
  report.generate(table_nyctaxi, sample, metadata.get_table_meta("nyctaxi_yellow"))
  
  # Log metrics and plots with MLflow
  mlflow.log_metric("Quality Score", report.get_score())
  for (prop, score) in report.get_properties().to_numpy().tolist():
    mlflow.log_metric(prop, score)
    mlflow.log_dict(report.get_details(prop).to_dict(orient='records'), f"{prop}.json")
    prop_viz = report.get_visualization(prop)
    display(prop_viz)
    mlflow.log_figure(prop_viz, f"{prop}.png")
  
  # Log wrapper model for synthesis of data, if desired
  # Not strictly necessary; this model's .pkl serialization could have been logged as an artifact,
  # or not at all
  if use_gpu:
    # Assign model to CPU for later inference; GPU not really useful
    model._model.set_device('cpu')
  synthesize_model = SynthesizeModel(model)
  dummy_input = pd.DataFrame([True], columns=["dummy"]) # dummy value
  signature = infer_signature(dummy_input, synthesize_model.predict(None, dummy_input))
  mlflow.pyfunc.log_model("model", python_model=synthesize_model, 
                          registered_model_name="sdv_synth_model",
                          input_example=dummy_input, signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC Quality is up slightly, to 83.2%.
# MAGIC
# MAGIC Check out the MLflow run linked in the cell output above to see what MLflow captured, and even registered as model `sdv_synth_model`. MLflow has all of the plots and metrics, and even the data generator as a 'model' for later use.
# MAGIC
# MAGIC Also, **move this latest version of `sdv_synth_model` into `Production`** for the next step, in the UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generating the Data
# MAGIC
# MAGIC Now, to generate some synthetic data! Above, fitting the generative model that synthesizes data took a while and does not parallelize across a cluster, but, applying it to create data can be neatly parallelized with Spark. It'll be simpler here to just load the original SDV model from MLflow, write a simple function to make data, and then apply it in parallel with Spark:

# COMMAND ----------

import mlflow

# Pick out the raw SDV model inside the wrapper
sdv_model = mlflow.pyfunc.load_model("models:/sdv_synth_model/Production")._model_impl.python_model.model

# Simple function to generate data from the model. The input could really be anything; here the input
# is assumed to be the number of rows to generate.
def synthesize_data(how_many_dfs):
  for how_many_df in how_many_dfs:
    # This will generate different data every run, note; can't be seeded, except to make it return
    # the same data every single call!
    # output_file_path='disable' is a workaround (?) for temp file errors
    yield sdv_model.sample(num_rows=how_many_df.sum().item(), output_file_path='disable')

# Generate, for example, the same number of rows as in the input
how_many = len(table_nyctaxi)
partitions = 256
synth_df = spark.createDataFrame([(how_many // partitions,)] * partitions).\
  repartition(partitions).\
  mapInPandas(synthesize_data, schema=df.schema)

display(synth_df)

# COMMAND ----------

synth_data_path = f"{tmp_dir}/synth/nyctaxi_synth"
synth_df.write.format("delta").save(synth_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Much better! Whole numbers of passengers, money that looks like dollars and cents, reasonable-looking locations.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing Synthetic Data
# MAGIC
# MAGIC In Databricks, you can generate a profile of any dataset. Here we want to compare the original and synthetic data sets, to get a sense of how much they match. Use `pandas-profiling`:

# COMMAND ----------

from pandas_profiling import ProfileReport

synth_data_df = spark.read.format("delta").load(synth_data_path).toPandas()

original_report = ProfileReport(table_nyctaxi, title='Original Data', minimal=True)
synth_report = ProfileReport(synth_data_df, title='Synthetic Data', minimal=True)
compare_report = original_report.compare(synth_report)
compare_report.config.html.navbar_show = False
compare_report.config.html.full_width = True
displayHTML(compare_report.to_html())

# COMMAND ----------

# MAGIC %md
# MAGIC There is definitely broad similarity in distributions of individual features. There are also some odd differences, like the non-uniformity of pickup/dropoff time in the synthetic data. Addressing this may be a matter of further tuning the synthetic data process, and is out of scope here.
# MAGIC
# MAGIC It's worth saying that there are limits to any data synthesis process. For example, here the model can generate pickup/dropoff points that resemble the input's points, in their range and distribution and even in their relation to each other considering the distance. But it has no direct way of knowing whether the points make sense as places on a street to pick up a person.
# MAGIC
# MAGIC There isn't a free lunch here, and the generated data at best roughly resembles real data. This is part of the point, of course, that it not be so realistic that real data 'leaks' into the output. How good is it? Let's try to build a model with it. Save the synthetic data set.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling on Synthetic Data
# MAGIC
# MAGIC This is the point where the data scientist reenters the picture. We're in a development environment. He or she has access to this synthetic data set now, and the task is to build a model that will predict _real_ fare tips reasonably well. A good start would be to simply use auto ML to fit a reasonable model to the synthetic data:

# COMMAND ----------

synth_data_path = f"{tmp_dir}/synth/nyctaxi_synth"

# COMMAND ----------

import databricks.automl
databricks.automl.regress(
  spark.read.format("delta").load(synth_data_path),
  target_col="tip_amount",
  primary_metric="rmse",
  experiment_dir=tmp_experiment_dir,
  experiment_name="Synth models",
  timeout_minutes=120)

# COMMAND ----------

# MAGIC %md
# MAGIC The best model (after a couple hours, at least) had a (test) RMSE of about 1.1 and R-squared of 0.63. Not awful. The real question is, how well does this model trained on synthetic data hold up when applied to real data? 
# MAGIC
# MAGIC This is the kind of next step that might happen in a staging or testing environment, where the result of the modeling process produced by the data science team is tested before deployment, and this environment _should_ have some real data to test on, before the model faces, well, real data!
# MAGIC
# MAGIC A simple snippet of what might transpire is below. Here we actually have loaded real data already, so, evaluate metrics on that:

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

test_pd = test_df.toPandas()

def print_metrics(exp_name):
  best_runs = mlflow.search_runs(
    experiment_names=[f"{tmp_experiment_dir}/{exp_name}"], 
    order_by=["metrics.val_root_mean_squared_error"],
    max_results=1)
  run_id = best_runs['run_id'].item()
  model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
  y_pred = model.predict(test_pd.drop("tip_amount", axis=1))
  y_true = test_pd["tip_amount"]
  print(f"RMSE: {sqrt(mean_squared_error(y_true, y_pred))}")
  print(f"R^2:  {r2_score(y_true, y_pred)}")
  
print_metrics("Synth models")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing to Modeling on Original Data
# MAGIC
# MAGIC Pretty comparable metrics, actually, which is good news. RMSE and R^2 are 1.52 and 0.49, versus 1.4 and 0.49 estimated from the held-out (synthetic) test set in auto ML. The model performance in this case held up reasonably on real data.
# MAGIC
# MAGIC But, wait, how would we have done if we'd fit a model on a roughly equal amount of real data?

# COMMAND ----------

import databricks.automl

databricks.automl.regress(
  train_df,
  target_col="tip_amount",
  primary_metric="rmse",
  experiment_dir=tmp_experiment_dir,
  experiment_name="Actual data models",
  timeout_minutes=120)

# COMMAND ----------

print_metrics("Actual data models")

# COMMAND ----------

# MAGIC %md
# MAGIC Modeling on real data would have done better here for sure. RMSE is 0.93, and R^2 is 0.78, vs 1.5 and 0.49. That's a significant difference. In other cases, it's possible the model on synthetic data is just about as good, but not quite here.
# MAGIC
# MAGIC However, the synthetic data modeling process did provide something. Data scientists verified the viability of building a decent model, and the model building approach, without using real data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Synthetic Data for Testing
# MAGIC
# MAGIC One might also think of synthetic data as a tool for testing a model's behavior across a wider range of inputs. Does it fail on some input, or give an outlandish answer? The synthetic data, by nature, won't look extreme compared to real data, but might exercise unusual but realistic inputs not found in training or test data.
# MAGIC
# MAGIC For example, synthetic data might be used in some kind of integration test like below, where one looks for predictions that seem simply out of normal ranges. 

# COMMAND ----------

import mlflow

best_runs = mlflow.search_runs(
  experiment_names=[f"{tmp_experiment_dir}/Synth models"], 
  order_by=["metrics.val_root_mean_squared_error"],
  max_results=1)
run_id = best_runs['run_id'].item()
model_udf = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id}/model")

synth_df = spark.read.format("delta").load(synth_data_path).drop("tip_amount")
display(synth_df.withColumn("prediction", model_udf(*synth_df.drop("tip_amount").columns)).filter("prediction < 0 OR prediction > 100"))

# COMMAND ----------

# MAGIC %md
# MAGIC Oops, this reveals a problem with this model, though one that would have been visible testing on any data, probably: predicted tips are sometimes less than 0. Really, this regression should have been construed as something like a log-linear model given the distribution of tips (non-zero, likely exponential) to avoid this, but this example will forego pursuing this.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Real _and_ Synthetic Data: Why Don't We Have Both?
# MAGIC
# MAGIC In many cases, data scientists _do_ have access to real production data to train models. What use is synthetic data? It can be viewed as a form of data augmentation, generating more real-ish data to fit on. More data usually leads to better models. In theory, nothing really new has entered the picture here. There is no additional real data. Nevertheless, sometimes additional synthetic data _does_ improve the result of a modeling process.
# MAGIC
# MAGIC In particular, synthesis is useful when the data set is imbalanced in some way; some types of inputs are rare or missing from the input. Synthesizing data to fill that gap might be especially useful. By definition, it's harder to make realistic data like subsets that are rare!
# MAGIC
# MAGIC As a final experiment, try modeling on a mix of real and synthetic data:

# COMMAND ----------

import databricks.automl

databricks.automl.regress(
  train_df.union(spark.read.format("delta").load(synth_data_path)),
  target_col="tip_amount",
  primary_metric="rmse",
  experiment_dir=tmp_experiment_dir,
  experiment_name="Hybrid data models",
  timeout_minutes=120)

# COMMAND ----------

print_metrics("Hybrid data models")

# COMMAND ----------

# MAGIC %md
# MAGIC Very nearly identical performance in this case. The synthetic data didn't help or hurt. It's possible that more accurate synthetic data would have a better result in another case.
