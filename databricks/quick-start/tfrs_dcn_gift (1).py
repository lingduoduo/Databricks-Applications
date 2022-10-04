# Databricks notebook source
display(dbutils.fs.ls("/FileStore/tables/65cb05a3_e45a_4a15_915b_90cf082dc203.csv"))

# COMMAND ----------

import mlflow

# COMMAND ----------

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load Data

# COMMAND ----------

def load_data_file_gift(file):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["broadcaster", "viewer", "product_name", "order_time", "count"],
        dtype={
            "broadcaster": str,
            "viewer": str,
            "product_name": str,
            "order_time": str,
            "count": int,
        },
    )

    values = {
        "broadcaster": "unknown",
        "viewer": "unknown",
        "product_name": "unknown",
        "order_time": "0",
        "count": 0,
    }

    training_df = training_df.sample(n=1000)
    training_df.fillna(value=values, inplace=True)
    return training_df


def load_training_gift(file):
    df = load_data_file_gift(file)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(df["viewer"].values, tf.string),
                "broadcaster": tf.cast(df["broadcaster"].values, tf.string),
                "product_name": tf.cast(df["product_name"].values, tf.string),
                "order_time": tf.cast(df["order_time"].values, tf.string),
                "count": tf.cast(df["count"].values, tf.int64),
            }
        )
    )
    return training_ds, len(df)


def prepare_training_data_gift(train_ds):
    print("prepare_training_data")
    training_ds = train_ds.map(
        lambda x: {
            "broadcaster": x["broadcaster"],
            "viewer": x["viewer"],
            "product_name": x["product_name"],
            "order_time": x["order_time"],
            "count": x["count"],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return training_ds


def feature_mapping(train_ds, feature_name):
    vocab = train_ds.batch(1_000_000).map(
        lambda x: x[feature_name],
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return np.unique(np.concatenate(list(vocab)))

# COMMAND ----------

filename = "/dbfs/FileStore/tables/65cb05a3_e45a_4a15_915b_90cf082dc203.csv"
dataset, nrow = load_training_gift(filename)

# COMMAND ----------

gift = prepare_training_data_gift(dataset)
shuffled = gift.shuffle(nrow, seed=42, reshuffle_each_iteration=False)

# COMMAND ----------

from collections import defaultdict
conf = defaultdict(dict)
conf["embedding_dimension"] = 32
conf["batch_size"] = 16384
conf["learning_rate"] = 0.05
conf["epochs"] = 5
conf["deep_layer_sizes"] = [192, 192]
conf["str_features"] = ["broadcaster", "viewer", "product_name", "order_time"]
conf["int_features"] = []
conf["label_name"] = "count"

# COMMAND ----------

ds_train = shuffled.take(int(nrow * 0.8))
ds_train = ds_train.cache()
ds_train = ds_train.batch(conf["batch_size"])
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = shuffled.skip(int(nrow * 0.8)).take(int(nrow * 0.2))
ds_test = ds_test.batch(conf["batch_size"])
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Define DCN

# COMMAND ----------

class DCN(tfrs.Model):
    def __init__(self, conf, use_cross_layer, deep_layer_sizes, projection_dim=None):
        super().__init__()

        self.embedding_dimension = conf["embedding_dimension"]
        str_features = conf["str_features"]
        int_features = conf["int_features"]
        self._all_features = str_features + int_features
        self._embeddings = {}
        self.label_name = conf["label_name"]

        # Compute embeddings for string features.
        vocabularies = conf["vocabularies"]
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.experimental.preprocessing.StringLookup(
                        vocabulary=vocabulary, mask_token=None
                    ),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1,
                        # self.embedding_dimension
                        6 * int(pow(len(vocabulary), 0.25)),
                    ),
                ]
            )

        # Compute embeddings for int features.
        for feature_name in int_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.IntegerLookup(
                        vocabulary=vocabulary, mask_value=None
                    ),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1,
                        # self.embedding_dimension
                        6 * int(pow(len(vocabulary), 0.25)),
                    ),
                ]
            )

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim, kernel_initializer="glorot_uniform"
            )
        else:
            self._cross_layer = None

        self._deep_layers = [
            tf.keras.layers.Dense(layer_size, activation="relu")
            for layer_size in deep_layer_sizes
        ]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")],
        )

    def call(self, inputs):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(inputs[feature_name]))

        x = tf.concat(embeddings, 1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)

        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

        return self._logit_layer(x)

    def compute_loss(self, inputs, training=False):
        labels = inputs.pop(self.label_name)
        scores = self(inputs)
        return self.task(
            labels=labels,
            predictions=scores,
        )

# COMMAND ----------

# Fetch feature and vocabularies
features = ["viewer", "broadcaster", "product_name", "order_time"]
vocabularies = {}
for idx, feature in enumerate(features):
    print(f"{idx}: {feature}")
    vocabularies[feature] = feature_mapping(gift, feature)
    conf["vocabularies"] = vocabularies

# Train the Model.
model = DCN(
conf=conf,
use_cross_layer=True,
deep_layer_sizes=[192, 192],
projection_dim=None,
)

model.compile(optimizer=tf.keras.optimizers.Adam(conf["learning_rate"]))
model.fit(ds_train, epochs=conf["epochs"], verbose=False)
metrics = model.evaluate(ds_test, return_dict=True)
print(f"metrics: {metrics}")

# COMMAND ----------

# save the model
tf.saved_model.save(model, "/databricks/tfrs_dcn_gift/model")
artifacts = {"model_path": "/databricks/tfrs_dcn_gift/model"}

# COMMAND ----------

def predict_ranking_model(input_example, path):
    model = tf.saved_model.load(f"{path}")
    score = model(
        {
            "viewer": np.array([input_example["viewer"]]),
            "broadcaster": np.array([input_example["broadcaster"]]),
            "product_name": np.array([input_example["product_name"]]),
            "order_time": np.array([input_example["order_time"]]),
        }
    ).numpy()
    return score

# COMMAND ----------

input_example = {
     "viewer": "kik:user:unknown",
     "broadcaster": "kik:user:unknown",
     "product_name": "Rose",
     "order_time": "10",
}
pred = predict_ranking_model(input_example, artifacts["model_path"])
print(pred)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Use MLflow mlflow.tensorflow.autolog()

# COMMAND ----------

# enable auto logging
mlflow.tensorflow.autolog()
# mlflow.set_experiment("gift dcn")
with mlflow.start_run():
    model = DCN(
        conf=conf,
        use_cross_layer=True,
        deep_layer_sizes=[192, 192],
        projection_dim=None,
        )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(conf["learning_rate"]))
    model.fit(ds_train, epochs=conf["epochs"], verbose=False)
    metrics = model.evaluate(ds_test, return_dict=True)
    
    mlflow.log_param("size", nrow)
    mlflow.log_metric("RMSE", metrics["RMSE"])
    mlflow.end_run()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Use Mlflow.pyfunc.log_model to register unsupported model

# COMMAND ----------

class DCNWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        tf.saved_model.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        score = predict_ranking_model(
            model_input["viewer"],
            model_input["broadcaster"],
            model_input["product_name"],
            model_input["order_time"],
            context.artifacts["model_path"],
        )
        return score

# COMMAND ----------

# Train the Model.
model = DCN(
    conf=conf,
    use_cross_layer=True,
    deep_layer_sizes=[192, 192],
    projection_dim=None,
)
model.compile(optimizer=tf.keras.optimizers.Adam(conf["learning_rate"]))
model.fit(ds_train, epochs=conf["epochs"], verbose=False)
metrics = model.evaluate(ds_test, return_dict=True)
print(f"metrics: {metrics}")

# save the model
tf.saved_model.save(model, "/databricks/tfrs_dcn_gift/model")
artifacts = {"model_path": "/databricks/tfrs_dcn_gift/model"}

# enable auto logging
mlflow.tensorflow.autolog()
# mlflow.set_experiment("gift dcn")
with mlflow.start_run():
    mlflow.log_param("size", nrow)
    mlflow.log_metric("RMSE", metrics["RMSE"])

    mlflow.pyfunc.log_model(
        artifact_path=artifacts["model_path"],
        python_model=DCNWrapper(),
        artifacts=artifacts,
    )
    mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### hyperopt

# COMMAND ----------

def create_model():
    # Fetch feature and vocabularies
    features = ["viewer", "broadcaster", "product_name", "order_time"]
    vocabularies = {}
    for idx, feature in enumerate(features):
        print(f"{idx}: {feature}")
        vocabularies[feature] = feature_mapping(gift, feature)
    conf["vocabularies"] = vocabularies

    # Train the Model.
    model = DCN(
        conf=conf,
        use_cross_layer=True,
        deep_layer_sizes=[192, 192],
        projection_dim=None,
    )
    return model

# COMMAND ----------

from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

# COMMAND ----------

def run_model(config):

    # Log run information with mlflow.tensorflow.autolog()
    mlflow.tensorflow.autolog()

    model = create_model()

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]))
    model.fit(ds_train, epochs=config["epochs"], verbose=False)
    

    # Evaluate the model
    metrics = model.evaluate(ds_test, return_dict=True)
    return {"RMSE": metrics["RMSE"], "status": STATUS_OK}


# COMMAND ----------

space = {
  "learning_rate": hp.loguniform("learning_rate", -5, 0),
  "epochs": hp.choice("epochs", [3, 10])
 }

# COMMAND ----------

spark_trials = SparkTrials()

# COMMAND ----------

with mlflow.start_run():
    best_hyperparam = fmin(fn=run_model, 
                         space=space, 
                         algo=tpe.suggest, 
                         trials=spark_trials)

# COMMAND ----------


