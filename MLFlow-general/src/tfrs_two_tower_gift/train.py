import argparse
import os
from collections import defaultdict
from typing import Dict, Text

import mlflow
import mlflow.keras
import mlflow.pyfunc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.viewer_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=conf["unique_user_ids"], mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(conf["unique_user_ids"]) + 1, conf["viewer_embedding_dimension"]
                ),
            ]
        )

    def call(self, inputs):
        return tf.concat(
            [
                self.viewer_embedding(inputs["viewer"]),
            ],
            1,
        )


class BroadcasterModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()

        self.broadcaster_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=conf["unique_broadcasters"], mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(conf["unique_broadcasters"]) + 1,
                    conf["broadcaster_embedding_dimension"],
                ),
            ]
        )

    def call(self, broadcaster):
        return tf.concat(
            [
                self.broadcaster_embedding(broadcaster),
            ],
            1,
        )


class TwoTowers(tf.keras.Model):
    def __init__(self, broadcaster_model, user_model, task):
        super().__init__()
        self.user_model: tf.keras.Model = user_model
        self.broadcaster_model = broadcaster_model
        self.task: tf.keras.layers.Layer = task

    def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:
            # Loss computation.

            user_embeddings = self.user_model(
                {
                    "viewer": features["viewer"],
                }
            )
            broadcaster_embeddings = self.broadcaster_model(features["broadcaster"])
            loss = self.task(user_embeddings, broadcaster_embeddings)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # Loss computation.
        user_embeddings = self.user_model(
            {
                "viewer": features["viewer"],
            }
        )
        broadcaster_embeddings = self.broadcaster_model(features["broadcaster"])
        loss = self.task(user_embeddings, broadcaster_embeddings)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss
        return metrics


def load_data_file_gift(file):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["broadcaster", "viewer", "count"],
        dtype={"broadcaster": np.unicode, "viewer": np.unicode, "count": np.unicode},
    )

    values = {"broadcaster": "unknown", "viewer": "unknown", "count": "unknown"}
    training_df = training_df.sample(n=10000)
    training_df.fillna(value=values, inplace=True)
    print(training_df.head(10))
    return training_df


def load_training_gift(file):
    df = load_data_file_gift(file)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(df["viewer"].values, tf.string),
                "broadcaster": tf.cast(df["broadcaster"].values, tf.string),
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
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    return training_ds


def get_broadcaster_data_set(train_ds):
    broadcasters = train_ds.cache().map(
        lambda x: x["broadcaster"],
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    broadcasters_ds = tf.data.Dataset.from_tensor_slices(
        np.unique(list(broadcasters.as_numpy_iterator()))
    )
    return broadcasters_ds


def get_list(training_data, key):
    return training_data.batch(1_000_000).map(
        lambda x: x[key], num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )


def get_unique_list(data):
    return np.unique(np.concatenate(list(data)))


class TwoTowerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.two_tower_model = None

    def load_context(self, context):
        self.two_tower_model = tf.saved_model.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.two_tower_model


def parse_args():
    parser = argparse.ArgumentParser(description="Gift model using Two Tower")
    parser.add_argument(
        "--broadcaster_embedding_dimension",
        default=96,
        type=int,
        help="broadcaster_embedding_dimension",
    )
    parser.add_argument(
        "--viewer_embedding_dimension",
        default=96,
        type=int,
        help="viewer_embedding_dimension",
    )
    parser.add_argument("--batch_size", default=16384, type=int, help="batch_size")
    parser.add_argument(
        "--learning_rate", default=0.05, type=float, help="learning_rate"
    )
    parser.add_argument("--epochs", default=3, type=int, help="epochs")
    parser.add_argument("--top_k", default=1000, type=int, help="top_k")
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()
    conf = defaultdict(dict)
    conf["broadcaster_embedding_dimension"] = args.broadcaster_embedding_dimension
    conf["viewer_embedding_dimension"] = args.viewer_embedding_dimension
    conf["batch_size"] = args.batch_size
    conf["learning_rate"] = args.learning_rate
    conf["epochs"] = 3
    conf["top_k"] = 1000

    # Fetch the data
    local_file = "src/csv/data_latest.csv"
    if not os.path.exists(local_file) and not os.path.isfile(local_file):
        filename = "s3://tmg-machine-learning-models-dev/for-you-payer-training-data/data_latest.csv"
    else:
        filename = local_file

    dataset, nrow = load_training_gift(filename)
    gift = prepare_training_data_gift(dataset)
    shuffled = gift.shuffle(nrow, seed=42, reshuffle_each_iteration=False)

    ds_train = shuffled.take(int(nrow * 0.8))
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(conf["batch_size"])
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = shuffled.skip(int(nrow * 0.8)).take(int(nrow * 0.2))
    ds_test = ds_test.batch(conf["batch_size"])
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Fetch unique user and broadcaster
    user_ids = get_list(gift, "viewer")
    unique_user_ids = get_unique_list(user_ids)
    print(f"len(unique_user_ids) = {len(unique_user_ids)}")
    conf["unique_user_ids"] = unique_user_ids

    broadcaster_ids = get_list(gift, "broadcaster")
    unique_broadcasters = get_unique_list(broadcaster_ids)
    print(f"len(unique_broadcasters) = {len(unique_broadcasters)}")
    conf["unique_broadcasters"] = unique_broadcasters

    # enable auto logging
    # mlflow.set_experiment("gift dcn")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(
        run_name="Gift Model Experiments Using Two Tower Model"
    ) as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print(f"run_id: {run_id}")
        print(f"experiment_id: {experiment_id}")

        # Train the Model.
        user_model = UserModel(conf)
        broadcaster_model = BroadcasterModel(conf)
        broadcaster_data_set = get_broadcaster_data_set(gift)
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=broadcaster_data_set.batch(128).map(broadcaster_model)
        )
        task = tfrs.tasks.Retrieval(metrics=metrics)
        model = TwoTowers(broadcaster_model, user_model, task)
        model.compile(optimizer=tf.keras.optimizers.Adam(conf["learning_rate"]))
        hist = model.fit(ds_train, epochs=conf["epochs"], verbose=False)
        train_accuracy = hist.history["factorized_top_k/top_100_categorical_accuracy"][
            -1
        ]
        print(f"train_accuracy: {train_accuracy}")
        mlflow.log_param("size", nrow)
        mlflow.log_metric("train_accuracy", train_accuracy)

        print("create index")
        index = tfrs.layers.factorized_top_k.BruteForce(
            query_model=user_model,
            k=conf["top_k"],
        )

        index.index_from_dataset(
            tf.data.Dataset.zip(
                (
                    broadcaster_data_set.batch(1000),
                    broadcaster_data_set.batch(1000).map(model.broadcaster_model),
                )
            )
        )

        _, broadcasters = index(
            {
                "viewer": tf.constant(["kik:user:iamtesla215_ju3"]),
            }
        )
        print(f"Recommendations for user kik:user:iamtesla215_ju3: {broadcasters}")

        # save the model
        artifacts = {"model_path": "model"}
        tf.saved_model.save(index, artifacts["model_path"])
        mlflow.pyfunc.log_model(
            artifact_path=artifacts["model_path"],
            python_model=TwoTowerWrapper(),
            artifacts=artifacts,
        )
        mlflow.end_run()


if __name__ == "__main__":
    main()
