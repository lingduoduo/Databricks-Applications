# in case this is run outside of conda environment with python2
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Text

import mlflow
import mlflow.keras
import mlflow.pyfunc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):
    def __init__(self, user_conf):
        super().__init__()
        self.viewer_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=user_conf["unique_user_ids"], mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(user_conf["unique_user_ids"]) + 1,
                    user_conf["viewer_embedding_dimension"],
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
    def __init__(self, broadcaster_conf):
        super().__init__()

        self.broadcaster_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=broadcaster_conf["unique_broadcasters"], mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(broadcaster_conf["unique_broadcasters"]) + 1,
                    broadcaster_conf["broadcaster_embedding_dimension"],
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
    def __init__(
        self, two_tower_broadcaster_model, two_tower_user_model, two_tower_task
    ):
        super().__init__()
        self.user_model: tf.keras.Model = two_tower_user_model
        self.broadcaster_model = two_tower_broadcaster_model
        self.task: tf.keras.layers.Layer = two_tower_task

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

        two_tower_metrics = {metric.name: metric.result() for metric in self.metrics}
        two_tower_metrics["loss"] = loss
        two_tower_metrics["regularization_loss"] = regularization_loss
        two_tower_metrics["total_loss"] = total_loss

        return two_tower_metrics

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

        two_tower_metrics = {metric.name: metric.result() for metric in self.metrics}
        two_tower_metrics["loss"] = loss
        two_tower_metrics["regularization_loss"] = regularization_loss
        two_tower_metrics["total_loss"] = total_loss
        return two_tower_metrics


def load_data_file_gift(file):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["broadcaster", "viewer", "count"],
        dtype={"broadcaster": np.unicode, "viewer": np.unicode, "count": np.float32},
    )

    values = {"broadcaster": "unknown", "viewer": "unknown", "count": "unknown"}
    training_df = training_df.sample(n=10000)
    training_df.fillna(value=values, inplace=True)
    print(training_df.head(10))
    return training_df


def load_training_gift(file):
    load_df = load_data_file_gift(file)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(load_df["viewer"].values, tf.string),
                "broadcaster": tf.cast(load_df["broadcaster"].values, tf.string),
                "count": tf.cast(load_df["count"].values, tf.float32),
            }
        )
    )

    return training_ds, len(load_df)


def prepare_training_data_gift(train_ds):
    print("prepare_training_data")
    training_ds = train_ds.map(
        lambda x: {
            "broadcaster": x["broadcaster"],
            "viewer": x["viewer"],
            "count": x["count"],
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


class RankingModel(tfrs.Model):
    def __init__(self, loss, ranking_conf):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=ranking_conf["unique_user_ids"]
                ),
                tf.keras.layers.Embedding(
                    len(ranking_conf["unique_user_ids"]) + 1, embedding_dimension
                ),
            ]
        )

        # Compute embeddings for broadcasters.
        self.broadcaster_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=ranking_conf["unique_broadcasters"]
                ),
                tf.keras.layers.Embedding(
                    len(ranking_conf["unique_broadcasters"]) + 1, embedding_dimension
                ),
            ]
        )

        # Compute predictions.
        self.score_model = tf.keras.Sequential(
            [
                # Learn multiple dense layers.
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # Make rating predictions in the final layer.
                tf.keras.layers.Dense(1),
            ]
        )

        self.task = tfrs.tasks.Ranking(
            loss=loss,
            metrics=[
                tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
                tf.keras.metrics.RootMeanSquaredError(),
            ],
        )

    def call(self, features):
        user_embeddings = self.user_embeddings(features["viewer"])
        broadcaster_embeddings = self.broadcaster_embeddings(features["broadcaster"])

        list_length = features["broadcaster"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1
        )
        concatenated_embeddings = tf.concat(
            [user_embedding_repeated, broadcaster_embeddings], 2
        )

        return self.score_model(concatenated_embeddings)

    def compute_loss(self, inputs, training=False):
        labels = inputs.pop("count")

        ranking_scores = self(inputs)

        return self.task(
            labels=labels,
            predictions=tf.squeeze(ranking_scores, axis=-1),
        )


class RankingModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.ranking_model = None

    def load_context(self, context):
        self.ranking_model = tf.saved_model.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.ranking_model


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
    train_accuracy = hist.history["factorized_top_k/top_100_categorical_accuracy"][-1]
    print(f"train_accuracy: {train_accuracy}")

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

    model_save_name = "model/train_gift_two_tower"
    tf.saved_model.save(index, model_save_name)

    output_conf = {
        "viewers_dim": len(unique_user_ids),
        "broadcasters_dim": len(unique_broadcasters),
        "viewer_embedding_dimension": conf["viewer_embedding_dimension"],
        "broadcaster_embedding_dimension": conf["broadcaster_embedding_dimension"],
        "train_accuracy": f"Top-100 accuracy of training: {train_accuracy:.4f}.",
        "unique_user_ids": ",".join(str(x) for x in unique_user_ids),
        "unique_broadcasters": ",".join(str(x) for x in unique_broadcasters),
    }
    file_path = "model/train_gift_two_tower/model_conf.json"
    print(file_path)
    with open(file_path, "w", encoding="utf8") as obj_file:
        json.dump(output_conf, obj_file, separators=(",", ":"))

    ### Ranking Model
    retrieval_model = tf.saved_model.load("model/train_gift_two_tower")
    json_file = open("model/train_gift_two_tower/model_conf.json")
    model_config = json.load(json_file)
    regex = re.compile("'(.*?)'")
    unique_user_ids = regex.findall(model_config["unique_user_ids"])
    unique_broadcasters = regex.findall(model_config["unique_broadcasters"])

    print(f"len(unique_user_ids) = {len(unique_user_ids)}")
    print(f"model_config['viewers_dim'] = {model_config['viewers_dim']}")
    print(f"len(unique_broadcasters) = {len(unique_broadcasters)}")
    print(f"model_config['broadcasters_dim'] = {model_config['broadcasters_dim']}")

    df = load_data_file_gift(local_file)
    retrieval_results = {
        "viewer": [],
        "broadcaster": [],
        "retrieval_score": [],
        "count": [],
    }
    for user_id in unique_user_ids:
        # for user_id in ['zoosk:de347500a97c284a84c1b14071f4c0cd', 'agged:5404088037']:
        scores, topk_broadcasters = retrieval_model(
            {
                "viewer": tf.constant([user_id]),
            },
        )
        topk_broadcasters = topk_broadcasters.numpy()[0][:100]
        scores = scores.numpy()[0][:100]
        retrieval_results["viewer"].append(user_id)
        d = dict(
            zip(
                df.loc[df["viewer"] == user_id, "broadcaster"],
                df.loc[df["viewer"] == user_id, "count"],
            )
        )
        user_score = []
        user_broadcaster = []
        user_count = []
        for v, b in zip(scores, topk_broadcasters):
            user_score.append(v)
            user_broadcaster.append(b.decode("utf-8"))
            if b.decode("utf-8") in d:
                user_count.append(d[b.decode("utf-8")])
            else:
                user_count.append(0.0)
        retrieval_results["retrieval_score"].append(user_score)
        retrieval_results["broadcaster"].append(user_broadcaster)
        retrieval_results["count"].append(user_count)
    retrieval_results_by_user_ds = tf.data.Dataset.from_tensor_slices(retrieval_results)
    cached_train = retrieval_results_by_user_ds.shuffle(100_000).batch(8192).cache()

    conf = defaultdict(dict)
    conf["broadcaster_embedding_dimension"] = args.broadcaster_embedding_dimension
    conf["viewer_embedding_dimension"] = args.viewer_embedding_dimension
    conf["batch_size"] = args.batch_size
    conf["learning_rate"] = args.learning_rate
    conf["epochs"] = 5
    conf["unique_user_ids"] = unique_user_ids
    conf["unique_broadcasters"] = unique_broadcasters

    # enable auto logging
    mlflow.set_experiment("gift listwise")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(
        run_name="Gift Model Experiments Using Two Tower Model"
    ) as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print(f"run_id: {run_id}")
        print(f"experiment_id: {experiment_id}")

        listwise_model = RankingModel(tfr.keras.losses.ListMLELoss(), conf)
        listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.01))
        hist = listwise_model.fit(cached_train, epochs=conf["epochs"], verbose=True)
        mlflow.log_param("size", len(unique_user_ids))
        mlflow.log_metric("listwise_ndcg_metric", hist.history["ndcg_metric"][-1])
        mlflow.log_metric(
            "root_mean_squared_error", hist.history["root_mean_squared_error"][-1]
        )
        mlflow.log_metric("total_loss", hist.history["total_loss"][-1])

        # save the model
        artifacts = {"model_path": "model"}
        tf.saved_model.save(listwise_model, artifacts["model_path"])
        mlflow.pyfunc.log_model(
            artifact_path=artifacts["model_path"],
            python_model=RankingModelWrapper(),
            artifacts=artifacts,
        )
        mlflow.end_run()


if __name__ == "__main__":
    main()
