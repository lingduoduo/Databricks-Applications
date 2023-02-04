# in case this is run outside of conda environment with python2
import argparse
import os
import sys
from collections import defaultdict

import mlflow
import mlflow.keras
import mlflow.pyfunc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback, mlflow_mixin


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


def parse_args():
    parser = argparse.ArgumentParser(description="Gift model using DCM")
    parser.add_argument(
        "--embedding_dimension", default=96, type=int, help="embedding_dimension"
    )
    parser.add_argument("--batch_size", default=16384, type=int, help="batch_size")
    parser.add_argument(
        "--learning_rate", default=0.05, type=float, help="learning_rate"
    )
    return parser.parse_args()


@mlflow_mixin
def finetuning_dcn_model(conf, data_train, data_test):
    # Train the Model.
    model = DCN(
        conf=conf,
        use_cross_layer=True,
        deep_layer_sizes=[192, 192],
        projection_dim=None,
    )

    mlflow.tensorflow.autolog()
    model.compile(optimizer=tf.keras.optimizers.Adam(conf["learning_rate"]))
    trainer = model.fit(data_train, epochs=conf["epochs"], verbose=False)
    # metrics = model.evaluate(data_test, return_dict=True)
    # print(f"metrics: {metrics}")

    metrics = {"loss": "val_cross_entropy", "f1": "val_f1"}
    trainer.finetune(model, datamodule = datamodule, strategy = config['finetuning_strategies'])
    mlflow.log_param('batch_size', conf['batch_size'])
    mlflow.set_tag('pipeline_step', __file__)


def main():
    # parse command-line arguments
    args = parse_args()
    conf = defaultdict(dict)
    conf["embedding_dimension"] = args.embedding_dimension
    conf["batch_size"] = args.batch_size
    conf["learning_rate"] = args.learning_rate
    conf["epochs"] = 5
    conf["deep_layer_sizes"] = [192, 192]
    conf["str_features"] = ["broadcaster", "viewer", "product_name", "order_time"]
    conf["int_features"] = []
    conf["label_name"] = "count"

    # Fetch the data
    print(os.getcwd())
    if "src.tfrs_dcn_gift" in os.getcwd():
        filename = "src/csv/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    elif "src/tfrs_dcn_gift" in os.getcwd():
        filename = "../csv/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    else:
        filename = "s3://tmg-machine-learning-models-dev/for-you-payer-training-data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"

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

    # Fetch feature and vocabularies
    features = ["viewer", "broadcaster", "product_name", "order_time"]
    vocabularies = {}
    for idx, feature in enumerate(features):
        print(f"{idx}: {feature}")
        vocabularies[feature] = feature_mapping(gift, feature)
    conf["vocabularies"] = vocabularies



if __name__ == "__main__":
    print(__package__)
    if __package__ is None:
        sys.path.append("src.tfrs_dcn_gift")
    main()
