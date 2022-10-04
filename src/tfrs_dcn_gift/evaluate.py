import numpy as np
import tensorflow as tf


def predict_ranking_model(viewer, broadcaster, product_name, order_time, path):
    model = tf.saved_model.load(f"{path}")
    score = model(
        {"viewer": np.array([viewer]),
         "broadcaster": [broadcaster],
         "product_name": [product_name],
         "order_time": [order_time]
         }
    ).numpy()
    return score
