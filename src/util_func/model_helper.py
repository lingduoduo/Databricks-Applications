import mlflow
import numpy as np
import tensorflow as tf


def register_model(run, model_name, client = mlflow.tracking.MlflowClient()):
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass
    source = f"{run.info.artifact_uri}/model"
    client.create_model_version(model_name, source, run.info.run_id)


def predict_retrieval_model(viewer, path):
    model = tf.saved_model.load(f"model/{path}")
    scores, broadcasters = model(
        {
            "viewer": tf.constant([viewer]),
        }
    )
    scores = scores.numpy()[0]
    broadcasters = broadcasters.numpy()[0]
    preds = {}
    for i in range(len(scores)):
        preds[str(broadcasters[i])] = str(scores[i])
    print(",".join(preds.keys()))
    return preds


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


def predict_listwise_ranking_model(viewer, path):
    retrieval_model = tf.saved_model.load(f"model/train_gift_two_tower")
    test_data = {"viewer": [], "broadcaster": [], "retrieval_score": []}
    scores, topk_broadcasters = retrieval_model({"viewer": tf.constant([viewer]), })
    topk_broadcasters = topk_broadcasters.numpy()[0][:100]
    scores = scores.numpy()[0][:100]
    test_data["viewer"].append(viewer)
    test_data["broadcaster"].append(topk_broadcasters)
    test_data["retrieval_score"].append(scores)
    test_data_ds = tf.data.Dataset.from_tensor_slices(test_data)

    ranking_model = tf.saved_model.load(f"model/{path}")
    test_data_ds_cached = test_data_ds.batch(1)
    for cached_test_batch in test_data_ds_cached:
        scores = ranking_model(cached_test_batch)
    scores = tf.reshape(scores, [100, ]).numpy()
    return list(zip(topk_broadcasters, scores))
