import argparse
import os

# In[1]:
import warnings
from typing import Tuple, Dict, Iterable

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Gift model using DCM")
    parser.add_argument(
        "--experiment_name", default="gift_model", type=str, help="experiment_name"
    )
    parser.add_argument("--n_estimators", default=300, type=int, help="n_estimators")
    parser.add_argument("--num_leaves", default=164, type=int, help="num_leaves")
    parser.add_argument(
        "--learning_rate", default=0.1, type=float, help="learning_rate"
    )

    parser.add_argument("--max_depth", default=-1, type=float, help="max_depth")
    return parser.parse_args()


def feature_encoder(training_data, encoding_features):
    feature_mappings = {}
    for c in encoding_features:
        temp = training_data[c].astype("category").cat
        training_data[c] = temp.codes + 1
        feature_mappings[c] = {cat: n for n, cat in enumerate(temp.categories, start=1)}
    return training_data, feature_mappings


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """return distinct train and test sets"""
    return train_test_split(df, random_state=0, test_size=0.2)


def decompose(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """break down data into features, labels and weights"""
    return df.drop(["y", "w"], axis=1), df.y, df.w


def ungroup(X: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """expand all repeated columns"""
    return X.reindex(X.index.repeat(w))


def group(X: pd.DataFrame) -> pd.DataFrame:
    """append weight to pandas dataframe by groupby all columns"""
    return X.groupby(list(X.columns)).size().to_frame("w").reset_index()


def etl(
    X: pd.DataFrame, y: pd.Series, w: pd.Series = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """transform the data"""
    X = X.assign(y=y)
    if w is not None:
        X_train, X_test = split(ungroup(X, w))
        return group(X_train), group(X_test)
    else:
        X["w"] = 1
        return split(X)


def predict(model: lgb.Booster, X: pd.DataFrame, regression: bool = False) -> pd.Series:
    """prediction wrapper"""
    return model.predict(X) if regression else model.predict_proba(X)[:, 1]


def prediction_metrics(var_y, var_y_pred, weights=None):
    if weights is None:
        weights = np.ones(len(var_y))
    avg = np.average(var_y, weights=weights)
    pred_avg = np.average(var_y_pred, weights=weights)
    bias2 = np.average(var_y - var_y_pred, weights=weights) ** 2

    var = (
        np.average(var_y_pred**2, weights=weights)
        - np.average(var_y_pred, weights=weights) ** 2
    )
    square_error = mean_squared_error(var_y, var_y_pred, sample_weight=weights)
    abs_err = mean_absolute_error(var_y, var_y_pred, sample_weight=weights)
    r2_scores = r2_score(var_y, var_y_pred, sample_weight=weights)
    return dict(
        avg=avg,
        pred_avg=pred_avg,
        bias2=bias2,
        var=var,
        square_error=square_error,
        abs_err=abs_err,
        abs_err_ratio=abs_err / avg,
        r2_score=r2_scores,
    )


def evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    y_pred: pd.Series,
    regression: bool = False,
) -> Dict[str, any]:
    """calculate evaluation metrics"""
    metrics = {"records": str(len(y)), "weights": str(sum(w))}

    if not regression:
        metrics["log_loss"] = log_loss(y, y_pred, sample_weight=w)
        metrics["roc_auc_score"] = roc_auc_score(y, y_pred, sample_weight=w)
    else:
        metrics["raw_accuracy"] = prediction_metrics(y, y_pred, weights=w)

    return metrics


def build_model(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series = None,
    categoricals: Iterable[str] = [],
    feature_mappings: Dict = {},
    params: Dict[str, any] = {},
    regression: bool = False,
) -> Tuple[lgb.LGBMModel, Dict[str, any], pd.DataFrame, pd.DataFrame]:
    """take the sql data and output model and metadata"""
    metadata = {
        "model_type": "lightgbm",
        "categorical_feature": categoricals,
        "mappings": feature_mappings,
        "early_stop": 300,
        "max_round": 10000,
    }

    train_df, test_df = etl(X, y, w)
    X, y, w = decompose(train_df)

    model = (lgb.LGBMRegressor if regression else lgb.LGBMClassifier)(
        random_state=0, **params
    )
    model.fit(X, y, w, categorical_feature=categoricals)
    train_df["y_pred"] = predict(model, X, regression)

    metadata["train_accuracy"] = evaluate(X, y, w, train_df["y_pred"], regression)
    metadata["train_accuracy"]["data_type"] = "train"

    X, y, w = decompose(test_df)
    test_df["y_pred"] = predict(model, X, regression)
    metadata["test_accuracy"] = evaluate(X, y, w, test_df["y_pred"], regression)
    metadata["test_accuracy"]["data_type"] = "test"
    return model, metadata, train_df, test_df


def main():
    # parse command-line arguments
    args = parse_args()

    local_file = "csv/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    if not os.path.exists(local_file) and not os.path.isfile(local_file):
        filename = (
            "s3://for-you-payer-training-data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
        )
    else:
        filename = local_file

    df = pd.read_csv(filename)
    FEATURES = ["broadcaster_id", "viewer_id", "product_name", "ordered_time"]
    df_filled, feature_mappings = feature_encoder(df, FEATURES)
    df_filled["weight"] = 1
    nrow = len(df_filled)

    # params = {
    #     "boosting": "gbdt",
    #     "metric": ["mse", "mae"],
    #     # 'metric' : 'map',
    #     "objective": "regression",
    #     "learning_rate": 0.017,
    #     "max_depth": -1,
    #     "min_child_samples": 20,
    #     "max_bin": 255,
    #     "subsample": 0.85,
    #     "subsample_freq": 10,
    #     "colsample_bytree": 0.8,
    #     "min_child_weight": 0.001,
    #     "subsample_for_bin": 200000,
    #     "min_split_gain": 0,
    #     "reg_alpha": 0,
    #     "reg_lambda": 0,
    #     "num_leaves": 51,
    #     "nthread": 10,
    #     # 'is_unbalance': True,
    # }

    # enable auto logging
    # mlflow.lightgbm.autolog()

    pred_model, pred_metadata, train_df, test_df = build_model(
        X=df_filled[FEATURES],
        y=df_filled["cnt"],
        w=df_filled["weight"],
        categoricals=FEATURES,
        feature_mappings=feature_mappings,
        params={
            "n_estimators": args.n_estimators,
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
        },
        regression=True,
    )
    train_acc = pred_metadata["train_accuracy"]
    print(train_acc)
    test_acc = pred_metadata["test_accuracy"]
    print(test_acc)
    boost = pred_model.booster_

    # mlflow.set_experiment("train_gift-lightgbm")
    with mlflow.start_run(run_name="Gift Model Experiments using Lightgbm") as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.log_param("size", nrow)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("num_leaves", args.num_leaves)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.end_run()
        print(f"artfact_uri = {mlflow.get_artifact_uri()}")
        print(f"runid: {run_id}")
        print(f"experimentid: {experiment_id}")


if __name__ == "__main__":
    main()
