import argparse
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM Model for Gift")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="learning rate to update step size at each boosting step (default: 0.1)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="number of estimators to create trees (default: 300)",
    )
    parser.add_argument(
        "--num_leaves",
        type=int,
        default=256,
        help="number of leaves to create trees (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=-1,
        help="maximum depth to create trees (default: -1)",
    )
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
    categoricals: Iterable[str] = None,
    feature_mappings: Dict = None,
    params: Dict[str, any] = None,
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

    metadata["train_accuracy"] = evaluate(y, w, train_df["y_pred"], regression)
    metadata["train_accuracy"]["data_type"] = "train"

    X, y, w = decompose(test_df)
    test_df["y_pred"] = predict(model, X, regression)
    metadata["test_accuracy"] = evaluate(y, w, test_df["y_pred"], regression)
    metadata["test_accuracy"]["data_type"] = "test"
    return model, metadata, train_df, test_df


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    local_file = "src/csv/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    if not os.path.exists(local_file) and not os.path.isfile(local_file):
        filename = "s3://tmg-machine-learning-models-dev/for-you-payer-training-data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    else:
        filename = local_file

    df = pd.read_csv(filename)
    FEATURES = ["broadcaster_id", "viewer_id", "product_name", "ordered_time"]
    df_filled, feature_mappings = feature_encoder(df, FEATURES)
    df_filled["weight"] = 1

    # enable auto logging
    # mlflow.set_experiment("Baseline_Predictions")
    mlflow.lightgbm.autolog()
    # with mlflow.start_run(run_name='lightgbm_gift_model_baseline') as run:
    with mlflow.start_run() as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print(f"run_id: {run_id}")
        print(f"experiment_id: {experiment_id}")

        # train model
        params = {
            "n_estimators": args.n_estimators,
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
        }
        pred_model, pred_metadata, train_df, test_df = build_model(
            X=df_filled[FEATURES],
            y=df_filled["cnt"],
            w=df_filled["weight"],
            categoricals=FEATURES,
            feature_mappings=feature_mappings,
            params=params,
            regression=True,
        )

        # evaluate model
        print(pred_metadata["test_accuracy"])
        test_df_r2_score = pred_metadata["test_accuracy"]["raw_accuracy"]["r2_score"]
        test_df_mse = pred_metadata["test_accuracy"]["raw_accuracy"]["square_error"]

        # log metrics
        mlflow.log_metrics(
            {"r2_score": test_df_r2_score, " mean_squared_error": test_df_mse}
        )
        mlflow.end_run()

    print(f"train_df: {len(train_df)}")
    print(f"test_df: {len(test_df)}")
    boost = pred_model.booster_
    imps = pd.DataFrame(
        {"feature": boost.feature_name(), "importance": boost.feature_importance()}
    )
    print(imps.sort_values("importance", ascending=False))

    # Log the model manually
    input_example = df_filled[FEATURES].sample(n=1)
    # signature = infer_signature(df_filled[FEATURES], df_filled["cnt"])
    logged_model = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
    # mlflow.lightgbm.log_model(
    #     pred_model,
    #     artifact_path = logged_model,
    #     signature = signature,
    #     input_example = input_example
    # )
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    pred = loaded_model.predict(input_example)
    print(pred)


if __name__ == "__main__":
    main()
