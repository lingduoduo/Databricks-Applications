# Databricks notebook source
# MAGIC %pip install lightning-flash[all]==0.5.0
# MAGIC %pip install transformers==4.9.2
# MAGIC %pip install boto3==1.19.7
# MAGIC %pip install pytorch-lightning==1.4.9
# MAGIC %pip install datasets==1.9.0

# COMMAND ----------

import torch
import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

# COMMAND ----------

download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "/dbfs/tmp/ML/pl-flash-data/")

# COMMAND ----------

# MAGIC %sh mkdir -p /dbfs/tmp/ML/pl-flash-data/; wget -O /dbfs/tmp/ML/pl-flash-data/imdb.zip https://pl-flash-data.s3.amazonaws.com/imdb.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /tmp/ML/pl-flash-data/imdb.zip

# COMMAND ----------

display(dbutils.fs.ls("/tmp/ML/pl-flash-data"))

# COMMAND ----------

dbutils.fs.ls("dbfs:/tmp/ML/pl-flash-data/imdb/")

# COMMAND ----------

dbutils.fs.cp("/tmp/ML/pl-flash-data/imdb.zip", "file:/tmp/file.zip")

# COMMAND ----------

dbutils.fs.ls("dbfs:/tmp/ML/pl-flash-data/imdb/train.csv")

# COMMAND ----------

dbutils.fs.ls("dbfs:/tmp/ML/pl-flash-data/imdb/test.csv")

# COMMAND ----------

with open('/dbfs/tmp/ML/pl-flash-data/imdb/valid.csv', 'r') as f:
  for line in f:
    print(line)

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/root/.cache/huggingface/datasets/csv/default-8ba028e419a586fe/0.0.0/")

# COMMAND ----------

print('### download IMDb data to local folder')
datamodule = TextClassificationData.from_csv(
    input_fields="review",
    target_fields="sentiment",
    train_file="/dbfs/tmp/ML/pl-flash-data/imdb/train.csv",
    val_file="/dbfs/tmp/ML/pl-flash-data/imdb/valid.csv",
    test_file="/dbfs/tmp/ML/pl-flash-data/imdb/test.csv"
)

# COMMAND ----------

datamodule.analysis.dataframe()

# COMMAND ----------

classifier_model = TextClassifier(backbone="prajjwal1/bert-tiny", num_classes=datamodule.num_classes)
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())


# COMMAND ----------

print('### define the trainer')
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())


# COMMAND ----------

EXPERIMENT_NAME = "dl_bert_model"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

# COMMAND ----------

mlflow.pytorch.autolog()

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="bert_model"):
    trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")
    print('### get classifier test results')
    trainer.test()

# COMMAND ----------

print('### get prediction outputs for two sample sentences')
predictions = classifier_model.predict(
    [
        "Best movie I have seen.",
        "What a movie!",
    ]
)
print(predictions)

# COMMAND ----------


