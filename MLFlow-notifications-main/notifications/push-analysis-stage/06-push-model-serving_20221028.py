# Databricks notebook source
dbutils.credentials.showCurrentRole()

# COMMAND ----------

dbutils.fs.ls('/mnt')

# COMMAND ----------

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Download model artifacts

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

# COMMAND ----------

model_name = "Test-Stage-Model"

# COMMAND ----------

client = MlflowClient()
model_version_info = client.get_latest_versions(model_name, stages=["production"])
model_version = model_version_info[0].version
model_version

# COMMAND ----------

model_uri = client.get_model_version_download_uri(model_name, model_version)
print("Download URI: {}".format(model_uri))

# COMMAND ----------

model_path = '/dbfs/Users/{0}/ml-artifacts/model-{1}-{2}'.format(user, model_name, model_version)
model_path

# COMMAND ----------

ModelsArtifactRepository(model_uri).download_artifacts(artifact_path=model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Patch requisite packages for Model serving

# COMMAND ----------

# xgbc_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
# os.makedirs(xgbc_temp_dir)

# xgbc_client = MlflowClient()
# xgbc_model_env_path = xgbc_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", xgbc_temp_dir)
# xgbc_model_env_str = open(xgbc_model_env_path)
# xgbc_parsed_model_env_str = yaml.load(xgbc_model_env_str, Loader=yaml.FullLoader)

# xgbc_parsed_model_env_str["dependencies"][-1]["pip"].append(f"xgboost=={xgboost.__version__}")

# with open(xgbc_model_env_path, "w") as f:
#     f.write(yaml.dump(xgbc_parsed_model_env_str))
    
# xgbc_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=xgbc_model_env_path, artifact_path="model")
# shutil.rmtree(xgbc_temp_dir)

# COMMAND ----------

run_id = '1cd4da38afc24117a9f4637708b24267'
model_path = '/dbfs/Users/lhuang@themeetgroup.com/data'
# model_env_path = client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", model_path)
model_env_path = client.download_artifacts(run_id,  "model/conda.yaml", model_path)

# COMMAND ----------

dbutils.fs.ls('dbfs:/Users/lhuang@themeetgroup.com/data/model')

# COMMAND ----------

import xgboost
model_env_str = open(model_env_path)
parsed_model_env_str = yaml.load(model_env_str, Loader=yaml.FullLoader)

parsed_model_env_str["dependencies"][-1]["pip"].append(f"xgboost=={xgboost.__version__}")

# COMMAND ----------

with open(model_env_path, "w") as f:
    f.write(yaml.dump(parsed_model_env_str))
    
client.log_artifact(run_id=run_id, local_path=model_env_path, artifact_path="model")
# shutil.rmtree(model_path)

# COMMAND ----------

logged_model = f'runs:/{run_id}/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

model_path = '/dbfs/Users/{0}/ml-artifacts/model-{1}-{2}'.format(user, 'test', run_id)
mlflow.sklearn.save_model(sk_model=model, path=model_path)

# COMMAND ----------

display(dbutils.fs.ls(model_path.replace('/dbfs', 'dbfs:')))

# COMMAND ----------

import os

# COMMAND ----------

os.listdir('/dbfs/Users/lhuang@themeetgroup.com/ml-artifacts/')

# COMMAND ----------


