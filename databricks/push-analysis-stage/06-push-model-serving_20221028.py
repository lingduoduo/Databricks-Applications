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

# Fetch model uri
model_uri = "models:/{model_name}/production".format(model_name=model_name)
model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# clear any models in the path
# modelpath = "/dbfs/mnt/tmg-stage-ml-outputs/model-%s-%s" % (model_details.name, model_details.version)
modelpath = "/dbfs/mnt/tmg-stage-ml-artifacts/model-%s-%s" % ("newtest-prod", 10)
print(modelpath)
shutil.rmtree(modelpath, ignore_errors=True)

# COMMAND ----------

# Save models to DBFS
mlflow.sklearn.save_model(model, modelpath)

# COMMAND ----------

display(dbutils.fs.ls(modelpath.replace('/dbfs', 'dbfs:')))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download artifacts to local path and upload to S3

# COMMAND ----------

local_modelpath = '/dbfs/Users/{0}/ml-artifacts/model-{1}-{2}'.format(user, model_name, model_version)
print(local_modelpath)
shutil.rmtree(local_modelpath, ignore_errors=True)

# COMMAND ----------

mlflow.sklearn.save_model(model, local_modelpath)

# COMMAND ----------

display(dbutils.fs.ls(local_modelpath.replace('/dbfs', 'dbfs:')))

# COMMAND ----------

modelpath = "/dbfs/mnt/tmg-stage-ml-outputs/model-%s-%s" % (model_name, model_version)
print(modelpath)
shutil.rmtree(modelpath, ignore_errors=True)

# COMMAND ----------

modelpath = "/dbfs/mnt/tmg-stage-ml-outputs/model-%s-%s/upload" % (model_name, model_version)
dbutils.fs.mkdirs(modelpath)

# COMMAND ----------

# MAGIC %fs 
# MAGIC ls dbfs:/mnt/tmg-stage-ml-outputs/

# COMMAND ----------

dbutils.fs.mv(local_modelpath, modelpath)

# COMMAND ----------

dbutils.fs.ls(local_modelpath.replace('/dbfs', 'dbfs:'))

# COMMAND ----------

dbutils.fs.ls(dbutils.fs.ls(modelpath.replace('/dbfs', 'dbfs:')), )

# COMMAND ----------

modelpath

# COMMAND ----------

display(dbutils.fs.ls('/mnt/tmg-stage-ml-outputs'))

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



# COMMAND ----------

import os
import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

model_name = "Test-Stage-Model"
model_stage = "Production"  # Should be either 'Staging' or 'Production'

os.makedirs(local_modelpath+"model", exist_ok=True)
local_path = ModelsArtifactRepository(
    f'models:/{model_name}/{model_stage}').download_artifacts("", dst_path=local_modelpath)

print(f'{model_stage} Model {model_name} is downloaded at {local_path}')

# COMMAND ----------

local_modelpath

# COMMAND ----------

prod_model = mlflow.pyfunc.load_model(local_modelpath)

# COMMAND ----------

df = spark.table("ml_push.silver_l7_push_meetme_train")
pdf = df.toPandas()
X_train=  pdf.drop("open_flag", axis=1)

# COMMAND ----------

sum(model.predict(X_train))/len(X_train)

# COMMAND ----------


