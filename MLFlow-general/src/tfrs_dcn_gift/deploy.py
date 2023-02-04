import mlflow.sagemaker as mfs
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import numpy as np

experiment_id = '0'
run_id = '59f7f0e968094277855cb750decd9ce8'
region = 'us-east-1'
aws_id = '882748442234'
arn = 'arn:aws:iam::882748442234:role/service-role/AmazonSageMaker-ExecutionRole-20210915T104260'

endpoint_name = 'tfrs-gift-dcn'
model_uri = 'mlruns/%s/%s/artifacts/model' % (experiment_id, run_id)
tag_id = 'latest'
image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:1.26.1'
# image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:latest'

# mfs.deploy(mode="create",
#            app_name=endpoint_name,
#            model_uri=model_uri,
#            image_url=image_url,
#            region_name=region,
#            execution_role_arn=arn,
#            instance_type='ml.m5.xlarge',
#            instance_count=1,
#            )

# directly deploy tensorflow model to SageMaker
role = sagemaker.get_execution_role()
bucket_name = "mlflow-sagemaker-us-east-1-882748442234/tfrs-gift-dcn-model-kkq3zhqatng2nyid0ramva"
sagemaker_model = TensorFlowModel(
    model_data=f"s3://{bucket_name}/model.tar.gz",
    role=arn,
    framework_version="2.9",
)
predictor = sagemaker_model.deploy(initial_instance_count=1, instance_type="ml.m5.2xlarge")

# Validate the endpoint for use
input_example = {
    "viewer": np.array(["kik:user:unknown"]),
    "broadcaster": ["kik:user:unknown"],
    "product_name": ["Rose"],
    "order_time": ["10"],
}
pred = predictor.predict(input_example)
print(pred)