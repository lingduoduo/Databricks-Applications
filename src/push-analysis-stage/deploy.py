import mlflow.sagemaker as mfs

experiment_id = '0'
run_id = '7f857d1c8ebb44bd80bd26424d537317'
region = 'us-east-1'
aws_id = '882748442234'
arn = 'arn:aws:iam::882748442234:role/service-role/AmazonSageMaker-ExecutionRole-20210915T104260'

endpoint_name = 'lightgbm-gift'
model_uri = 'mlruns/%s/%s/artifacts/model' % (experiment_id, run_id)
tag_id = 'latest'
image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:1.26.1'

mfs.deploy(mode="create",
           app_name=endpoint_name,
           model_uri=model_uri,
           image_url=image_url,
           region_name=region,
           execution_role_arn=arn,
           instance_type='ml.m5.xlarge',
           instance_count=1,
           )

# mfs.deploy(mode="add",
#            app_name=endpoint_name,
#            model_uri=model_uri,
#            image_url=image_url,
#            region_name=region,
#            execution_role_arn=arn,
#            instance_type='ml.m5.xlarge',
#            instance_count=1,
#            )