# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(**helper_arguments) # Create the DA object
DA.reset_environment()                   # Reset by removing databases and files from other lessons
DA.init(install_datasets=True,           # Initialize, install and validate the datasets
        create_db=True)                  # Continue initialization, create the user-db

DA.paths.datasets_path = DA.paths.datasets.replace("dbfs:/", "/dbfs/")
DA.paths.working_path = DA.paths.working_dir.replace("dbfs:/", "/dbfs/")

if dbgems.get_notebook_path().endswith("/01-Experimentation/01-Feature-Store"):
    DA.paths.airbnb = f"{DA.paths.working_dir}/airbnb/airbnb.delta"

DA.cleaned_username = re.sub("[^a-zA-Z0-9]", "_", DA.username.lower().split("@")[0])
DA.init_mlflow_as_job()

DA.conclude_setup()                      # Conclude setup by advertising environmental changes

