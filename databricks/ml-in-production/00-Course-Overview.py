# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="c83d51d6-428b-4691-82ea-778976cde46b"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Machine Learning in Production
# MAGIC ### Managing the Complete Machine Learning Lifecycle with MLflow, Deployment and CI/CD
# MAGIC 
# MAGIC In this 1-day course, machine learning engineers, data engineers, and data scientists learn the best practices for managing the complete machine learning lifecycle from experimentation and model management through various deployment modalities and production issues. Students begin with end-to-end reproducibility of machine learning models using MLflow including data management, experiment tracking, and model management before deploying models with batch, streaming, and real time as well as addressing related monitoring, alerting, and CI/CD issues. Sample code accompanies all modules and theoretical concepts.
# MAGIC 
# MAGIC First, this course explores managing the experimentation process using MLflow with a focus on end-to-end reproducibility including data, model, and experiment tracking. Second, students operationalize their models by integrating with various downstream deployment tools including saving models to the MLflow model registry, managing artifacts and environments, and automating the testing of their models. Third, students implement batch, streaming, and real time deployment options. Finally, additional production issues including continuous integration, continuous deployment are covered as well as monitoring and alerting.
# MAGIC 
# MAGIC By the end of this course, you will have built an end-to-end pipeline to log, deploy, and monitor machine learning models. This course is taught entirely in Python.
# MAGIC 
# MAGIC ## Lessons
# MAGIC 
# MAGIC | Time | Lesson &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 30m  | **Introductions & Setup**                               | *Registration, Courseware & Q&As* |
# MAGIC | 30m    | **ML in Production Overview**    | Introducing the full end-to-end ML lifecycle |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 20m    | **[Experimentation - Feature Store]($./01-Experimentation)**    | [Manage data with Delta & Databricks Feature Store]($./01-Experimentation/01-Feature-Store) |
# MAGIC | 40m  | **[Experimentation - Experiment Tracking & Lab]($./01-Experimentation)** | [Track ML experiment with MLflow]($./01-Experimentation/02-Experiment-Tracking) </br> [Experiment Tracking Lab]($./01-Experimentation/Labs/02-Experiment-Tracking-Lab) | 
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 30m  | **[Experimentation - Advanced Experiment Tracking & Lab]($./01-Experimentation)** | [Advanced Experiment Tracking]($./01-Experimentation/03-Advanced-Experiment-Tracking) </br> [Advanced Experiment Tracking Lab (Optional)]($./01-Experimentation/Labs/03-Advanced-Experiment-Tracking-Lab) | 
# MAGIC | 30m    | **[Model Management - MLflow Models & Lab]($./02-Model-Management)**    | [Model management with MLflow]($./02-Model-Management/01-Model-Management) </br> [Model managment lab]($./02-Model-Management/Labs/01-Model-Management-Lab) |
# MAGIC |  10m | **Break**                                               ||
# MAGIC | 35m  | **[Model Management - Model Registry]($./02-Model-Management)**       | [Register, version, and deploy models with MLflow]($./02-Model-Management/02-Model-Registry) |
# MAGIC | 25m  | **[Model Management - Webhooks]($./02-Model-Management)**      | [Create a testing job and a webhook of registered model]($./02-Model-Management/03a-Webhooks-and-Testing) </br> [Automated Testing]($./02-Model-Management/03b-Webhooks-Job-Demo)|
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 60m |**[Deployment Paradigms]($./03-Deployment-Paradigms)** | [Batch]($./03-Deployment-Paradigms/01-Batch)</br> [Real time]($./03-Deployment-Paradigms/02-Real-Time)</br> [Streaming (Reference)]($./Reference/03-Streaming-Deployment)</br> [Labs]($./03-Deployment-Paradigms/Labs)|
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 60m  | **[Production]($./04-Production)**  | [Monitoring]($./04-Production/01-Monitoring)</br> [Monitoring Lab]($./04-Production/Labs/01-Monitoring-Lab)</br>[Alerting (Reference)]($./Reference/02-Alerting)|
# MAGIC 
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Experience with Python (**`pandas`**, **`sklearn`**, **`numpy`**)
# MAGIC - Background in machine learning and data science
# MAGIC 
# MAGIC ## Cluster Requirements
# MAGIC - See your instructor for specific requirements
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> **Certain features used in this course, such as the notebooks API and model registry, are only available to paid or trial subscription users of Databricks.**
# MAGIC If you are using the Databricks Community Edition, click the **`Upgrade`** button on the landing page <a href="https://accounts.cloud.databricks.com/registration.html#login" target="_blank">or navigate here</a> to start a free trial.

# COMMAND ----------

# MAGIC %md <i18n value="35c71f4c-1ab2-4d02-a07f-c144d7fe7dfa"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the start of each lesson (see the next cell).

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="1464bb0e-c32c-4d92-b8a8-7d2e7767205f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Agile Data Science
# MAGIC 
# MAGIC Deploying machine learning models into production comes with a wide array of challenges, distinct from those data scientists face when they're initially training models.  Teams often solve these challenges with custom, in-house solutions that are often brittle, monolithic, time consuming, and difficult to maintain.
# MAGIC 
# MAGIC A systematic approach to the deployment of machine learning models results in an agile solution that minimizes developer time and maximizes the business value derived from data science.  To achieve this, data scientists and data engineers need to navigate various deployment solutions as well as have a system in place for monitoring and alerting once a model is out in production.
# MAGIC 
# MAGIC The main deployment paradigms are as follows:<br><br>
# MAGIC 
# MAGIC 1. **Batch:** predictions are created and stored for later use, such as a database that can be queried in real time in a web application
# MAGIC 2. **Streaming:** data streams are transformed where the prediction is needed soon after it arrives in a data pipeline but not immediately
# MAGIC 3. **Real time:** normally implemented with a REST endpoint, a prediction is needed on the fly with low latency
# MAGIC 4. **Mobile/Embedded:** entails embedding machine learning solutions in mobile or IoT devices and is outside the scope of this course
# MAGIC 
# MAGIC Once a model is deployed in one of these paradigms, it needs to be monitored for performance with regards to the quality of predictions, latency, throughput, and other production considerations.  When performance starts to slip, this is an indication that the model needs to be retrained, more resources need to be allocated to serving the model, or any number of improvements are needed.  An alerting infrastructure needs to be in place to capture these performance issues.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
