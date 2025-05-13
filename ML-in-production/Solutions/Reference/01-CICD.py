# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="61413ef3-42e5-4aa9-85c4-8c1d38cc46b5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Model CI/CD
# MAGIC 
# MAGIC While deploying machine learning models can be challenging, it's just half of the battle. Machine learning engineers can incorporate software development into their machine learning development process.  
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Define continuous integration and continuous deployment
# MAGIC  - Relate CI/CD to machine learning models
# MAGIC  - Describe the benefits of following a CI/CD workflow for machine learning
# MAGIC  - Identify tools used within the CI/CD workflow
# MAGIC  - Access helpful learning resources for CI/CD for machine learning

# COMMAND ----------

# MAGIC %md <i18n value="04810706-c6b1-4e15-aa4e-f879e3c4144d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Continuous Integration and Continuous Deployment
# MAGIC 
# MAGIC Continuous integration and continuous deployment, commonly referred to as **CI/CD**, bridges the gap between development and operations.
# MAGIC 
# MAGIC #### Integration
# MAGIC 
# MAGIC Generally, **integration** refers to:<br>
# MAGIC 
# MAGIC * Pushing updated code or other artifacts to a central repository
# MAGIC * Running automated tests on those codes or artifacts
# MAGIC * If tests pass, then the code or artifacts can be integrated into the system
# MAGIC 
# MAGIC #### Delivery and Deployment
# MAGIC 
# MAGIC Once code or artifacts are integrated, the updated software or artifacts need to be **delivered** or **deployed**:<br>
# MAGIC 
# MAGIC * **Delivery**: the process by which a software application is automatically released, subject to the approver of a developer
# MAGIC * **Deployment**: a fully automated version of delivery
# MAGIC 
# MAGIC #### Continuous
# MAGIC 
# MAGIC You might be thinking: what makes these **continuous**? Generally speaking, **continuous** means in rapid succession &mdash; to efficiently and frequently integrate and deploy software.

# COMMAND ----------

# MAGIC %md <i18n value="1b08a7f0-d9bb-4633-ae3d-e3a54903558f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## CI/CD for Machine Learning
# MAGIC 
# MAGIC CI/CD is general to many types of software development, but it's also vital for **machine learning engineering**.
# MAGIC 
# MAGIC #### Integration for Machine Learning
# MAGIC 
# MAGIC Integrating machine learning work can frequently include updates to a projects codebase, but we're usually talking about integrating updated artifacts. More specifically, we want to integrate updated **models**.
# MAGIC 
# MAGIC This process usually follows a workflow like:<br>
# MAGIC 
# MAGIC * Build a new model for an existing solution
# MAGIC * Push the model to a testing stage in a centralized model repository
# MAGIC * Test the model against a series of tests (unit, integration, regression, performance, etc.)
# MAGIC * Move the model into the production machine learning system using the model repository
# MAGIC 
# MAGIC #### Deployment for Machine Learning
# MAGIC 
# MAGIC We've been talking about different methods of machine learning deployment throughout this course. Whether we deploy in **batch**, a continuous **stream**, or in **real-time**, continuous deployment would **frequently update the model or predictions being served to the end user**.
# MAGIC 
# MAGIC But when using CI/CD for machine learning, we aren't quite done once we deploy:<br>
# MAGIC 
# MAGIC * Models should be **continuously evaluated** to ensure their performance is not degrading over time
# MAGIC * If there performance degrades, we call this **drift**
# MAGIC * There are a variety of reasons why drift occurs and ways to detect drift, and we'll talk more about them in the next lesson

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="91bbbcf5-dbc3-43d7-b309-a5aec7c1c22c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/Model-Staleness.png" style="height: 450px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="69237e1c-636c-4faa-8901-64eeb1240c2a"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Benefits of CI/CD in Machine Learning
# MAGIC 
# MAGIC When we employ CI/CD practices in machine learning, we **close a fully automated development and deployment loop.** This provides a variety of benefits.
# MAGIC 
# MAGIC #### Time Savings
# MAGIC 
# MAGIC Without CI/CD, updating a production machine learning application would be a **long, human-dependent process**:<br><br>
# MAGIC 
# MAGIC 0. A data scientist decides to build a new model
# MAGIC 0. A data scientist manually builds the new model
# MAGIC 0. A data scientist evaluates the performance of their model
# MAGIC 0. A data scientist decides the new model should be put into production
# MAGIC 0. A data scientist tells the machine learning engineer that the new model should be put into production
# MAGIC 0. A machine learning engineer manually moves the model into a test environment
# MAGIC 0. A machine learning engineer deploys the model within the test environment
# MAGIC 0. A machine learning engineer runs a series of manual tests on the model within the test environment
# MAGIC 0. A machine learning engineer manually moves the model into a production environment
# MAGIC 0. A machine learning engineer deploys the model within the production environment
# MAGIC 0. A data scientist repeatedly tests the model's performance to determine when an update might be needed
# MAGIC 
# MAGIC This takes a lot of person hours! By following a CI/CD process, this time-consuming workflow can be automated. This provides the **benefits of a faster update cycle**:<br><br>
# MAGIC 
# MAGIC * More up-to-date models
# MAGIC * Limit the negative impact of faults like poor models
# MAGIC 
# MAGIC #### Consistency
# MAGIC 
# MAGIC By following an automated integration and deployment pattern, the decision points of the process will be **consistent** and **reproducible**. This means that:<br><br>
# MAGIC 
# MAGIC * Each model is built with the same target calculation
# MAGIC * Each model is put through the exact same tests
# MAGIC * Each model is integrated into the staging and production environments in the same way
# MAGIC * Each model is deployed in the exact same way
# MAGIC * Each model is continuously evaluated using the same standards for detecting drift
# MAGIC 
# MAGIC This can ensure that the **bias** of having different data scientists and machine learning engineers (or the same ones with less or more time) doesn't negatively affect the machine learning application.
# MAGIC 
# MAGIC #### Hotfixes and Rollbacks
# MAGIC 
# MAGIC Another benefit of continuously integrating and deploying new code is the ability to quickly correct any mistakes or problems. This can come in one of two ways:<br><br>
# MAGIC 
# MAGIC * **Hotfixes**: a small piece of code written to quickly correct a bug in a production software application
# MAGIC * **Rollbacks**: reverting the software application to the last properly functioning version

# COMMAND ----------

# MAGIC %md <i18n value="a77d37ff-6aa1-4a02-abb4-002ac0c638aa"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Azure ML Pipeline](https://files.training.databricks.com/images/ml-deployment/model-cicd.png)

# COMMAND ----------

# MAGIC %md <i18n value="241daf63-5043-4caf-99d1-a074412f45ec"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Tools for Model CI/CD
# MAGIC 
# MAGIC When employing CI/CD for machine learning, there are a variety of tools that could be helpful.
# MAGIC 
# MAGIC Generally, tools fall into the following categories:<br><br>
# MAGIC 
# MAGIC * **Orchestration**: control the flow of your applications
# MAGIC * **Git Hooks**: automatically run code when a particular event occurs in a git repository
# MAGIC * **Artifact Management**: manage your artifacts like packaged software or machine learning models
# MAGIC * **Environment Management**: manage the software resources available to your application
# MAGIC * **Testing**: develop tests to assess the validity and effectiveness of your model
# MAGIC * **Alerting**: notify important stakeholders when a specific event or test result occurs
# MAGIC 
# MAGIC Common tools for each of these categories are mentioned below: <br><br>
# MAGIC 
# MAGIC 
# MAGIC |                        | OSS Standard                     | Databricks               | AWS                                 | Azure               | Third Party                       |
# MAGIC |------------------------|----------------------------------|--------------------------|-------------------------------------|---------------------|-----------------------------------|
# MAGIC | **Orchestration**         | Airflow, Jenkins                 | Jobs, notebook workflows | CodePipeline, CodeBuild, CodeDeploy | DevOps, Data Factory |                                   |
# MAGIC | **Git Hooks**              |                                  | MLflow Webhooks          |                                     |                     | Github Actions, Gitlab, Travis CI |
# MAGIC | **Artifact Management**    | PyPI, Maven                      | MLflow Model Registry    |                                     |                     | Nexus                             |
# MAGIC | **Environment Management** | Docker, Kubernetes, Conda, pyenv |                          | Elastic Container Repository        | Container Registry  | DockerHub                         |
# MAGIC | **Testing**                | pytest                           |                          |                                     |                     | Sonar                             |
# MAGIC | **Alerting**               |                                  | Jobs                     | CloudWatch                          | Monitor             | PagerDuty, Slack integrations     |

# COMMAND ----------

# MAGIC %md <i18n value="d6203326-d8a9-4dab-bc9a-9ee5f6f0c93c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ML CI/CD Resources
# MAGIC 
# MAGIC The below resources are helpful when wanting to learn more about CI/CD for machine learning.
# MAGIC 
# MAGIC **Blogs**<br>
# MAGIC 
# MAGIC * <a href="https://databricks.com/blog/2017/10/30/continuous-integration-continuous-delivery-databricks.html" target="_blank">Continuous Integration and Continuous Delivery with Databricks</a>
# MAGIC * <a href="https://databricks.com/blog/2019/09/18/productionizing-machine-learning-from-deployment-to-drift-detection.html" target="_blank">Productionizing Machine Learning: From Deployment to Drift Detection</a>
# MAGIC * <a href="https://databricks.com/blog/2020/10/13/using-mlops-with-mlflow-and-azure.html" target="_blank">Using MLOps with MLflow and Azure</a>
# MAGIC * <a href="https://ml-ops.org/content/mlops-principles" target="_blank">MLOps Principles</a>
# MAGIC 
# MAGIC **Documentation**<br>
# MAGIC 
# MAGIC * <a href="https://docs.microsoft.com/en-us/azure/databricks/dev-tools/ci-cd/ci-cd-azure-devops" target="_blank">Continuous integration and delivery on Azure Databricks using Azure DevOps</a>
# MAGIC * <a href="https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning" target="_blank">MLOps Continuous Delivery</a>
# MAGIC * <a href="https://docs.databricks.com/dev-tools/ci-cd.html" target="_blank">Continuous Integration and Delivery on Databricks using Jenkins</a>
# MAGIC 
# MAGIC **Tools**<br>
# MAGIC 
# MAGIC * <a href="https://github.com/databrickslabs/cicd-templates" target="_blank">CI/CD Templates</a>
# MAGIC * <a href="https://github.com/databrickslabs/dbx" target="_blank">DBX library</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
