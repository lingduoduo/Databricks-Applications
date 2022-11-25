# Databricks notebook source
# MAGIC %md
# MAGIC ## Problem Definition & Discussion

# COMMAND ----------

# MAGIC %md
# MAGIC **General objective**
# MAGIC 
# MAGIC The objective of this project is to build a sentiment model, use mlflow for its lifecycle management, incorporate the sentiment model into the pipeline to output sentiment, and evaluate the correlation between sentiment and the prices of 15 cryptocurrencies. In our case, the sentiment model follows a supervised, multi-class classification approach. 
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Evaluation metric of semantic model**
# MAGIC 
# MAGIC We will use accuracy to evaluate model performance if there is no class imbalance in the data. Alternatively, we would use the F-score (https://en.wikipedia.org/wiki/F-score) using precision and recall (\\(F_\beta = (1+\beta^2) \cdot \frac{precision \cdot recall}{(\beta^2 \cdot precision)+recall} \\)) as evaluation metric if class imbalance is present in the data (other alternatives in this case would include ROC-AUC or Precision-Recall-AUC). 
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Overview of potential algorithms**
# MAGIC 
# MAGIC Since sentiment analysis is a problem of great practical relevance, it is no surprise that multiple ML strategies can be found for it:
# MAGIC * Sentiment lexicons algorithm: these algorithms compare each word in a tweet to a database of words that are labeled as having positive or negative sentiment. A tweet with more positive words than negative would be scored as a positive. One with more negative words would be scored as a negative, and if there were no positive/negative words or the same number, it would be scored as neutral. However, the performance of this strategy greatly depends on the quality of the database of words (which has to be well maintained) and the context if the words is not considered at all. While this approach is somewhat straightforward, it performs poorly in general.
# MAGIC * Off-the-shelf sentiment analysis systems: Amazon Comprehend, Google Cloud Services, and Stanford Core NLP do not require any great preprocessing of the data and you can directly start the prediction "out of the box", i.e. you supply the text and the system calculates the sentiment. However, as downside, these systems are somewhat limited regarding fine-tuning for the underyling use-case (retraining might be needed to adjust the model performance).
# MAGIC * 'Classical' ML algorithms: application of a manually defined NLP pipeline using supervised classifiers like Logistic Regression, Random Forest, Support Vector Machine, Naive Bayes or Xgboost for the sentiment prediction problem allows to consider the details of the use-case (e.g. domain, language). One of the challenges is to find a proper embedding of the unstructured data, i.e. text. However, those classical ML algorithms are reported to perform surprisingly well for sentiment analysis.  
# MAGIC * Deep learning algorithms: in general, deep learning is considered to perform very well for unstructured data like text. Thus, many pre-trained neural networks for word embeddings and sentiment prediction already exist and can be fine-tuned to the underlying use-case. In general, this strategy is very promising.
# MAGIC 
# MAGIC We will focus on classical ML algorithms and deep learning algorithms for this project since they are supposed to be the most promising.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Challenges**
# MAGIC 
# MAGIC * It appears that the training dataset includes tweets in multiple languages (for example, beside English tweets we spotted a German tweet as well). This presents challenges with stemming, lemmatization, and removal of stop words, since we based these processes on the English language.
# MAGIC * Class imbalance in the dataset may be an issue. However, we accounted for that case already to some extent with the choice of an appropriate evaluation metric. Furthermore, we could use a resampling technique (upsampling or downsampling) or use a technique like SMOTE (Synthetic Minority Oversampling technique) to create synthetic data points. 
# MAGIC * To deal with text of different length and representation of text as features, we will use a proper representation in the form of a TF-IDF matrix as well as using (pre-trained) word embeddings, which also consider the context. 
# MAGIC 
# MAGIC Lastly, we will develop a workflow chart with pre-processing pipeline using mlflow.
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Outline**
# MAGIC 
# MAGIC We divided this notebook into eight major parts: 
# MAGIC 1. Setup Environment
# MAGIC 2. Import Raw Training Data
# MAGIC 3. Exploratory Data Analysis
# MAGIC 4. Classical ML Models
# MAGIC 5. Deep-Learning Model
# MAGIC 6. Predictions on Twitter Data
# MAGIC 7. Correlation Model
# MAGIC 8. Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Environment

# COMMAND ----------

# MAGIC %md
# MAGIC #### Packages

# COMMAND ----------

# MAGIC %pip install spark-nlp==3.3.3 wordcloud contractions gensim pyldavis==3.2.0

# COMMAND ----------

# MAGIC %md
# MAGIC #### Database

# COMMAND ----------

import re
userName = spark.sql("SELECT CURRENT_USER").collect()[0]['current_user()']
userName0 = userName.split("@")[0]
userName0 = re.sub('[!#$%&\'*+-/=?^`{}|\.]+', '_', userName0)
userName1 = userName.split("@")[1]
userName = f'{userName0}@{userName1}'
dbutils.fs.mkdirs(f"/Users/{userName}/data")
userDir = f"/Users/{userName}/data"
dbfsUserDir = f"/dbfs{userDir}"
databaseName = "group5_finalproject"

print('databaseName ' + databaseName)
print('UserDir ' + userDir)
print('userName '+ userName)

spark.sql(f"CREATE DATABASE IF NOT EXISTS {databaseName}")
spark.sql(f"use {databaseName}")

# COMMAND ----------

from pyspark.sql.types import StructType,StructField, StringType, IntegerType, TimestampType, DateType

#target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
#ids: The id of the tweet ( 2087)
#date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
#flag: The query (lyx). If there is no query, then this value is NO_QUERY.
#user: the user that tweeted (robotickilldozr)
#text: the text of the tweet (Lyx is cool)

schema = StructType([StructField('target', IntegerType(), True),
                     StructField('ids', IntegerType(), True),
                     StructField('date', StringType(), True),
                     StructField('flag', StringType(), True),
                     StructField('user', StringType(), True),
                     StructField('text', StringType(), True)])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Helper Functions

# COMMAND ----------

def print_registered_model_info(rm):
    #print("name: {}".format(rm.name))
    #print("tags: {}".format(rm.tags))
    #print("description: {}".format(rm.description))
    print(rm)
    print('----------')

def print_model_version_info(mv):
    #print("Name: {}".format(mv.name))
    #print("Version: {}".format(mv.version))
    #print("Tags: {}".format(mv.tags))    
    print(mv)
    print('----------')

def print_models_info(mv):
    for m in mv:
        print(m)
    print('-----------')
        #print("name: {}".format(m.name))
        #print("latest version: {}".format(m.version))
        #print("run_id: {}".format(m.run_id))
        #print("current_stage: {}".format(m.current_stage))    
        
def delete_registered_model(modelname) : 
  mv = client.get_registered_model(model_name)
  mvv = mv.latest_versions
  for i in mvv:
    try:
      if i.current_stage != "Archived":
        client.transition_model_version_stage(model_name, i.version, "archived")
      client.delete_registered_model(model_name)
    except Exception as e:
      print(traceback.format_exc())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import Raw Training Data

# COMMAND ----------

# MAGIC %fs head dbfs:/mnt/data/final-assignment-training/training.1600000.processed.noemoticon.csv 

# COMMAND ----------

from pyspark.sql import functions as F

# Read raw csv file into SparkDataFrame
sdf_raw = spark.read.options(header=False,delimiter=',').schema(schema).csv("dbfs:/mnt/data/final-assignment-training/training.1600000.processed.noemoticon.csv").repartition(16).cache()
sdf_raw.display()

# COMMAND ----------

# Show some stats 
sdf_raw.describe().show()

# COMMAND ----------

# Count number differenct values in target feature >> no class imbalance
sdf_raw.groupBy('target').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### Text length (#words)

# COMMAND ----------

df_raw = sdf_raw.toPandas()
df_raw['length']= df_raw['text'].str.split().map(lambda x: len(x))

df_class0 = df_raw[df_raw['target']==0]
df_class4 = df_raw[df_raw['target']==4]

print(f"Overall average text length: {df_raw['length'].mean()}")
print(f"Average text length class 0 (negative): {df_class0['length'].mean()}")
print(f"Average text length class 4 (positive): {df_class4['length'].mean()}")
print(f"Median text length class 0 (negative): {df_class0['length'].median()}")
print(f"Median text length class 4 (positive): {df_class4['length'].median()}")
print(f"Max text length class 0 (negative): {df_class0['length'].max()}")
print(f"Max text length class 4 (positive): {df_class4['length'].max()}")
print(f"Min text length class 0 (negative): {df_class0['length'].min()}")
print(f"Min text length class 4 (positive): {df_class4['length'].min()}")

# COMMAND ----------

import seaborn as sns

ax = sns.violinplot(x="target", y="length", data=df_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:** positive tweets tend to feature more words than negative tweets. 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Wordcloud

# COMMAND ----------

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Word cloud for negative tweets
neg = sdf_raw.filter(sdf_raw.target == 0).toPandas()
text = " ".join(i for i in neg.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:** the most common words in negative tweets are "work," "want", and "now". It is a little bit surprising/interesting to note, that "work" is so highly present in negative tweets. 

# COMMAND ----------

# Word cloud for positive tweets
pos = sdf_raw.filter(sdf_raw.target == 4).toPandas()
text = " ".join(i for i in pos.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:** the most common words in positive tweets are "thank," "love," and "lol" (which is somewhat expectable).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Topic Modeling

# COMMAND ----------

# Load the regular expression library
import re

# Remove punctuation
df_raw['text_processed'] = \
df_raw['text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Remove Twitter handle
df_raw['text_processed'] = \
df_raw['text_processed'].map(lambda x: re.sub("@[^\s]+", '', x))
# Remove URLs
df_raw['text_processed'] = \
df_raw['text_processed'].map(lambda x: re.sub(r"www\S+", '', x))
df_raw['text_processed'] = \
df_raw['text_processed'].map(lambda x: re.sub(r"http\S+", '', x))
# Convert the titles to lowercase
df_raw['text_processed'] = \
df_raw['text_processed'].map(lambda x: x.lower())

# Print out the first rows of papers
df_raw['text_processed'].head()

# COMMAND ----------

import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = df_raw['text_processed'].values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

# COMMAND ----------

import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])

# COMMAND ----------

from pprint import pprint
# number of topics
num_topics = 10
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# COMMAND ----------

import pyLDAvis.gensim
import pickle 
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_prepared

# COMMAND ----------

# MAGIC %md
# MAGIC Each bubble in the above chart represents a topic. The larger the bubble, the higher percentage of the number of tweets in the corpus is about that topic.
# MAGIC Blue bars represent the overall frequency of each word in the corpus. If no topic is selected, the blue bars of the most frequently used words will be displayed.
# MAGIC Red bars give the estimated number of times a given term was generated by a given topic. As you can see from the image below, there are about 22,000 of the word ‘go’, and this term is used about 10,000 times within topic 7. The word with the longest red bar is the word that is used the most by the tweets belonging to that topic.

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:** A good topic model should have fairly big, non-overlapping bubbles scattered throughout the chart instead of being clustered in one quadrant. While there are very few overlaps between the top 10 topics in the data, the topics aren't evenly distributed in the 4 quadrants. There is only 1 topic in quadrants 1 and 4, while the rest fell into quadrants 2 and 3.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 4. Classical ML Models

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

import pandas as pd
import sparknlp
import mlflow
import mlflow.spark
import tempfile
import pickle
import time
import traceback

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from sklearn.metrics import classification_report, accuracy_score
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier, LogisticRegression, NaiveBayes, RandomForestClassifier, LinearSVC
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, IndexToString
from pyspark.mllib.tree import RandomForest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

spark = sparknlp.start() 
print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pre-processing

# COMMAND ----------

# Replace 4 with 1 >> negative tweets will have 0 as target value and positive tweets 1 
sdf_raw_prep = sdf_raw
sdf_raw_prep = sdf_raw_prep.na.replace(4,1,'target')

# Renamce "target" column to "label"
sdf_raw_prep = sdf_raw_prep.withColumnRenamed("target", "label")

# Train-test split
sdf_train, sdf_test = sdf_raw_prep.randomSplit([0.8, 0.2], seed=42)

print(f"Shape of training set: {sdf_train.count()} rows, {len(sdf_train.columns)} columns")
print(f"Shape of test set: {sdf_test.count()} rows, {len(sdf_test.columns)} columns")

# COMMAND ----------

sdf_train.display()

# COMMAND ----------

sdf_train.groupBy('label').count().show()

# COMMAND ----------

# Create pre-processing stages

# Stage 1: DocumentAssembler as entry point
documentAssembler = DocumentAssembler() \
                    .setInputCol("text") \
                    .setOutputCol("document")

# Stage 2: Tokenizer
tokenizer = Tokenizer() \
              .setInputCols(["document"]) \
              .setOutputCol("token")

# Stage 3: Normalizer to lower text and to remove html tags, hyperlinks, twitter handles, 
# alphanumeric characters (integers, floats), timestamps in the format hh:mm (e.g. 10:30) and punctuation
cleanUpPatterns = ["<[^>]*>", r"www\S+", r"http\S+", "@[^\s]+", "[\d-]", "\d*\.\d+", "\d*\:\d+", "[^\w\d\s]"]
normalizer = Normalizer() \
                .setInputCols("token") \
                .setOutputCol("normalized") \
                .setCleanupPatterns(cleanUpPatterns) \
                .setLowercase(True)

# Stage 4: Remove stopwords
stopwords = StopWordsCleaner()\
              .setInputCols("normalized")\
              .setOutputCol("cleanTokens")\
              .setCaseSensitive(False)

# Stage 5: Lemmatizer
lemma = LemmatizerModel.pretrained() \
              .setInputCols(["cleanTokens"]) \
              .setOutputCol("lemma")

# Stage 6: Stemmer stems tokens to bring it to root form
#.setInputCols(["cleanTokens"]).setOutputCol("stem") \
stemmer = Stemmer() \
            .setInputCols(["lemma"]) \
            .setInputCols(["cleanTokens"]) \
            .setOutputCol("stem")

# Stage 7: Finisher to convert custom document structure to array of tokens
finisher = Finisher() \
            .setInputCols(["stem"]) \
            .setOutputCols(["token_features"]) \
            .setOutputAsArray(True) \
            .setCleanAnnotations(False)

# COMMAND ----------

# Check pre-processing pipeline
prep_pipeline = Pipeline(stages=[documentAssembler, tokenizer, normalizer, stopwords, lemma, stemmer, finisher])

empty_df = spark.createDataFrame([['']]).toDF("text")
prep_pipeline_model = prep_pipeline.fit(empty_df)

result = prep_pipeline_model.transform(sdf_train)
result.select('text', 'token_features').show(truncate=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Feature Vectorization

# COMMAND ----------

# Stage 8: Hashing TF to generate Term Frequency, set number features to 8192 (a power of 2)
hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=8192)

# Stage 9: IDF to generate Inverse Document Frequency with 4 as minimum of document in which a term should appear for filtering
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=4)

# COMMAND ----------

# Check pre-processing pipeline
prep_pipeline = Pipeline(stages=[documentAssembler, tokenizer, normalizer, stopwords, lemma, stemmer, finisher, hashingTF, idf])

empty_df = spark.createDataFrame([['']]).toDF("text")
prep_pipeline_model = prep_pipeline.fit(empty_df)

result = prep_pipeline_model.transform(sdf_train)
result.select('token_features', 'features').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Classic ML Model

# COMMAND ----------

# Stage 10: classifiers  
clf_nb = NaiveBayes()
clf_lr = LogisticRegression()
clf_rf = RandomForestClassifier(seed=42)
clf_svc = LinearSVC()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline in MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Easy train and evaluate - Just as trial

# COMMAND ----------

classicML_nlp_pipeline = Pipeline(stages=[documentAssembler, tokenizer, normalizer, stopwords, lemma, stemmer, finisher, hashingTF, idf, clf_lr])

classicML_nlp_pipelineModel = classicML_nlp_pipeline.fit(sdf_train)
predictions = classicML_nlp_pipelineModel.transform(sdf_test)

evaluator = BinaryClassificationEvaluator()
roc_auc = evaluator.evaluate(predictions)
print("ROC AUC = %g" % (roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperopt tuning

# COMMAND ----------

from hyperopt import hp, Trials, fmin, tpe
from hyperopt.pyll import scope 
  
#space = dict()    
#space['n_estimator'] =  scope.int(hp.quniform('n_estimator',100,2000,100)) 
#space['max_depth'] = scope.int(hp.quniform('max_depth',3,20,1))
#space['learning_rate'] = hp.loguniform('learning_rate', -5, 0)

def objective_function_classicalML(params):

    # Set the hyperparameters that we want to tune:
    max_iter = int(params["maxIter"])
    
     # Define classifier
    log_reg = LogisticRegression(maxIter=max_iter)
    
    # Set pipeline 
    classicML_nlp_pipeline = Pipeline(stages=[documentAssembler, tokenizer, normalizer, stopwords, lemma, stemmer, finisher, hashingTF, idf, clf_lr])
    
    # Make predictions
    classicML_nlp_pipelineModel = classicML_nlp_pipeline.fit(sdf_train)
    predictions = classicML_nlp_pipelineModel.transform(sdf_train)
    evaluator = BinaryClassificationEvaluator()
    areaUnderROC = evaluator.evaluate(predictions)
    #loss = hp.pchoice('loss',
                     # [(0.50, 'hinge'),
                      # (0.25, 'log'),
                      # (0.25, 'huber')])
     
    return {"loss": -areaUnderROC, "status": STATUS_OK}
  

search_space = {
  "maxIter": hp.quniform("max_Iter", 10, 100, 1)
}

# COMMAND ----------

# Hyperopt trial
from hyperopt import fmin, tpe, STATUS_OK, Trials

# Start a parent MLflow run
with mlflow.start_run():
    # The number of models we want to evaluate
    num_evals = 20

    # Set the number of models to be trained concurrently
    trials = Trials()

    # Run the optimization process
    best_hyperparam = fmin(
        fn=objective_function_classicalML, 
        space=search_space,
        algo=tpe.suggest, 
        trials=trials,
        max_evals=num_evals)

    # Get optimal hyperparameter values
    best_max_iter = int(best_hyperparam["maxIter"])
    
    # Train model on entire training data
    log_reg = LogisticRegression()
    classicML_nlp_pipeline = Pipeline(stages=[documentAssembler, tokenizer, normalizer, stopwords, lemma, stemmer, finisher, hashingTF, idf, clf_lr])
    classicML_nlp_pipelineModel = classicML_nlp_pipeline.fit(sdf_train)

    # Evaluator on test set
    predictions = classicML_nlp_pipelineModel.transform(sdf_test)
    evaluator = BinaryClassificationEvaluator()
    areaUnderROC = evaluator.evaluate(predictions)

    mlflow.log_param("maxIter", best_max_iter)
    mlflow.log_metric("areaUnderROC", areaUnderROC)

# COMMAND ----------

best_hyperparam

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:** We tried tuning with Hyperopt but it was very time consuming so we decided to use ParamgridBuilder instead.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ParamgridBuilder tuning

# COMMAND ----------

# Due to the reported issue at https://github.com/JohnSnowLabs/spark-nlp/issues/1158, we split up the entire pipeline into two parts
classicML_nlp_pipeline_pt1 = Pipeline(stages=[documentAssembler, tokenizer, normalizer, stopwords, lemma, stemmer, finisher])
empty_df = spark.createDataFrame([['']]).toDF("text")
classicML_nlp_pipeline_model_pt1 = classicML_nlp_pipeline_pt1.fit(empty_df)

result = classicML_nlp_pipeline_model_pt1.transform(sdf_train)
result.select('label', 'token_features').show(truncate=False)

# COMMAND ----------

# Write results of finisher as parquet file
result.select('label', 'token_features').write.parquet(f'{userDir}/pipeline_pt1.parquet', mode="overwrite")

# COMMAND ----------

!cd '{dbfsUserDir}' && ls

# COMMAND ----------

# Read results of finisher
result_pipeline_pt1 = spark.read.parquet(f'{userDir}/pipeline_pt1.parquet')

# COMMAND ----------

result_pipeline_pt1.show(truncate=False)

# COMMAND ----------

with mlflow.start_run() as run:
  # Here, we use the second part of our original pipeline consisting of ML stage and apply them to the outcome of the first pipeline
  classicML_nlp_pipeline_pt2_nb = Pipeline(stages=[hashingTF, idf, clf_nb])
  classicML_nlp_pipeline_pt2_lr = Pipeline(stages=[hashingTF, idf, clf_lr])
  classicML_nlp_pipeline_pt2_rf = Pipeline(stages=[hashingTF, idf, clf_rf])
  classicML_nlp_pipeline_pt2_svc = Pipeline(stages=[hashingTF, idf, clf_svc])
  
  # Create grid for hyperparameter tuning for different classifier stages
  grid_nb = ParamGridBuilder().addGrid(clf_nb.smoothing, [0.5, 1]).build()
    
  grid_lr = ParamGridBuilder().addGrid(clf_lr.maxIter, [10, 100, 1000]) \
                              .addGrid(clf_lr.regParam, [0.1, 0.01]) \
                              .build()
  
  grid_rf = ParamGridBuilder().addGrid(clf_rf.maxDepth, [8, 16, 24]) \
                              .addGrid(clf_rf.numTrees, [16, 32, 64]) \
                              .build()
  
  grid_svc = ParamGridBuilder().addGrid(clf_svc.maxIter, [10, 100, 1000]).build() 
                                  
  # Evaluator: we can get the f1 score, accuracy, precision and recall using MulticlassClassificationEvaluator which can be used for binary classification as well
  ev = MulticlassClassificationEvaluator (metricName="accuracy")
  
  # n-fold cross validation; in this case only 3-fold to save time
  cv_nb = CrossValidator(estimator=classicML_nlp_pipeline_pt2_nb, estimatorParamMaps=grid_nb, evaluator=ev, numFolds=3)
  cv_lr = CrossValidator(estimator=classicML_nlp_pipeline_pt2_lr, estimatorParamMaps=grid_lr, evaluator=ev, numFolds=3)
  cv_rf = CrossValidator(estimator=classicML_nlp_pipeline_pt2_rf, estimatorParamMaps=grid_rf, evaluator=ev, numFolds=3)
  cv_svc = CrossValidator(estimator=classicML_nlp_pipeline_pt2_svc, estimatorParamMaps=grid_svc, evaluator=ev, numFolds=3)

  # Fit cross-validated model and save best model for NaiveBayes
  cvModel_nb = cv_nb.fit(result_pipeline_pt1)
  tuned_model_nb = cvModel_nb.bestModel
  mlflow.spark.log_model(tuned_model_nb, "classicML_model_nb", registered_model_name="NaiveBayes")
  
  # Fit cross-validated model and save best model for LogiticRegression
  cvModel_lr = cv_lr.fit(result_pipeline_pt1)
  tuned_model_lr = cvModel_lr.bestModel
  mlflow.spark.log_model(tuned_model_lr, "classicML_model_lr", registered_model_name="LogisticRegression")
  
  # Fit cross-validated model and save best model for RandomForest
  cvModel_rf = cv_rf.fit(result_pipeline_pt1)
  tuned_model_rf = cvModel_rf.bestModel
  mlflow.spark.log_model(tuned_model_rf, "classicML_model_rf", registered_model_name="RandomForest")
  
  # Fit cross-validated model and save best model for Support Vector Machine
  cvModel_svc = cv_svc.fit(result_pipeline_pt1)
  tuned_model_svc = cvModel_svc.bestModel
  mlflow.spark.log_model(tuned_model_svc, "classicML_model_svc", registered_model_name="SupportVectorMachine")
  
  # The parent run
  runinfo = run.info

# COMMAND ----------

print(f"runinfo.run_id: {runinfo.run_id}")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Prediction & Model Selection

# COMMAND ----------

# Get best model of each classifier from MLflow
loaded_model_nb = mlflow.spark.load_model(model_uri=f'runs:/{runinfo.run_id}/classicML_model_nb')
loaded_model_lr = mlflow.spark.load_model(model_uri=f'runs:/{runinfo.run_id}/classicML_model_lr')
loaded_model_rf = mlflow.spark.load_model(model_uri=f'runs:/{runinfo.run_id}/classicML_model_rf')
loaded_model_svc = mlflow.spark.load_model(model_uri=f'runs:/{runinfo.run_id}/classicML_model_svc')

# COMMAND ----------

# Transform test set first
test_trans = classicML_nlp_pipeline_model_pt1.transform(sdf_test)
test_trans.select('label', 'token_features').show(truncate=False)

# COMMAND ----------

# Make prredictions on test set
ev = MulticlassClassificationEvaluator(metricName="accuracy")

preds = loaded_model_nb.transform(test_trans)
print(f"Accuracy for NaiveBayes: {ev.evaluate(preds)}")

preds = loaded_model_lr.transform(test_trans)
print(f"Accuracy for LogisticRegression: {ev.evaluate(preds)}")

preds = loaded_model_rf.transform(test_trans)
print(f"Accuracy for RandomForest: {ev.evaluate(preds)}")

preds = loaded_model_svc.transform(test_trans)
print(f"Accuracy for SupportVectorMachine: {ev.evaluate(preds)}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register Model & Promote to Production

# COMMAND ----------

# Assign model name, i.e. best model coming out of pipeline
model_name = "best_model_classicalML"

# Assign MLflow client
client = mlflow.tracking.MlflowClient()

try:
  delete_registered_model(model_name)
except Exception as e:
  #print('---------')
  #print(traceback.format_exc())
  pass
tags = {'estimator_name': 'SVCLinear', 'estimator_params': "maxIter"}
description = "Best classical ML model for sentiment analysis using a Linear Support Vector Classifier" 

client.create_registered_model(model_name, tags, description)
registered_model = print_registered_model_info(client.get_registered_model(model_name))

# Add version 1 of the model
runs_uri = "runs:/" + runinfo.run_uuid + "/classicML_model_svc"
model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)

client.create_model_version(model_name, model_src, runinfo.run_id, tags=tags, description=description)

regmodel = client.get_model_version(model_name, 1)
print_model_version_info(regmodel)

# COMMAND ----------

time.sleep(2) # Just to make sure it's had a second to register
mv = client.transition_model_version_stage(name=model_name, version=regmodel.version, stage="Production", archive_existing_versions=True)
print_model_version_info(mv)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Access Production Model

# COMMAND ----------

latest_prod_model_detail = client.get_latest_versions(model_name, stages=['Production'])[0]
print(latest_prod_model_detail.run_id)

latest_prod_model =  mlflow.spark.load_model(f"runs:/{latest_prod_model_detail.run_id}/sentimentDL_model")
latest_prod_model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Potential Enhancements

# COMMAND ----------

# MAGIC %md
# MAGIC To improve the performance of classical ML algorithms, other approaches like XGBoost or GradientBoostedTrees could be tested. Besides, the individual algorithms could be combined to an ensemble which is then used for prediction (e.g. majority voting, stacking). Though, we feel that such an analysis is out of scope for this project.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deep-Learning Model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

import pandas as pd
import sparknlp
import mlflow
import mlflow.spark
import tempfile
import pickle
import time
import traceback

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from sklearn.metrics import classification_report, accuracy_score
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

spark = sparknlp.start() 
# For training on GPU
sparknlp.start(gpu=True) 

print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pre-processing

# COMMAND ----------

# Cast target to string and replace 4 with "positve" and 0 with "negative" (can be processed by the SentimentDL model)
sdf_raw_prep = sdf_raw
sdf_raw_prep = sdf_raw_prep.withColumn("target",sdf_raw_prep["target"].cast(StringType()))
sdf_raw_prep = sdf_raw_prep.na.replace("4","positive",'target')
sdf_raw_prep = sdf_raw_prep.na.replace("0","negative",'target')

# Renamce "target" column to "label"
sdf_raw_prep = sdf_raw_prep.withColumnRenamed("target", "label")

# Train-test split
sdf_train, sdf_test = sdf_raw_prep.randomSplit([0.8, 0.2], seed=42)

print(f"Shape of training set: {sdf_train.count()} rows, {len(sdf_train.columns)} columns")
print(f"Shape of test set: {sdf_test.count()} rows, {len(sdf_test.columns)} columns")

# COMMAND ----------

# Take a look at the training dataset
sdf_train.display()

# COMMAND ----------

# Create pre-processing stages

# Stage 1: DocumentAssembler as entry point
documentAssembler = DocumentAssembler() \
                    .setInputCol("text") \
                    .setOutputCol("document")

# Stage 2: Normalizer to lower text and to remove html tags, hyperlinks, twitter handles, alphanumeric characters (integers, floats) and timestamps in the format hh:mm (e.g. 10:30) 
cleanUpPatterns = ["<[^>]*>", r"www\S+", r"http\S+", "@[^\s]+", "[\d-]", "\d*\.\d+", "\d*\:\d+"]
documentNormalizer = DocumentNormalizer() \
                      .setInputCols("document") \
                      .setOutputCol("normalizedDocument") \
                      .setAction("clean") \
                      .setPatterns(cleanUpPatterns) \
                      .setReplacement("") \
                      .setPolicy("pretty_all") \
                      .setLowercase(True)

# COMMAND ----------

# Check pre-processing pipeline
prep_pipeline = Pipeline(stages=[documentAssembler, documentNormalizer])
empty_df = spark.createDataFrame([['']]).toDF("text")
prep_pipeline_model = prep_pipeline.fit(empty_df)

result = prep_pipeline_model.transform(sdf_train)
result.select('text', 'normalizedDocument.result').show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Word Embeddings

# COMMAND ----------

# Stage 3: apply the universal sentence encoding for word embeddings 
use = UniversalSentenceEncoder \
        .pretrained() \
        .setInputCols(["normalizedDocument"]) \
        .setOutputCol("sentence_embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deep-Learning Model

# COMMAND ----------

# MAGIC %md
# MAGIC **Notes:** 
# MAGIC * we are using a pre-trained deep neural network (see: https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.SentimentDLModel.html) trained on the well known IMDB dataset. SparkNLP actually offers more pre-trained models which could be used via NLP Models Hub (https://nlp.johnsnowlabs.com/models?task=Sentiment+Analysis).
# MAGIC 
# MAGIC * we used neither ParamGridBuilder() nor hyperopt in this case to tune the relevant hyperparameters (#epochs, learning ratem dropout rate and batch size) for various reasons. First of all, we did not have direct access to the evaluation metrics during training (only via a log file). More importantly, based on the log file it is not clear if loss/accuracy values are those for validation or training set (or if cross-validation is even used). Based on our investigations, we assume that the loss/accuracy values represent those for the training set. Thus, tuning the hyperparameters based on those numbers would not make sense and likely result in models which overfit a lot. For instance, to determine the best number of epochs we would need to get the point where the loss value of the validation - rather than the loss for the training set -  starts to increase/stays flat for a certain number of epochs. In fact, we tested model configurations which achieved accuracy values around 90% in the log file but performance on the test file was poor due to overfitting. However, since the model we use was already pre-trained we expected the performance to be good with the basic parameter configuration. In fact, we manually tested different values for number of epochs and dropout rate in several runs, but finally found that the default configuration seems to be the best choice for our use-case.

# COMMAND ----------

# Stage 4: use pre-trained deep-learning model for sentiment detection
# 
# Hyperparameters to be optimized: 
# * lr : Learning Rate, by default 0.005
# * batchSize : Batch size, by default 64
# * dropout : Dropout coefficient, by default 0.5
# * maxEpochs : Maximum number of epochs to train, by default 30
sentimentdl = SentimentDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("label")\
  .setMaxEpochs(5)\
  .setBatchSize(64) \
  .setLr(5e-3) \
  .setDropout(0.5) \
  .setEnableOutputLogs(True) \
  .setOutputLogsPath(userDir)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline in MLflow

# COMMAND ----------

with mlflow.start_run() as run:
  # Create pipeline out of above stages 1-4
  use_clf_pipeline = Pipeline(stages = [documentAssembler, documentNormalizer, use, sentimentdl])
  
  # grid = (ParamGridBuilder().addGrid(use_clf_pipeline.getStages()[-1].maxEpochs, [5, 25, 50]) \
  #                          .addGrid(use_clf_pipeline.getStages()[-1].batchSize, [32, 64]) \
  #                          .addGrid(use_clf_pipeline.getStages()[-1].dropout, [0.3, 0.5]) \
  #                          .addGrid(use_clf_pipeline.getStages()[-1].lr, [5e-3, 1e-2]) \
  #                          .build())
  
  # Log parameter
  mlflow.log_param("max_epochs", 5)
  mlflow.log_param("batch_size", 64)
  mlflow.log_param("learning_rate", 0.005)
  mlflow.log_param("dropout", 0.5)
  
  # Fit pipeline
  use_pipelineModel = use_clf_pipeline.fit(sdf_train)
  
  # Save model
  mlflow.spark.log_model(use_pipelineModel, "sentimentDL_model", registered_model_name="SentimentDL")
  
  # The parent run
  runinfo = run.info

# COMMAND ----------

print(f"{runinfo.run_id}")

# COMMAND ----------

# Looking up log files of the model
!cd '{dbfsUserDir}' && ls -lt

# COMMAND ----------

!cat '{dbfsUserDir}/SentimentDLApproach_9ceff30ea694.log'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Selection 

# COMMAND ----------

# Get best model from MLflow
loaded_model = mlflow.spark.load_model(model_uri=f'runs:/{runinfo.run_id}/sentimentDL_model')
loaded_model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** the DL model we use provides only strings or discrete values as output, but not probabilities or raw predictions (at least we could not find a way how to retrieve them).

# COMMAND ----------

# Predictions
preds = loaded_model.transform(sdf_test)
preds.select("label", "text", "class.result").show(10, truncate=100)

# COMMAND ----------

preds.groupBy('class.result').count().show()

# COMMAND ----------

# Transform to pandas dataframe
df_preds = preds.select("label", "text", "class.result").toPandas()

# Select only results with positive and negative tweets as we did not have neutral tweets in raw data. Besides, do not consider empty tweets.
df_preds = df_preds[(df_preds['result'] == 'positive') | (df_preds['result'] == 'negative')]

# The result is an array since in Spark NLP you can have multiple sentences.
# This means you can add SentenceDetector in the pipeline and feed it into
# UniversalSentenceEncoder and you can have prediction based on each sentence.
# Let's explode the array and get the item(s) inside of result column out
df_preds['result'] = df_preds['result'].apply(lambda x: x[0])

# Show classification report
print(classification_report(df_preds['result'], df_preds['label']))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Register Model & Promote to Production

# COMMAND ----------

# Assign model name
model_name = f"best_model_sentimentDL"

# Assign MLflow client
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

try:
  delete_registered_model(model_name)
except Exception as e:
  #print('---------')
  #print(traceback.format_exc())
  pass
tags = {'estimator_name': 'SentimentDL', 'estimator_params': "dropout=0.25, batchSize=64, maxEpochs=50, learning_rate=0.001"}
description = "Sentiment model based on pre-trained word embeddings using SparkNLP" 

client.create_registered_model(model_name, tags, description)
registered_model = print_registered_model_info(client.get_registered_model(model_name))

# COMMAND ----------

# Add version 1 of the model
runs_uri = "runs:/" + runinfo.run_uuid + "/sentimentDL_model"
model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)

client.create_model_version(model_name, model_src, runinfo.run_id, tags=tags, description=description)

# COMMAND ----------

regmodel = client.get_model_version(model_name, 1)
print_model_version_info(regmodel)

# COMMAND ----------

time.sleep(2) # Just to make sure it's had a second to register
mv = client.transition_model_version_stage(name=model_name, version=regmodel.version, stage="Production", archive_existing_versions=True)
print_model_version_info(mv)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Access Production Model

# COMMAND ----------

latest_prod_model_detail = client.get_latest_versions(model_name, stages=['Production'])[0]
print(latest_prod_model_detail.run_id)

latest_prod_model =  mlflow.spark.load_model(f"runs:/{latest_prod_model_detail.run_id}/sentimentDL_model")
latest_prod_model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Potential Enhancements

# COMMAND ----------

# MAGIC %md
# MAGIC To improve the performance of a DL model for sentiment analysis, other pre-trained models for word embeddings and/or prediction (e.g. https://nlp.johnsnowlabs.com/models?task=Sentiment+Analysis) should be investigated and tested. Besides, other NLP libraries than SparkNLP could be investigated as well. Though, we feel that it is out of scope for this project to do such an analysis.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Predictions on Twitter Data

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** since the deep learning model produced higher accuracy on the test set, we decided to use it for the twitter sentiment prediction

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * from group5_finalproject.crypto_df_tf;

# COMMAND ----------

import pandas as pd
import sparknlp
import mlflow
import mlflow.spark
import tempfile
import pickle
import time
import traceback
import pyspark
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

from pyspark.sql.functions import *
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

# Get data in twitter table
sdf_twitter = spark.sql("SELECT * FROM group5_finalproject.crypto_df_tf")

# Select model for prediction
model_name = "best_model_sentimentDL"

# Assign MLflow client and get model for prediction from production stage
client = mlflow.tracking.MlflowClient()
latest_prod_model_detail = client.get_latest_versions(model_name, stages=["Production"])[0]
latest_prod_model =  mlflow.spark.load_model(f"runs:/{latest_prod_model_detail.run_id}/sentimentDL_model")

# Make prediction
sdf_preds = latest_prod_model.transform(sdf_twitter)

# Select columns in prediction dataframe and unnest result column
sdf_preds_trans = sdf_preds.select("id", "text", "QUERY", "yfinance_ticker", "user_name", "result_type", \
                                   "favorite_count", "followers_count", "retweet_count", "created_at","day", \
                                   "hour","minute", F.col("class.result").getItem(0))

# Rename prediction column
sdf_preds_trans = sdf_preds_trans.withColumnRenamed(sdf_preds_trans.columns[-1], "sentiment")

# Write dataframe to a table
sdf_preds_trans.write.format('delta').mode("overwrite").saveAsTable("group5_finalproject.twitter_silver")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM group5_finalproject.twitter_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Correlation Model

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Setup

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Import

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** as we did not have the groundtruth with respect to sentiment for a certain time window in the financial dataset, we used the predicted sentiment instead to build a correlation model 

# COMMAND ----------

# Import data from gold table
sdf_corr = spark.table('group5_finalproject.gold')

# Convert the sentiment from string to integer for correlation model
sdf_corr = sdf_corr \
            .withColumn('sentiment_encoded', regexp_replace('sentiment', 'positive', '1')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded', 'negative', '-1')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded','neutral', '0')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded','null', '0'))
sdf_corr.display()

# Transform spark dataframe to pandas dataframe
df_corr = sdf_corr.toPandas()
df_corr = df_corr.dropna(axis = 0, how ='any')
df_corr['sentiment_encoded'] = df_corr['sentiment_encoded'].astype(int)
df_corr["QUERY"] = df_corr["QUERY"].str.lower()

# COMMAND ----------

# Group data by cryptocurrency and date, and add up the total sentiment score for each crptopcurreny in a 15-minute increments
df_corr_grouped = df_corr.groupby(['QUERY', 'date']).sum('sentiment_endcoded')
x = df_corr_grouped [['sentiment_encoded']]
y = df_corr_grouped ['delta'] # % change in closing price
df_corr_grouped 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training

# COMMAND ----------

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

# COMMAND ----------

with mlflow.start_run() as run:
  # Build linear regression model
  corr_model_lr = LinearRegression()
  corr_model_lr.fit(x_train, y_train)

  # Model score 
  r2 = corr_model_lr.score(x_train, y_train)

  # Use mlflow.sklearn
  mlflow.sklearn.log_model(corr_model_lr, "corr_model", registered_model_name="LinearRegression")
  mlflow.log_param("coeff", corr_model_lr.coef_)
  mlflow.log_metric("r2", r2)
    
  # The parent run
  runinfo = run.info

# COMMAND ----------

print(f"{runinfo.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prediction

# COMMAND ----------

# Get correlation model of each classifier from MLflow
loaded_model_corr = mlflow.sklearn.load_model(model_uri=f'runs:/{runinfo.run_id}/corr_model')
loaded_model_corr

# COMMAND ----------

# Making predictions and plot regression line
preds = loaded_model_corr.predict(x_test)
plt.scatter(y_test, preds)
plt.plot(x_test, preds, color='k')

# COMMAND ----------

preds

# COMMAND ----------

plt.hist(y_test - preds)

# COMMAND ----------

# Evaluate model
mae = metrics.mean_absolute_error(y_test, preds)
mse = metrics.mean_squared_error(y_test, preds)
rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))
print("Mean absolute error (test set): ", mae)
print("Mean squared error (test set): ", mse)
print("Root mean squared error (test set): ", rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:** There does not appear to be a linear relationship between cryptocurrencies' sentiment and change in price (within a 15 minute window). We observed in the gold table that several crpyocurrencies' prices decreased despite having a positive sentiment score overall. Possible future improvements include building an individual correlation model for each crpytocurrency. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register Model & Promote to Production

# COMMAND ----------

# Assign model name, i.e. best model coming out of pipeline
model_name = "best_model_corr"

# Assign MLflow client
client = mlflow.tracking.MlflowClient()

try:
  delete_registered_model(model_name)
except Exception as e:
  #print('---------')
  #print(traceback.format_exc())
  pass
tags = {'estimator_name': 'LinearRegression'}
description = "Best correlation model used to predict change in price for cryptocurrencies based on sentiment" 

client.create_registered_model(model_name, tags, description)
registered_model = print_registered_model_info(client.get_registered_model(model_name))

# Add version 1 of the model
runs_uri = "runs:/" + runinfo.run_uuid + "/corr_model"
model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)

client.create_model_version(model_name, model_src, runinfo.run_id, tags=tags, description=description)

regmodel = client.get_model_version(model_name, 1)
print_model_version_info(regmodel)

# COMMAND ----------

time.sleep(2) # Just to make sure it's had a second to register
mv = client.transition_model_version_stage(name=model_name, version=regmodel.version, stage="Production", archive_existing_versions=True)
print_model_version_info(mv)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Access Production Model

# COMMAND ----------

latest_prod_model_detail = client.get_latest_versions(model_name, stages=['Production'])[0]
print(latest_prod_model_detail.run_id)

latest_prod_model =  mlflow.sklearn.load_model(f"runs:/{latest_prod_model_detail.run_id}/corr_model")
latest_prod_model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make Inference

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pyspark.sql.functions import *

# Import data from gold table and transform it further
sdf_corr = spark.table('group5_finalproject.gold')
sdf_corr = sdf_corr \
            .withColumn('sentiment_encoded', regexp_replace('sentiment', 'positive', '1')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded', 'negative', '-1')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded','neutral', '0')) \
            .withColumn('sentiment_encoded', regexp_replace('sentiment_encoded','null', '0'))

df_corr = sdf_corr.toPandas()
df_corr = df_corr.dropna(axis = 0, how ='any')
df_corr['sentiment_encoded'] = df_corr['sentiment_encoded'].astype(int)
df_corr["QUERY"] = df_corr["QUERY"].str.lower()

# Group data by cryptocurrency and date, and add up the total sentiment score for each crptopcurreny in a 15-minute increments
df_corr_grouped = df_corr.groupby(['QUERY', 'date']).sum('sentiment_endcoded')
x = df_corr_grouped [['sentiment_encoded']]
y = df_corr_grouped ['delta'] # % change in closing price

# Select model for prediction
model_name = "best_model_corr"

# Assign MLflow client and get model for prediction from production stage
client = mlflow.tracking.MlflowClient()
latest_prod_model_detail = client.get_latest_versions(model_name, stages=["Production"])[0]
latest_prod_model =  mlflow.sklearn.load_model(f"runs:/{latest_prod_model_detail.run_id}/corr_model")

# Make prediction
preds = latest_prod_model.predict(x)

# Transform predictions in numpy to pandas DataFrame and then spark DataFrame, write spark DataFrame to a table
df_preds = pd.DataFrame(preds, columns=["preds"])
sdf_corr = spark.createDataFrame(df_preds, ["preds"])
sdf_corr.write.format('delta').mode("overwrite").saveAsTable("group5_finalproject.corr_test")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM corr_test

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC Overall, our data science pipeline is successful in addressing the problem defined in the beginning, as well as offering valuable insights and making predictions on sentiment. Our EDA process provides valuable insights and graphics on the text data, while our sentiment model produces accurate sentiment classification results. On the other hand, some possible future improvements include:
# MAGIC 1. models can be improved by using ensemble models (e.g. stacking) for classical ML, or using different pre-trained word embeddings/models (e.g. BERT) for deep learning.
# MAGIC 2. Building correlation model for each individual cryptocurrency using polarity
