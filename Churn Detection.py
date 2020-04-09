# Databricks notebook source
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from sklearn.utils import resample
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# DBTITLE 1,Read Data
#reading data 
spark = SparkSession.builder.appName('logReg').getOrCreate()
df = spark.read.csv('/FileStore/tables/customer_churn.csv', header=True, inferSchema=True)

# COMMAND ----------

# DBTITLE 1,Check Class Distribution
display(df)

# COMMAND ----------

# DBTITLE 1,Calculate churn and no churn data count
#check churn vs no churn data distribution
no_churn = df.filter(df['Churn']==0).count()
churn = df.filter(df['Churn']==1).count()
print("Churn :", churn)
print("No Churn :", no_churn)
diff = (no_churn - churn) /2
upSampleLength = churn + diff
downSampleLength = no_churn - diff
downSampleLength

# COMMAND ----------

df_no_churn = df.filter(df['Churn']==0)
df_no_churn.toPandas()

# COMMAND ----------

# DBTITLE 1,Downsampling and Upsampling
#spparating churn and no_churn and converting to pandas dataframe
df_no_churn = df.filter(df['Churn']==0).toPandas()
df_churn = df.filter(df['Churn']==1).toPandas()
RANDOM_SEED = 10

#applying upsampling and downsampling
no_churn_downsampled = resample(df_no_churn,
              replace=True,
              n_samples=int(downSampleLength),
              random_state=10)
churn_upsampled = resample(df_churn,
              replace=True,
              n_samples=int(upSampleLength),
              random_state=10)

#concatenating pandas dataframe: churn and no_churn
pd_df = [no_churn_downsampled, churn_upsampled]
pd_df = pd.concat(pd_df)

#random shuffling
pd_df = pd_df.sample(frac=1).reset_index(drop=True)

#convert pandas df to spark df
df = spark.createDataFrame(pd_df)


# COMMAND ----------

# DBTITLE 1,Check Class distribution after sampling
display(df)

# COMMAND ----------

display(df)

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Perform feature conversion
#selecting columns 
df_select_col = df.select(['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Company', 'Churn'])

#drop na
df_select_col = df_select_col.na.drop()
df_select_col.printSchema()

# COMMAND ----------

#vectorizing categorical features using string indexer and one hot encoder
categoricalColumns = ['Company']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
    #stages += [stringIndexer]

numericCols = ['Age', 'Total_Purchase', 'Years', 'Num_Sites']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

#create the assembler
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

#pipeline of transforming features
cols = df_select_col.columns
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df_select_col)
df = pipelineModel.transform(df_select_col)
selectedCols = ['features', 'Churn']
df = df.select(selectedCols)
df.printSchema()

# COMMAND ----------

#train-test split
train, test = df.randomSplit([0.7, 0.3], seed = 20)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

#Logistic regression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'Churn', maxIter= 50)
lrModel = lr.fit(train)
predictions = lrModel.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Churn')
print('Linear Regression Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

#Random Forest
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'Churn', numTrees=100)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Churn')
print('Random Forest Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

#Gradient Boost
gbt = GBTClassifier(featuresCol = 'features', labelCol = 'Churn',maxIter=100)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Churn')
print('Gradient Boost Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------


