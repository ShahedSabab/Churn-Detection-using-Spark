# Databricks notebook source
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from sklearn.utils import resample
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# COMMAND ----------

# DBTITLE 1,Read Data
spark = SparkSession.builder.appName('logReg').getOrCreate()
df = spark.read.csv('/FileStore/tables/customer_churn.csv', header=True, inferSchema=True)

# COMMAND ----------

# DBTITLE 1,Check Class Distribution
display(df)

# COMMAND ----------

# DBTITLE 1,Calculate churn and no churn data count
no_churn = df.filter(df['Churn']==0).count()
churn = df.filter(df['Churn']==1).count()
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

#performing one hot encoding
company_indexer = StringIndexer(inputCol='Company', outputCol='CompanyIndex')
indexer = company_indexer.fit(df_select_col)
company_vec = indexer.transform(df_select_col)

# COMMAND ----------

# initialize assembler
assembler = VectorAssembler(inputCols=['Age', 'Total_Purchase', 'Years', 'Num_Sites', 'CompanyIndex'], outputCol='features')
output = assembler.transform(company_vec)
# Check 
final_data = output.select('features', 'Churn')
display(final_data)

# COMMAND ----------

# DBTITLE 1,Train-Test Split
#train test split 
train_data, test_data = final_data.randomSplit([0.7,0.3], seed=10)


# COMMAND ----------

# DBTITLE 1,Model generation
#define model
model = LogisticRegression(labelCol='Churn')
fitted_model = model.fit(train_data)
training_sum = fitted_model.summary
training_sum.predictions.describe().show()

# COMMAND ----------

# DBTITLE 1,Model Evaluation 
results = fitted_model.evaluate(test_data)
results.predictions.show()

# COMMAND ----------

#evaluate
model_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Churn')
auc = model_eval.evaluate(results.predictions)
print("auc: ", auc)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC roc = training_sum.roc.toPandas()
# MAGIC plt.figure(figsize=(10,10))
# MAGIC plt.plot(roc['FPR'],roc['TPR'])
# MAGIC plt.ylabel('False Positive Rate')
# MAGIC plt.xlabel('True Positive Rate')
# MAGIC plt.title('ROC Curve')
# MAGIC plt.show()
# MAGIC print('Training set areaUnderROC: ' + str(training_sum.areaUnderROC))

# COMMAND ----------


