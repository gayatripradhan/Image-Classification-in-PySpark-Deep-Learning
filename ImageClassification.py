# ./bin/pyspark --packages databricks:spark-deep-learning:0.1.0-spark2.1-s_2.11 --driver-memory 5g
from sparkdl import readImages
from pyspark.sql.functions import lit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

img_dir = "/home/gayatri/Documents/Spark/DeepLearning/personalities/"

#Read images and Create training & test DataFrames for transfer learning
person1_df = readImages(img_dir + "/person1").withColumn("label", lit(1))
person2_df = readImages(img_dir + "/person2").withColumn("label", lit(0))
person1_train, person1_test = person1_df.randomSplit([0.6, 0.4])
person2_train, person2_test = person2_df.randomSplit([0.6, 0.4])

#dataframe for training a classification model
train_df = person1_train.unionAll(person2_train)

#dataframe for testing the classification model
test_df = person1_test.unionAll(person2_test)


featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)

predictions = p_model.transform(test_df)

predictions.select("filePath", "prediction").show(truncate=False)

df = p_model.transform(test_df)
df.show()

predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
