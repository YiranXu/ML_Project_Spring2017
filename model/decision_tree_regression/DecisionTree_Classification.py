#!/usr/bin/env python
import sys
import itertools
from pyspark import SparkContext
from pyspark.sql import HiveContext, Row
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
from sklearn.grid_search import ParameterGrid
from pyspark.mllib.evaluation import MulticlassMetrics


if __name__ == "__main__":

    conf = SparkConf().set("spark.executor.memory", "8g").set("spark.yarn.executor.memoryOverhead", "2048")
    sc = SparkContext(conf = conf)
    sqlContext = HiveContext(sc)

        # read file
    def read_hdfs_csv(sqlContext, filename, header='true'):
        csvreader = (sqlContext.read.format('com.databricks.spark.csv').options(header = header, inferschema='true'))
        return csvreader.load(filename)


    def write_hdfs_csv(df, filename):
        csvwriter = (df.write.format('com.databricks.spark.csv').options(header='true'))
        return csvwriter.save(filename)

    data = read_hdfs_csv(sqlContext, 'features_merge_unwrapped_sample.csv')
    #data = data.map(lambda df: Vectors.dense([float(c) for c in df]))

    def labelData(data):
        return data.map(lambda row: LabeledPoint(row[2], row[3:]))

    training, test = labelData(data).randomSplit([0.8, 0.2])
    numTraining = training.count()
    numTest = test.count()


    grid = [{'maxDepth': [5,7,10], 'maxBins': [16, 32], 'minInstancesPerNode': [1,5]}]
    paramGrid = list(ParameterGrid(grid))
    numModels = len(paramGrid)

    f = open('decisiontree_classification_evaluation', 'w')

    def getPredictionsLabels(model, test):
        predictions = model.predict(test.map(lambda r: r.features))
        return predictions.zip(test.map(lambda r: r.label))

    def printMetrics(predictions_and_labels):
        metrics = MulticlassMetrics(predictions_and_labels)
        f.write('Precision of True:{0}\n'.format(metrics.precision(1)))
        f.write('Precision of False:{0}\n'.format(metrics.precision(0)))
        f.write('Recall of True:{0}\n'.format(metrics.recall(1)))
        f.write('Recall of False:{0}\n'.format(metrics.recall(0))
        f.write('Confusion Matrix\n'.format(metrics.confusionMatrix().toArray()))

    for j in range(numModels):
        regp = paramGrid[j]['regParam']
        iters = paramGrid[j]['iterations']
        regt = paramGrid[j]['regType']
        con = paramGrid[j]['convergenceTol']

        f.write('Model{0}: maxDepth = {1}, maxBins = {2}, minInstancesPerNode = {3}\n'.format(str(j), depth, bins, minins))
        # Train decision tree classification model with hypermarameter set
        model = DecisionTree.trainClassifier(training, categoricalFeaturesInfo = {}, impurity='variance', \
            maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0)

        predictions_and_labels = getPredictionsLabels(model, test)
        printMetrics(predictions_and_labels)

    f.close()
    sc.stop()
