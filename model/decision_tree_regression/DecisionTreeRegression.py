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


    '''
    grid = [{'regParam': [0.01, 0.1], 'iterations': [10, 20], 'regType': ["l1", "l2"], 'convergenceTol': [1e-3, 1e-4]}]
    paramGrid = list(ParameterGrid(grid))
    numModels = len(paramGrid)

    f = open('model_evaluation', 'w')
    '''

    def getPredictionsLabels(model, test):
        predictions = model.predict(test.map(lambda r: r.features))
        return predictions.zip(test.map(lambda r: r.label))

    def printMetrics(predictions_and_labels):
        metrics = RegressionMetrics(predictions_and_labels)
        f.write('Explained Variance:{0}\n'.format(metrics.explainedVariance))
        f.write('Mean Absolute Error:{0}\n'.format(metrics.meanAbsoluteError))
        f.write('Mean Squared Error:{0}\n'.format(metrics.meanSquaredError))
        f.write('Root Mean Squared Error:{0}\n'.format(metrics.rootMeanSquaredError))
        f.write('R^2 :{0}\n'.format(metrics.r2))

    for j in range(numModels):
        regp = paramGrid[j]['regParam']
        iters = paramGrid[j]['iterations']
        regt = paramGrid[j]['regType']
        con = paramGrid[j]['convergenceTol']

        #f.write('Model{0}: regParam = {1}, iterations = {2}, regType = {3}, convergenceTol = {4}\n'.format(str(j), regp, iters, regt, con))
        # Train decision tree regression model with hypermarameter set
        model = DecisionTree.trainRegressor(training, categoricalFeaturesInfo = {}, impurity='variance', \
            maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0)

        predictions_and_labels = getPredictionsLabels(model, test)
        printMetrics(predictions_and_labels)

    f.close()
    sc.stop()



