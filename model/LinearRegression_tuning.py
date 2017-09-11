#!/usr/bin/env python
import sys
import datetime
import itertools
from pyspark import SparkContext
from pyspark.sql import HiveContext, Row
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
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

    def labelData(data):
        return data.map(lambda row: LabeledPoint(row[2], row[3:]))

    training, test = labelData(data).randomSplit([0.8, 0.2])
    training.cache()
    test.cache()
    numTraining = training.count()
    numTest = test.count()


    grid = [{'regParam': [0.001, 0.01, 0.1, 1, 10, 100], 'iterations': [100,200], 'regType': ["l1", "l2"], 'convergenceTol': [1e-4,1e-5]}]
    paramGrid = list(ParameterGrid(grid))
    numModels = len(paramGrid)

    f = open('linear_model_evaluation.txt', 'w')


    def printMetrics(model):
        predictions_and_labels = test.map(lambda lr: (float(model.predict(lr.features)), lr.label))
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

        timestart = datetime.datetime.now()

        f.write('Model{0}: regParam = {1}, iterations = {2}, regType = {3}, convergenceTol = {4}\n'.format(str(j), regp, iters, regt, con))
        # Train linear regression model with hypermarameter set
        model = LinearRegressionWithSGD.train(training, iterations=iters, \
            step=1.0, miniBatchFraction=1.0, initialWeights=None, regParam=regp, \
            regType=regt, intercept=False, validateData=True, convergenceTol=con)
        printMetrics(model)

        timesend = datetime.datetime.now()
        timedelta = round((timeend-timestart).total_seconds(), 2) 
        f.write("Time taken to execute this model is: " + str(timedelta) + " seconds.\n")

    f.close()
    sc.stop()











