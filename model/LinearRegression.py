#!/usr/bin/env python
import sys
import itertools
from pyspark import SparkContext
from pyspark.sql import HiveContext, Row
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics



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

    model = LinearRegressionWithSGD.train(training, iterations=100, \
        step=1.0, miniBatchFraction=1.0, initialWeights=None, regParam=0.0, \
        regType=None, intercept=False, validateData=True, convergenceTol=0.001)

    
    def printMetrics(predictions_and_labels):
        metrics = RegressionMetrics(predictions_and_labels)
        print 'Explained Variance     ', metrics.explainedVariance
        print 'Mean Absolute Error    ', metrics.meanAbsoluteError
        print 'Mean Squared Error     ', metrics.meanSquaredError
        print 'Root Mean Squared Error', metrics.rootMeanSquaredError
        print 'R^2                    ', metrics.r2

    predictions_and_labels = test.map(lambda lr: (float(model.predict(lr.features)), lr.label))
    printMetrics(predictions_and_labels)













