#!/usr/bin/env python
#try to build ALS model with pyspark.mllib.recommendation
import sys
import itertools
import numpy as np
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkContext
from pyspark.sql import HiveContext, Row
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

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

    def computeRmse(model, data, n):
        """
        Compute RMSE (Root Mean Squared Error).
        """
        predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
        predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
                                .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
                                .values()
        return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

    f = open('model_evaluation.txt', 'w')

    df = read_hdfs_csv(sqlContext, 'track_rate_bin.csv')


    def id_map(df, col):
        series = df.select(col).distinct().toPandas()[col]
        return series, dict(zip(series, range(len(series))))

    users, user_map = id_map(df, 'userID')
    items, item_map = id_map(df, 'itemID')

    def mapper(row):
        return Row(userID=user_map[row['userID']],
            itemID=item_map[row['itemID']],
            rating=row['rating_cat'])

    df_mapped = df.map(mapper).toDF().select('userID', 'itemID', 'rating')
    training, validation, test = df_mapped.randomSplit([0.6,0.2,0.2], seed = 12345)

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    f.write("Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest))

    ranks = [20,50,100]
    lambdas = [0.1,1,10]
    numIters = [10,20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0f
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        f.write("RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter))
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)

    # evaluate the best model on the test set
    f.write("The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse))

    f.close()
    # clean up
    sc.stop()

    


    
