import sys
import datetime
import itertools
from pyspark import SparkContext
from pyspark.sql import HiveContext, Row
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.evaluation import RegressionMetrics
from sklearn.grid_search import ParameterGrid



if __name__ == "__main__":

    conf = SparkConf().set("spark.executor.memory", "32g").set("spark.yarn.executor.memoryOverhead", "2048")
    sc = SparkContext(conf = conf)
    sqlContext = HiveContext(sc)

        # read file
    def read_hdfs_csv(sqlContext, filename, header='true'):
        csvreader = (sqlContext.read.format('com.databricks.spark.csv').options(header = header, inferschema='true'))
        return csvreader.load(filename)


    def write_hdfs_csv(df, filename):
        csvwriter = (df.write.format('com.databricks.spark.csv').options(header='true'))
        return csvwriter.save(filename)

    data = read_hdfs_csv(sqlContext, sys.argv[1])
    #data = data.map(lambda df: Vectors.dense([float(c) for c in df]))

    def labelData(data):
        return data.map(lambda row: LabeledPoint(row[2], row[3:]))

    f = open('GradientBoostedTree_regression_evaluation7.txt', 'w')

    training, test = labelData(data).randomSplit([0.8, 0.2])
    numTraining = training.count()
    numTest = test.count()


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


    timestart = datetime.datetime.now()
    model = GradientBoostedTrees.trainRegressor(training, categoricalFeaturesInfo = {},\
 loss='leastSquaresError', numIterations=10, learningRate=0.1, maxDepth=15, maxBins=16)
    f.write(model.toDebugString())
    predictions_and_labels = getPredictionsLabels(model, test)
    printMetrics(predictions_and_labels)
    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds(), 2) 
    f.write("Time taken to execute this model is: " + str(timedelta) + " seconds.\n")


    f.close()
    sc.stop()
