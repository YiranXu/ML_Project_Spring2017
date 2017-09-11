import sys
import datetime
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

    conf = SparkConf().set("spark.executor.memory", "32g").set("spark.yarn.executor.memoryOverhead", "2048").set("spark.driver.memory","8g")
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

    training, test = labelData(data).randomSplit([0.8, 0.2])
    training.cache()
    test.cache()
    numTraining = training.count()
    numTest = test.count()


    grid = [{'maxDepth': [5,10,15,20,25], 'maxBins': [4,8,16,32]}]
    paramGrid = list(ParameterGrid(grid))
    numModels = len(paramGrid)

    f = open('decisiontree_regression_evaluation8_add.txt', 'w')
    

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
        depth = paramGrid[j]['maxDepth']
        bins = paramGrid[j]['maxBins']
   
        timestart = datetime.datetime.now()

        f.write('Model{0}: maxDepth = {1}, maxBins = {2}\n'.format(str(j), depth, bins))
        # Train decision tree regression model with hypermarameter set
        model = DecisionTree.trainRegressor(training, categoricalFeaturesInfo = {}, impurity='variance', \
            maxDepth=depth, maxBins=bins, minInfoGain=0.0)

        predictions_and_labels = getPredictionsLabels(model, test)
        printMetrics(predictions_and_labels)

        timeend = datetime.datetime.now()
        timedelta = round((timeend-timestart).total_seconds(), 2) 
        f.write("Time taken to execute this model is: " + str(timedelta) + " seconds.\n")


    f.close()
    sc.stop()

