package com.WordCount

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD

object NaiveBayesTest {
  def main(args: Array[String]){
     val conf = new SparkConf().setAppName("test");
    val sc=new SparkContext(conf);
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    
    var dataDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("D:\\bigdata\\spark\\testdata\\nb\\iris.csv").cache()
    
    val classToDouble = sqlContext.udf.register("classToDouble",(class_value:String) => {
       if (class_value.equalsIgnoreCase("Iris-setosa"))
        0.0
      else if(class_value.equalsIgnoreCase("Iris-versicolor"))
        1.0
      else
        2.0
    })
  
    dataDF = dataDF.withColumn("label", classToDouble(dataDF("class_value")))
  
    val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
    
    val nb = new NaiveBayes().setLabelCol("label").setFeaturesCol("features")
    
    val pipeline = new Pipeline().setStages(Array(assembler,nb))
    
    val splits = dataDF.randomSplit(Array(0.3,0.3), seed = 11L)
    val traindataDF = splits(0).cache()
    val testdataDF = splits(1).cache()
    
    val model = pipeline.fit(traindataDF);
    
    val result = model.transform(testdataDF)
    
    val lp = result.select( "label", "prediction")
    
  /* val counttotal = lp.count()
    
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    
    //RMSE Calculation
    val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("label").setPredictionCol("prediction")
    val rmse = evaluator.evaluate(result)
    println(s"Root-mean-square error = $rmse")
    * */
   // Since this is a multi-class classifier, we have to use MulticlassMetrics() to examine model accuracy.
    
    
    //multi class evaluation
    val redeictioAndLabelsTest = lp.map {
        row => (row.get(0).asInstanceOf[Double], row.get(1).asInstanceOf[Double])
    }
    //convert dataset to RDD
    val redeictioAndLabelsTestrdd : RDD[(Double, Double)] = redeictioAndLabelsTest.rdd
    val metrics = new MulticlassMetrics(redeictioAndLabelsTestrdd)
    val confusionMatrix = metrics.confusionMatrix 
   println("Confusion Matrix= n",confusionMatrix)
    println("Precision",metrics.precision,metrics.fMeasure,metrics.recall)

    
    //Now we will predit with some manual data
     val someDF = Seq( (5.1,3.5,1.4,0.2,0.0),  (7.0,3.2,4.7,1.4,1.0)).toDF("sepal_length","sepal_width","petal_length","petal_width","label")
     val resultTest = model.transform(someDF)
    
    val lpTest = resultTest.select( "label", "prediction")
    /*
+-----+----------+
|label|prediction|
+-----+----------+
|  0.0|       0.0|
|  1.0|       1.0|
+-----+----------+
    */
    //This is working perfectly
    val withoutLabelDF = Seq( (5.1,3.5,1.4,0.2),  (7.0,3.2,4.7,1.4)).toDF("sepal_length","sepal_width","petal_length","petal_width")
     val withoutLabelTest = model.transform(withoutLabelDF)
    
    val lpTest1 = withoutLabelTest.select( "prediction")
    lpTest1.show()
    
    //Now save the model for predicting real data
    model.write.overwrite().save("D:\\bigdata\\spark\\testdata\\nbsave\\")
  
  }
}