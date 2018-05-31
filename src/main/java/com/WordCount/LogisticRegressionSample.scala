package com.WordCount

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

object LogisticRegressionSample {
  
  case class TitanicData(PassengerId:String, Survived:String, Pclass:String, 
      Name:String, Sex:String, Age:String, SibSp:String, Parch:String, 
      Ticket:String, Fare:String, Cabin:String, Embarked:String)

  
  def main(args: Array[String]){
    
   val conf = new SparkConf().setAppName("test");
    val sc=new SparkContext(conf);
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    
    /*val valRDD = sc.textFile("D:\\bigdata\\spark\\testdata\\titanic\\train.csv");
    val valMap = valRDD.map { line => line.split(",") }.map { col => TitanicData(col(1), col(2), col(3), col(4), col(5), col(6),col(7), col(8), col(9), col(10), col(11), col(11)) }
    var tDF = valMap.toDF();
        val skipable_first_row = tDF.first() 
    var trainDF    = tDF.filter(row => row != skipable_first_row) */
    
    //val toDoubleOld = udf[Double, String]( _.toDouble)
    // val toInt    = udf[Int, String]( _.toInt)
    
    

    
    var trainDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("D:\\bigdata\\spark\\testdata\\titanic\\train.csv").cache()
    
    //fill nan values
    var avgAge = trainDF.select(mean("Age")).first()(0).asInstanceOf[Double]
    trainDF=trainDF.na.fill(avgAge, Seq("Age"))
    
     val addChild = sqlContext.udf.register("addChild", (sex: String, age: Double) => {
      if (age < 15)
        "Child"
      else
        sex
    })
    trainDF = trainDF.withColumn("Sex", addChild(trainDF("Sex"), trainDF("Age")))
    
    val withFamilly = sqlContext.udf.register("withFamilly",(SibSp:Int, Parch:Int) => {
       if (SibSp + Parch > 3)
        1.0
      else
        0.0
    })
    trainDF = trainDF.withColumn("Familly", withFamilly(trainDF("SibSp"), trainDF("Parch")))
    
    val toDouble = sqlContext.udf.register("toDouble", ((n: Int) => { n.toDouble }))
    val findTitle = sqlContext.udf.register("findTitle", (name: String) => {
      val pattern = "(Dr|Mrs?|Ms|Miss|Master|Rev|Capt|Mlle|Col|Major|Sir|Lady|Mme|Don)\\.".r
      val matchedStr = pattern.findFirstIn(name)  
      var title = matchedStr match {
        case Some(s) => matchedStr.getOrElse("Other.")
        case None => "Other."
      }
      if (title.equals("Don.") || title.equals("Major.") || title.equals("Capt."))
        title = "Sir."
      if (title.equals("Mlle.") || title.equals("Mme."))
          title = "Miss."
      title 
    })
    
    

    trainDF = trainDF.withColumn("Title", findTitle(trainDF("Name")))
    trainDF = trainDF.withColumn("Pclass", toDouble(trainDF("Pclass")))
    trainDF = trainDF.withColumn("Survived", toDouble(trainDF("Survived")))
    
    val sexIdx = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex");
    val titleIdx = new StringIndexer().setInputCol("Title").setOutputCol("TitleIndex")
    
   
    //trainDF = trainDF.withColumn("fare",      toInt(trainDF("Fare")))
    //trainDF = trainDF.withColumn("age",      toInt(trainDF("Age")))
    //trainDF = trainDF.withColumn("pclass",      toInt(trainDF("Pclass")))
    
    val fareSplits = Array(0.0,10.0,20.0,30.0,40.0,Double.PositiveInfinity)
    val fareBucketize = new Bucketizer().setInputCol("Fare").setOutputCol("FareBucketed").setSplits(fareSplits)
    
    val vecrorArray = Array("SexIndex", "Age", "TitleIndex", "Pclass", "Familly","FareBucketed")
    val assembler = new VectorAssembler().setInputCols(vecrorArray).setOutputCol("features_temp")
    val normalize = new Normalizer().setInputCol("features_temp").setOutputCol("features").setP(1.0)
    
    val lr = new LogisticRegression().setMaxIter(10)
    lr.setLabelCol("Survived")
    
    val pipeline = new Pipeline().setStages(Array(sexIdx, titleIdx,fareBucketize, assembler, normalize,lr))
    
    val splits = trainDF.randomSplit(Array(0.8,0.2), seed = 11L)
    val traindataDF = splits(0).cache()
    val testdataDF = splits(1).cache()
    
    var model = pipeline.fit(traindataDF)
    var test = model.transform(testdataDF)
    
    var result = test.select("prediction","Survived")
    
    val redeictioAndLabels = result.map {
        row => (row.get(0).asInstanceOf[Double], row.get(1).asInstanceOf[Double])
    }
    //convert dataset to RDD
    val redeictioAndLabelsrdd : RDD[(Double, Double)] = redeictioAndLabels.rdd
    val matrices = new BinaryClassificationMetrics(redeictioAndLabelsrdd)
    println("Area under ROC = " + matrices.areaUnderROC())
    //The closer  the area Under ROC is to 1, the better the model is making predictions
    model = pipeline.fit(trainDF)
    
    
    
    //test data
    
    var testdf = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("D:\\bigdata\\spark\\testdata\\titanic\\test.csv").cache()
    //option("inferSchema", "true")  - This will take data types of each column in csv
    //If we don't put this option then all columns are String
    avgAge = testdf.select(mean("Age")).first()(0).asInstanceOf[Double]
    testdf = testdf.na.fill(avgAge, Seq("Age"))
    
    testdf = testdf.withColumn("Sex", addChild(testdf("Sex"), testdf("Age")))
    testdf = testdf.withColumn("Title", findTitle(testdf("Name")))
    testdf = testdf.withColumn("Pclass", toDouble(testdf("Pclass")))
    testdf = testdf.withColumn("Familly", withFamilly(testdf("SibSp"), testdf("Parch")))
    
    val getZero = sqlContext.udf.register("toDouble", ((n: Int) => { 0.0 }))
    testdf = testdf.withColumn("Survived", getZero(testdf("PassengerId")))
    
    result = model.transform(testdf)
    
    //calculate ROC and correct result
    val lp = result.select( "Survived", "prediction")
    val counttotal = lp.count()
    
    val correct = lp.filter($"Survived" === $"prediction").count()
    val wrong = lp.filter(not($"Survived" === $"prediction")).count()
    val truep = lp.filter($"prediction" === 0.0).filter($"Survived" === $"prediction").count()
    val falseN = lp.filter($"prediction" === 0.0).filter(not($"Survived" === $"prediction")).count()
    val falseP = lp.filter($"prediction" === 1.0).filter(not($"Survived" === $"prediction")).count()
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble
    
    val redeictioAndLabelsTest = lp.map {
        row => (row.get(0).asInstanceOf[Double], row.get(1).asInstanceOf[Double])
    }
    //convert dataset to RDD
    val redeictioAndLabelsTestrdd : RDD[(Double, Double)] = redeictioAndLabelsTest.rdd
    val matricesTest = new BinaryClassificationMetrics(redeictioAndLabelsTestrdd)
    println("Area under ROC = " + matricesTest.areaUnderROC())
    println("area under the precision-recall curve: " + matricesTest.areaUnderPR)

    
//save data
    
    result = result.select("PassengerId","prediction")
    
    val submitRDD = result.map {row => 
      (row.get(0).asInstanceOf[Int],row.get(1).asInstanceOf[Double])
      }
    
     val submitRDDrdd : RDD[(Int, Double)] = submitRDD.rdd
      submitRDDrdd.saveAsTextFile("D:\\bigdata\\spark\\testdata\\titanic\\output")
    
    
    
  }
}