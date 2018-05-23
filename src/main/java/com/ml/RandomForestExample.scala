package com.ml

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


/**
 * Data you can download from
 * https://apsportal.ibm.com/exchange-api/v1/entries/8044492073eb964f46597b4be06ff5ea/data?accessKey=9561295fa407698694b1e254d0099600
 */
object RandomForestExample {

	def main(args: Array[String]){
		val conf = new SparkConf().setAppName("test");
		val sc=new SparkContext(conf);
		val sqlContext = new org.apache.spark.sql.SQLContext(sc)
		import sqlContext.implicits._

		var dataDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("D:\\bigdata\\spark\\testdata\\sales\\GoSales_Tx_NaiveBayes.csv").cache()

		//dataDF = dataDF.withColumn("label", dataDF("PRODUCT_LINE"))
		val stringIndexer_label = new StringIndexer().setInputCol("PRODUCT_LINE").setOutputCol("label").fit(dataDF)
		val stringIndexer_gend = new StringIndexer().setInputCol("GENDER").setOutputCol("GENDER_IX")
		val encoder = new OneHotEncoder().setInputCol("GENDER_IX").setOutputCol("gendervect")
		val stringIndexer_mar = new StringIndexer().setInputCol("MARITAL_STATUS").setOutputCol("MARITAL_STATUS_IX")
		val maritalencoder = new OneHotEncoder().setInputCol("MARITAL_STATUS_IX").setOutputCol("maritalvect")    
		val stringIndexer_prof = new StringIndexer().setInputCol("PROFESSION").setOutputCol("PROFESSION_IX")



		val assembler = new VectorAssembler().setInputCols(Array("gendervect","AGE","maritalvect","PROFESSION_IX")).setOutputCol("features")

		val rm = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(10)
		
		val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(stringIndexer_label.labels)

		val pipeline = new Pipeline().setStages(Array(stringIndexer_label,stringIndexer_gend,encoder,stringIndexer_mar,maritalencoder,stringIndexer_prof,assembler,rm,labelConverter))
		
		//split data
		val splits = dataDF.randomSplit(Array(0.8,0.18,0.02), seed=24L)
		val trainDF = splits(0).cache()
		val testDF = splits(1).cache()
		val predictDF = splits(2).cache()
		
		
		val model = pipeline.fit(trainDF)
		
		val result = model.transform(testDF)
		
		val evaluateRF = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
		val accuracy = evaluateRF.evaluate(result)
		
		println("Accuracy ="+ accuracy)
		
		//accuracy is .58 - not good
		
		val resultPredicted = model.transform(predictDF)
		
		resultPredicted.select("predictedLabel", "PRODUCT_LINE").show()
		

	}
}