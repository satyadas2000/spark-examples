package com.WordCount

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.rdd.RDD
import com.datastax.spark.connector._
import java.util.UUID

object KMean {
   
  def main(args : Array[String]){
    val conf = new SparkConf().setAppName("test").set("spark.cassandra.connection.host", "127.0.0.1");
    val sc=new SparkContext(conf);
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    
    var dataDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("D:\\bigdata\\spark\\testdata\\kmean\\test.csv").cache()
    //missing value
    var salavg = dataDF.select(mean("income")).first()(0).asInstanceOf[Double]
    dataDF = dataDF.na.fill(salavg)
    
    val genderIndex = new StringIndexer().setInputCol("gender").setOutputCol("genderIndex")
    val encoder = new OneHotEncoder().setInputCol("genderIndex").setOutputCol("gendervect")
    val assembler = new VectorAssembler().setInputCols(Array("income","gendervect")).setOutputCol("features")
    val lr = new KMeans().setK(3).setFeaturesCol("features").setPredictionCol("prediction")
    
    val pipeline = new Pipeline().setStages(Array(genderIndex,encoder,assembler,lr))
    val model = pipeline.fit(dataDF)
    val predictionResult = model.transform(dataDF)
    

    
    
    val result = predictionResult.select("email","prediction")
    
        //save to cassandra

    
  val user_table = sc.cassandraTable("library", "books")
  val saveRDD1 = result.map(row => (row.get(0).asInstanceOf[String], row.get(1).asInstanceOf[Integer]))
    
     val submitRDDrdd1 : RDD[(String, Integer)] = saveRDD1.rdd
  
     submitRDDrdd1.saveAsCassandraTable("library", "books_1")
     
     //save to file system
    
    
    val saveRDD = result.map(row => (row.get(0).asInstanceOf[String], row.get(1).asInstanceOf[Integer]))
    
     val submitRDDrdd : RDD[(String, Integer)] = saveRDD.rdd
     
    submitRDDrdd.saveAsTextFile("D:\\bigdata\\spark\\testdata\\kmean\\output")
    

  }
}