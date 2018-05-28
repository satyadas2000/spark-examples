package com.WordCount

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, DataFrame}
import java.io.File



object ReadMultipleExcel {
  
  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("test");
		val sc=new SparkContext(conf);
		val sqlContext = new org.apache.spark.sql.SQLContext(sc)
		import sqlContext.implicits._		
		sc.setLogLevel("INFO")
		
  

def readExcel(file: String): DataFrame = sqlContext.read
    .format("com.crealytics.spark.excel")
    .option("location", file)
    .option("useHeader", "true")
    .option("treatEmptyValuesAsNulls", "true")
    .option("inferSchema", "true")
    .option("addColorColumns", "False")
    .load()
		
    
   // val dir = new File("D:\\Users\\245-0117\\Desktop\\lines\\notusedtest\")
    val dir = new File("D:\\Users\\245-0117\\Desktop\\lines\\notusedtest")
    val excelFiles = dir.listFiles.sorted.map(f => f.toString) 
    
    excelFiles.foreach { x => print(x) }
    
    val df = excelFiles.map { x => readExcel(x) }
    val dfd=df.reduce(_.union(_))
		
    dfd.coalesce(1)
  .write.format("com.databricks.spark.csv")
  .option("header", "true")
  .save("D:\\Users\\245-0117\\Desktop\\lines\\notusedtest\\mydata.csv")
		
  }
  
  		
}