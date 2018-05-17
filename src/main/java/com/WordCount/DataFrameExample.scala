package com.WordCount

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext


object DataFrameExample {
  
  case class Customer(customer_id:Int, name:String, city:String, state:String, zipcode:String)
  
  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("test");
    val sc=new SparkContext(conf);
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    
  
     
    val data = sc.textFile("D:\\bigdata\\spark\\testdata\\customers.txt");

      
    val dataDF = data.map (_.split(",")).map(x=>Customer(x(0).trim.toInt,x(1),x(2),x(3),x(4))).toDF()
    dataDF.createOrReplaceTempView("customers")
    
    //sql
    val custNames = sqlContext.sql("SELECT name FROM customers")
    custNames.map(t=>"Name:"+t(0)).collect().foreach(print)
    
    custNames.map(t => "Name: " + t.getAs[String]("name")).collect().foreach(println)
    
  }
}