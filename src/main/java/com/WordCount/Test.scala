package com.WordCount

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager

object Test extends Serializable{
  def main(args: Array[String]){
     val log = LogManager.getRootLogger
     log.info("Inside main method")
    val conf = new SparkConf().setAppName("Test").setMaster("local");
    val sc = new SparkContext(conf);
    
    val txt = sc.textFile("D:\\bigdata\\spark\\testdata\\input.txt");
    val fltmap = txt.flatMap { x => x.split(" ") };
    val cntmap = fltmap.map { x => 
      log.info("mapping: " + x)
      (x,1)};
    val cntarry = cntmap.reduceByKey((x,y)=>x+y);
    cntarry.saveAsTextFile("D:\\bigdata\\spark\\testdata\\output.txt")
    sc.stop()
    
    //spark-submit --class com.WordCount.Test --master local[12] D:\bigdata\spark\testdata\WordCount-0.0.1-SNAPSHOT.jar
  }
}