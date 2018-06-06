package com.WordCount

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.Dataset
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object DataseCaseClass {
  case class Account(state: String, len: Integer, acode: String,
    intlplan: String, vplan: String, numvmail: Double,
    tdmins: Double, tdcalls: Double, tdcharge: Double,
    temins: Double, tecalls: Double, techarge: Double,
    tnmins: Double, tncalls: Double, tncharge: Double,
    timins: Double, ticalls: Double, ticharge: Double,
    numcs: Double, churn: String)

val schema = StructType(Array(
    StructField("state", StringType, true),
    StructField("len", IntegerType, true),
    StructField("acode", StringType, true),
    StructField("intlplan", StringType, true),
    StructField("vplan", StringType, true),
    StructField("numvmail", DoubleType, true),
    StructField("tdmins", DoubleType, true),
    StructField("tdcalls", DoubleType, true),
    StructField("tdcharge", DoubleType, true),
    StructField("temins", DoubleType, true),
    StructField("tecalls", DoubleType, true),
    StructField("techarge", DoubleType, true),
    StructField("tnmins", DoubleType, true),
    StructField("tncalls", DoubleType, true),
    StructField("tncharge", DoubleType, true),
    StructField("timins", DoubleType, true),
    StructField("ticalls", DoubleType, true),
    StructField("ticharge", DoubleType, true),
    StructField("numcs", DoubleType, true),
    StructField("churn", StringType, true)
  ))
  
  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("test");
    val sc=new SparkContext(conf);
        val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
     val train: Dataset[Account] = sqlContext.read.option("inferSchema", "false")
      .schema(schema).csv("/user/user01/data/churn-bigml-80.csv").as[Account]
  }
  
 
}