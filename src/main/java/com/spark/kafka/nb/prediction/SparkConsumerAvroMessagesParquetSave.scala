package com.spark.kafka.nb.prediction

import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.spark.sql.SparkSession
import org.apache.avro.generic.GenericRecord
import scala.reflect.runtime.universe._
import scala.io.Source
import org.apache.avro.generic.GenericDatumReader
import org.apache.avro.io.DecoderFactory
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.PipelineModel

object SparkConsumerAvroMessagesParquetSave {
    case class IrisData(sepal_length: Double, sepal_width: Double, petal_length:Double, petal_width:Double)
   val messageSchema = new Schema.Parser().parse(Source.fromFile("D:\\bigdata\\spark\\testdata\\avro\\nb.avsc").mkString)
  val reader = new GenericDatumReader[GenericRecord](messageSchema)
  // Binary decoder
  val decoder = DecoderFactory.get()
  
  def main(args: Array[String]){
     val KafkaBroker = "localhost:9092";
    val InTopic = "avro-topic";

    // Get Spark session
    val session = SparkSession
      .builder
      .master("local[*]")
      .appName("myapp")
      .getOrCreate()
      
      
    val model = PipelineModel.load("D:\\bigdata\\spark\\testdata\\nbsave")

    // Load streaming data
    import session.implicits._
     val data = session
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", KafkaBroker)
      .option("subscribe", InTopic)
      .load()
      .select($"value".as[Array[Byte]])
      .map(d => {
        val rec = reader.read(null, decoder.binaryDecoder(d, null))
        val sepal_length = rec.get("sepal_length").toString.toDouble
        val sepal_width = rec.get("sepal_width").toString.toDouble
        val petal_length = rec.get("petal_length").toString.toDouble
        val petal_width = rec.get("petal_width").toString.toDouble
        // val name = rec.get("name").asInstanceOf[Byte].toString
        //  val email = rec.get("email").asInstanceOf[Byte].toString
       println("sepal_length"+sepal_length)
        println("sepal_width"+sepal_width)
        println("petal_length"+petal_length)
        println("petal_width"+petal_width)
           


   new IrisData(sepal_length,sepal_width,petal_length,petal_width)
     
      })
      
     /* val query = data.writeStream
      .outputMode("Append")
      .format("console")
      .start()*/
      
     val query = data.writeStream
    .format("parquet")        // can be "orc", "json", "csv", etc.
    .outputMode("append")
    .option("checkpointLocation","checkpoints")
    .option("path", "D:\\bigdata\\spark\\testdata\\parquet")
    .start()
      
      

    query.awaitTermination()

   }

}