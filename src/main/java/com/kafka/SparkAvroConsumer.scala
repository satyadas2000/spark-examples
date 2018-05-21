package com.kafka

import java.util.Collections


import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.spark.sql.SparkSession
import org.apache.avro.generic.GenericRecord
import org.apache.avro.generic.GenericData
import org.apache.avro.specific.SpecificDatumWriter

import scala.reflect.runtime.universe._
import java.sql.Timestamp
import scala.io.Source
import org.apache.avro.generic.GenericDatumReader
import org.apache.avro.io.DecoderFactory

object KafkaAvroConsumer {
  case class User(id: Int, name: String, email:String)
   val messageSchema = new Schema.Parser().parse(Source.fromFile("D:\\bigdata\\spark\\testdata\\avro\\schema.avsc").mkString)
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
        val id = rec.get("id").toString.toInt
        val name = rec.get("name").toString
        val email = rec.get("email").toString
        // val name = rec.get("name").asInstanceOf[Byte].toString
        //  val email = rec.get("email").asInstanceOf[Byte].toString
        new User(id,name,email)
      })
     
      val query = data.writeStream
      .outputMode("Append")
      .format("console")
      .start()

    query.awaitTermination()
   }

}