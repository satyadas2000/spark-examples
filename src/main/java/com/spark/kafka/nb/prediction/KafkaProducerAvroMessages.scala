package com.spark.kafka.nb.prediction

import java.util.Properties
import java.util.UUID

import org.apache.avro.io._
import kafka.producer.{KeyedMessage, Producer, ProducerConfig}
import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.avro.Schema
import org.apache.avro.Schema.Parser
import scala.io.Source
import org.apache.avro.generic.GenericRecord
import org.apache.avro.generic.GenericData
import org.apache.avro.specific.SpecificDatumWriter
import java.io.ByteArrayOutputStream


object KafkaProducerAvroMessages {

  case class IrisData(sepal_length: Double, sepal_width: Double, petal_length:Double, petal_width:Double)
  
  def main(args: Array[String]){
    
    val props = new Properties()
    props.put("metadata.broker.list", "localhost:9092")
    props.put("message.send.max.retries", "5")
    props.put("request.required.acks", "-1")
    props.put("serializer.class", "kafka.serializer.DefaultEncoder")
    props.put("client.id", UUID.randomUUID().toString())
    
    val producer = new Producer[String, Array[Byte]](new ProducerConfig(props))
    
  //Read avro schema file and
 // val schema: Schema = new Parser().parse(Source.fromURL(getClass.getResource("/schema.avsc")).mkString)
    val schema: Schema = new Parser().parse(Source.fromFile("D:\\bigdata\\spark\\testdata\\avro\\nb.avsc").mkString)
    
    
    def send(topic:String, lst:List[IrisData]) : Unit ={
      val genericUser: GenericRecord = new GenericData.Record(schema)
      
      try{
       val messages= lst.map { iris =>
         genericUser.put("sepal_length", iris.sepal_length)
         genericUser.put("sepal_width", iris.sepal_width)
         genericUser.put("petal_length", iris.petal_length)
         genericUser.put("petal_width", iris.petal_width)
         
          // Serialize generic record object into byte array
        val writer = new SpecificDatumWriter[GenericRecord](schema)
        val out = new ByteArrayOutputStream()
        val encoder: BinaryEncoder = EncoderFactory.get().binaryEncoder(out, null)
        
        writer.write(genericUser, encoder)
        encoder.flush()
        out.close()
        
         val serializedBytes: Array[Byte] = out.toByteArray()

        new KeyedMessage[String, Array[Byte]](topic, serializedBytes)
        
          
        }
        producer.send(messages: _*)
      }catch{
        case ex: Exception => ex.printStackTrace()
      }
      
    }
    
     val topic = "avro-topic"
    val d1 = IrisData(5.1,3.5,1.4,0.2)
    val d2 = IrisData(7.0,3.2,4.7,1.4)

    send(topic, List(d1, d2))
    
  }
}