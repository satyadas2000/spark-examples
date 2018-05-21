package com.kafka

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
  
  case class User(id: Int, name: String, email:String)
  
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
    val schema: Schema = new Parser().parse(Source.fromFile("D:\\bigdata\\spark\\testdata\\avro\\schema.avsc").mkString)
    
    
    def send(topic:String, users:List[User]) : Unit ={
      val genericUser: GenericRecord = new GenericData.Record(schema)
      
      try{
       val messages= users.map { user =>
         genericUser.put("id", user.id)
         genericUser.put("name", user.name)
         genericUser.put("email", user.email)
         
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
    val user1 = User(1, "abc","a@a.com")
    val user2 = User(2, "xyz", "b@b.com")

    send(topic, List(user1, user2))
    
  }
}