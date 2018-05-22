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
import org.apache.spark.SparkConf
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import kafka.producer.ProducerConfig
import org.apache.kafka.clients.consumer.ConsumerConfig
import org.apache.kafka.common.serialization.StringSerializer
import org.apache.kafka.common.serialization.BytesSerializer
import org.apache.spark.sql.streaming.ProcessingTime
import org.apache.spark.sql.ForeachWriter
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.Row




object SparkPredictionAvroMessages {
case class IrisData(sepal_length: Double, sepal_width: Double, petal_length:Double, petal_width:Double)

val schema = StructType(List(
		StructField("sepal_length", DoubleType, true),
		StructField("sepal_width", DoubleType, true),
		StructField("petal_length", DoubleType, true),
		StructField("petal_width", DoubleType, true)
		)
		)

		val messageSchema = new Schema.Parser().parse(Source.fromFile("D:\\bigdata\\spark\\testdata\\avro\\nb.avsc").mkString)
		val reader = new GenericDatumReader[GenericRecord](messageSchema)
		// Binary decoder
		val decoder = DecoderFactory.get()

		def main(args: Array[String]){
	val KafkaBroker = "localhost:9092";
	val InTopic = "avro-topic";
	val batchInterval = "2"


			// Get Spark session
			val session = SparkSession
			.builder
			.master("local[*]")
			.appName("myapp")
			.getOrCreate()

			import session.implicits._

			val model = PipelineModel.load("D:\\bigdata\\spark\\testdata\\nbsave")

			def processRow(row: Row) = {
		val spark = SparkSession
				.builder
				.master("local[*]")
				.appName("myapp")
				.getOrCreate()

				val rdd = spark.sparkContext.makeRDD(List(row))
				val dataFrame = spark.createDataFrame(rdd, schema)

				dataFrame.show()

				val withoutLabelTest = model.transform(dataFrame)

				val lpTest1 = withoutLabelTest.select( "prediction")
				lpTest1.show()

	}





	// Load streaming data

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




						new IrisData(sepal_length,sepal_width,petal_length,petal_width)

			})


			val writer = new ForeachWriter[IrisData] {
		override def open(partitionId: Long, version: Long) = true
				override def process(value: IrisData) = {
			println("sepal_length"+value.sepal_length+"*****************************************************")
			val dataseq = List(value.sepal_length,value.sepal_width,value.petal_length,value.petal_width)
			val row = Row.fromSeq(dataseq)
			processRow(row)

		}
		override def close(errorOrNull: Throwable) = {}
	}

	val query = data.writeStream.foreach(writer).start()





			query.awaitTermination()



}
}