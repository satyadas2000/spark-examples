package com.ml

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.param.IntArrayParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.log4j.Logger

object ArtificialNeuralNetwork {
  val rootLogger = Logger.getRootLogger()
  def main(args: Array[String]){
    	val conf = new SparkConf().setAppName("test");
		val sc=new SparkContext(conf);
		val sqlContext = new org.apache.spark.sql.SQLContext(sc)
		import sqlContext.implicits._

		//data received from
		//https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
		
		var df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("D:\\bigdata\\spark\\testdata\\loan\\train_u6lujuX_CVtuZ9i.csv").cache()

		//df.filter(df("Self_Employed").isNull || df("Self_Employed") === "" || df("Self_Employed").isNaN).count()
	//	df.filter(df("Gender").isNull || df("Gender") === "" || df("Gender").isNaN).count()
		//df.filter(df("Married").isNull || df("Married") === "" || df("Married").isNaN).count()
		//df.filter(df("Education").isNull || df("Education") === "" || df("Education").isNaN).count()
	//	df.filter(df("Property_Area").isNull || df("Property_Area") === "" || df("Property_Area").isNaN).count()
		//df.filter(df("Loan_Status").isNull || df("Loan_Status") === "" || df("Loan_Status").isNaN).count()

		//fill NO value as , most of the columns has no value
		df=df.na.fill("No",Seq("Self_Employed"))
		df=df.na.fill("Yes",Seq("Married"))
		df=df.na.fill("Male",Seq("Gender"))
		df=df.na.fill("Urban",Seq("Property_Area"))
		df=df.na.fill(0,Seq("Credit_History"))
		df=df.na.fill(0,Seq("Loan_Amount_Term"))

		//fill missing values
		//fill nan values "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"
		//var avgApplicantIncome = df.select(mean("ApplicantIncome")).first()(0).asInstanceOf[Double]
			//	df=df.na.fill(avgApplicantIncome, Seq("ApplicantIncome"))

				var avgLoanAmount = df.select(mean("LoanAmount")).first()(0).asInstanceOf[Double]
						df=df.na.fill(avgLoanAmount, Seq("LoanAmount"))

						var avgLoan_Amount_Term = df.select(mean("Loan_Amount_Term")).first()(0).asInstanceOf[Double]
								df=df.na.fill(avgLoan_Amount_Term, Seq("Loan_Amount_Term"))

								//we can add ApplicantIncome,CoapplicantIncome to create new columns
								//ApplicantIncome is integer , convert to double
								val toDouble = sqlContext.udf.register("toDouble", ((n: Int) => { n.toDouble }))
								df= df.withColumn("ApplicantIncome", toDouble(df("ApplicantIncome")))
								
							
								df= df.withColumn("CoapplicantIncome", toDouble(df("CoapplicantIncome")))
								df= df.withColumn("Credit_History", toDouble(df("Credit_History")))

								val addIncomes = sqlContext.udf.register("addIncomes", (ApplicantIncome: Double, CoapplicantIncome: Double) => {ApplicantIncome+CoapplicantIncome})
								df= df.withColumn("TotalIncome", addIncomes(df("ApplicantIncome"),df("CoapplicantIncome")))

								//Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status

								val stringIndexer_label = new StringIndexer().setInputCol("Loan_Status").setOutputCol("label").fit(df)
								
								val Gender_ix = new StringIndexer().setInputCol("Gender").setOutputCol("Gender_ix")
								val Gender_ix_vect = new OneHotEncoder().setInputCol("Gender_ix").setOutputCol("Gender_ix_vect")

								val Married_ix = new StringIndexer().setInputCol("Married").setOutputCol("Married_ix")
								val Married_ix_vect = new OneHotEncoder().setInputCol("Married_ix").setOutputCol("Married_ix_vect")

								val Education_ix = new StringIndexer().setInputCol("Education").setOutputCol("Education_ix")

								val Self_Employed_ix = new StringIndexer().setInputCol("Self_Employed").setOutputCol("Self_Employed_ix")
								val Self_Employed_vect = new OneHotEncoder().setInputCol("Self_Employed_ix").setOutputCol("Self_Employed_vect")

								val Property_Area_ix = new StringIndexer().setInputCol("Property_Area").setOutputCol("Property_Area_ix")

								//now we will check accuracy of model one by one variable

								val assembler = new VectorAssembler().setInputCols(Array("Gender_ix","Married_ix","Self_Employed_ix","Education_ix","Property_Area_ix","TotalIncome","LoanAmount","Credit_History","Loan_Amount_Term")).setOutputCol("features")

               
                
								//val layers = Array[Int](8,  28, 25, 2)   //4 layers, computational units 8,28,25,2
								val hiddenLayerOpts = "35,32"
								val iterationOpts = "200"
								val blockSizeOpts = "128"
								val layerOptions = hiddenLayerOpts.split(",").map(_.toInt).map(x=> Array[Int](9,x,2))
								//9- is number of features , 2 is no of classes predicted (here loan_status Y, N)
								//hiddenLayerOpts - 2 hidden layers consist of 35, 32 computational units respectively

								val iterationOptions = iterationOpts.split(",").map(_.toInt)
								val blockSizeOptions = blockSizeOpts.split(",").map(_.toInt)

								val ann = new MultilayerPerceptronClassifier().setSeed(1234L).setLabelCol("label").setFeaturesCol("features")
								val paramGrid3 = new ParamGridBuilder()
                		.addGrid(ann.layers, layerOptions)
                		.addGrid(ann.maxIter, iterationOptions)
                		.addGrid(ann.blockSize, blockSizeOptions)
                		.build()
    					
    					
    					          
             val pipeline3 = new Pipeline().setStages(Array(stringIndexer_label,Gender_ix, Married_ix,Self_Employed_ix,Education_ix,Property_Area_ix,assembler,ann))
          
          		val splits3 = df.randomSplit(Array(0.8,0.2), seed=24L)
          		val trainDF3 = splits3(0).cache()
          		val testDF3 = splits3(1).cache()
              val evaluator3 = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("label")

          		val cv3 = new CrossValidator().setEstimator(pipeline3).setEvaluator(evaluator3).setEstimatorParamMaps(paramGrid3).setNumFolds(10) 

          		val pipelineFittedModel3 = cv3.fit(trainDF3)
          		val predictions3 = pipelineFittedModel3.transform(testDF3)
              val accuracy3 = evaluator3.evaluate(predictions3) 
              
              
              
              val predictionAndLabels3 =predictions3.select("prediction", "label").rdd.map(x =>
                (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
              val metrics3 = new BinaryClassificationMetrics(predictionAndLabels3)
		
		           val rows= predictionAndLabels3.collect()
		           var totalLabel0,totalLabel1=0
		           var totalPred0,totalPred1=0
		           
		           for(x <-rows ){
		             val label = x._2;
                 val prediction = x._1;
                 if(label == 0.0){
                   totalLabel0+=1
                 }else{
                   totalLabel1+=1
                 }
                if(prediction == 0.0){
                   totalPred0+=1
                 }else{
                   totalPred1+=1
                 }
                 
		           }
		
		println("****************************************************************totalLabel0 : " + totalLabel0)
		println("****************************************************************totalLabel1 : " + totalLabel1)
		println("****************************************************************totalPred0 : " + totalPred0)
		println("****************************************************************totalPred1 : " + totalPred1)
		
				rootLogger.info("****************************************************************totalLabel0 : " + totalLabel0)
		rootLogger.info("****************************************************************totalLabel1 : " + totalLabel1)
		rootLogger.info("****************************************************************totalPred0 : " + totalPred0)
		rootLogger.info("****************************************************************totalPred1 : " + totalPred1)
		           
		            var correctlyPredicted0=0;
		        var correctlyPredicted1=0;
		        var wronglyPredicted1=0;
		        var wronglyPredicted0=0;
		           
		           for ( x <- rows ) {
               val label = x._2;
                 val prediction = x._1;
                  if(label == 0.0){
                if(prediction == 0.0){
                    correctlyPredicted0=correctlyPredicted0+1;
                }else{
                    wronglyPredicted1=wronglyPredicted1+1;
                }
            }else{
                if(prediction == 1.0){
                    correctlyPredicted1=1+correctlyPredicted1;
                }else{
                    wronglyPredicted0=wronglyPredicted0+1;
                }
            }
                  }
		
		       
		
		        
		
		     
		println("****************************************************************correctlyPredicted0 : " + correctlyPredicted0)
		println("****************************************************************correctlyPredicted1 : " + correctlyPredicted1)
		println("****************************************************************wronglyPredicted1 : " + wronglyPredicted1)
		println("****************************************************************wronglyPredicted0 : " + wronglyPredicted0)
		
		
		         println("****************************************************************MultilayerPerceptron accuracy : " + accuracy3)
		         println("area under the receiver operating characteristic (ROC) curve : " + metrics3.areaUnderROC)
		         
		         val bestModel3 = pipelineFittedModel3.bestModel
             val treeModel3 = bestModel3.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(7).asInstanceOf[MultilayerPerceptronClassificationModel]
            // println("Learned classification tree model:\n" + treeModel3.toDebugString)

		         val rm3 = new RegressionMetrics(predictions3.select("prediction", "label").rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))) 
             println("MSE: " + rm3.meanSquaredError) 
             println("MAE: " + rm3.meanAbsoluteError) 
             println("RMSE Squared: " + rm3.rootMeanSquaredError) 
             println("R Squared: " + rm3.r2) 
             println("Explained Variance: " + rm3.explainedVariance + "\n")
             
              val metricsConfustion3 = new MulticlassMetrics(predictionAndLabels3)
              println("Confusion matrix MultilayerPerceptronClassifier:"+metricsConfustion3.confusionMatrix)
              rootLogger.info("Confusion matrix MultilayerPerceptronClassifier:"+metricsConfustion3.confusionMatrix)
  }
}