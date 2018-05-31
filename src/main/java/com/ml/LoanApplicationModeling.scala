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

object LoanApplicationModeling {
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

         //First we will check for LogisticRegression
								
							val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10)

							val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.fitIntercept, Array(true, false)).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()

							val pipeline = new Pipeline().setStages(Array(stringIndexer_label,Gender_ix, Married_ix,Self_Employed_ix,Education_ix,Property_Area_ix,assembler,lr))
          
          		val splits = df.randomSplit(Array(0.8,0.2), seed=24L)
          		val trainDF = splits(0).cache()
          		val testDF = splits(1).cache()
              
          		val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")

          		val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(10) 

          		val pipelineFittedModel = cv.fit(trainDF)
          		val predictions = pipelineFittedModel.transform(testDF)
              val accuracy = evaluator.evaluate(predictions) 
              
              
              
              val predictionAndLabels =predictions.select("prediction", "label").rdd.map(x =>
                (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
              val metrics = new BinaryClassificationMetrics(predictionAndLabels)
		
		         println("****************************************************************LogisticRegression accuracy : " + accuracy)
		         println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)
	
		        val bestModel = pipelineFittedModel.bestModel
            val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(7).asInstanceOf[LogisticRegressionModel]
            println("Learned classification tree model:\n" + treeModel.toString())
		
		         val rm = new RegressionMetrics(predictions.select("prediction", "label").rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))) 
             println("MSE: " + rm.meanSquaredError) 
             println("MAE: " + rm.meanAbsoluteError) 
             println("RMSE Squared: " + rm.rootMeanSquaredError) 
             println("R Squared: " + rm.r2) 
             println("Explained Variance: " + rm.explainedVariance + "\n")
             
             //now we will try for Random Forest Classifier
             
             	val lr1 = new RandomForestClassifier().setImpurity("gini") 
                      .setMaxDepth(30) 
                      .setNumTrees(30) 
                      .setFeatureSubsetStrategy("auto") 
                      .setSeed(1234567) 
                      .setMaxBins(40) 
                      .setMinInfoGain(0.001)
                      .setLabelCol("label")
                      .setFeaturesCol("features")
                      
               val paramGrid1 = new ParamGridBuilder()
                      .addGrid(lr1.maxBins, Array(25, 28, 31))
                      .addGrid(lr1.maxDepth, Array(4, 6, 8))
                      .addGrid(lr1.impurity, Array("entropy", "gini"))
                      .build()

						
							val pipeline1 = new Pipeline().setStages(Array(stringIndexer_label,Gender_ix, Married_ix,Self_Employed_ix,Education_ix,Property_Area_ix,assembler,lr1))
          
          		val splits1 = df.randomSplit(Array(0.8,0.2), seed=24L)
          		val trainDF1 = splits(0).cache()
          		val testDF1 = splits(1).cache()
              

          		val cv1 = new CrossValidator().setEstimator(pipeline1).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid1).setNumFolds(10) 

          		val pipelineFittedModel1 = cv1.fit(trainDF1)
          		val predictions1 = pipelineFittedModel1.transform(testDF1)
              val accuracy1 = evaluator.evaluate(predictions1) 
              
              
              
              val predictionAndLabels1 =predictions1.select("prediction", "label").rdd.map(x =>
                (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
              val metrics1 = new BinaryClassificationMetrics(predictionAndLabels1)
		
		         println("****************************************************************RandomForestClassifier accuracy : " + accuracy1)
		         println("area under the receiver operating characteristic (ROC) curve : " + metrics1.areaUnderROC)
		         
		         pipelineFittedModel1.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0).extractParamMap 
             println("The best fitted model:" + pipelineFittedModel1.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(7).asInstanceOf[RandomForestClassificationModel] ) 

		         val rm1 = new RegressionMetrics(predictions1.select("prediction", "label").rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))) 
             println("MSE: " + rm1.meanSquaredError) 
             println("MAE: " + rm1.meanAbsoluteError) 
             println("RMSE Squared: " + rm1.rootMeanSquaredError) 
             println("R Squared: " + rm1.r2) 
             println("Explained Variance: " + rm1.explainedVariance + "\n")
		

             // Now test with DecisionTree
             
             val lr2 = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
             val paramGrid2 = new ParamGridBuilder().addGrid(lr2.maxDepth,Array(2, 3, 4, 5, 6, 7)).build()
             
             val pipeline2 = new Pipeline().setStages(Array(stringIndexer_label,Gender_ix, Married_ix,Self_Employed_ix,Education_ix,Property_Area_ix,assembler,lr2))
          
          		val splits2 = df.randomSplit(Array(0.8,0.2), seed=24L)
          		val trainDF2 = splits(0).cache()
          		val testDF2 = splits(1).cache()
              

          		val cv2 = new CrossValidator().setEstimator(pipeline2).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid2).setNumFolds(10) 

          		val pipelineFittedModel2 = cv2.fit(trainDF2)
          		val predictions2 = pipelineFittedModel2.transform(testDF2)
              val accuracy2 = evaluator.evaluate(predictions2) 
              
              
              
              val predictionAndLabels2 =predictions2.select("prediction", "label").rdd.map(x =>
                (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
              val metrics2 = new BinaryClassificationMetrics(predictionAndLabels2)
		
		         println("****************************************************************DecisionTreeClassifier accuracy : " + accuracy2)
		         println("area under the receiver operating characteristic (ROC) curve : " + metrics2.areaUnderROC)
		         
		         val bestModel2 = pipelineFittedModel2.bestModel
             val treeModel2 = bestModel2.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(7).asInstanceOf[DecisionTreeClassificationModel]
             println("Learned classification tree model:\n" + treeModel2.toDebugString)

		         val rm2 = new RegressionMetrics(predictions2.select("prediction", "label").rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))) 
             println("MSE: " + rm2.meanSquaredError) 
             println("MAE: " + rm2.meanAbsoluteError) 
             println("RMSE Squared: " + rm2.rootMeanSquaredError) 
             println("R Squared: " + rm2.r2) 
             println("Explained Variance: " + rm2.explainedVariance + "\n")
            
             

	}
}