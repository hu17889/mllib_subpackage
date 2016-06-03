/**
 * Created by 58 on 2015/4/11.
 */

import java.util.Calendar

import org.apache.spark.mllib.tree.{TreeInput, GradientBoostedTrees,LambdaMART}
import org.apache.spark.mllib.tree.configuration.LambdaBoostingStrategy
import org.apache.spark.mllib.tree.model.LambdaGradientBoostedTreesModel
import org.apache.spark.mllib.tree.loss.{Loss,AbsoluteError,SquaredError}
import org.apache.spark.mllib.tree.impurity.LambdaVariance
import org.apache.spark.SparkContext

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.LambdaLabeledPoint

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector=>LinalgVector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.collection.mutable.{StringBuilder, Map}
import scala.util.Random

import java.lang.Runtime

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path

import org.apache.commons.cli._

import common.mutil

class lambda_mart(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) {

  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data path")
    options.addOption("te", "testdata", true, "test data path")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; 40. gbdt_regression")
    options.addOption("nt", "numTrees", true, "num of trees")
    options.addOption("lr", "learningRate", true, "learning rate")
    options.addOption("loss", "loss", true, "SquaredError or AbsoluteError")
    options.addOption("imp", "impurity", true, "LambdaVariance or other")
    options.addOption("tmd", "treeMaxDepth", true, "max depth")
    options.addOption("tmb", "treeMaxBins", true, "max seperate number of bins")
    options.addOption("tsr", "treeSubsamplingRate", true, "sbsampling rate of all sample for san")
    options.addOption("tmp", "treeMinInstancesPerNode", true, "Convergence condition")
    options.addOption("tmg", "treeMinInfoGain", true, "Convergence condition")
    options.addOption("tmm", "treeMaxMemoryInMB", true, "")

    val cl = parser.parse( options, args )

    if( cl.hasOption('h') ) {
      val f:HelpFormatter = new HelpFormatter()
      f.printHelp("OptionsTip", options)
    }

    val cmdMap = Map[String, String]()
    val algo = cl.getOptionValue("algo")
    cmdMap += ("algo" -> algo)
    cmdMap += ("traindata"->cl.getOptionValue("traindata"))
    cmdMap += ("testdata"->cl.getOptionValue("testdata"))
    cmdMap += ("dstdata"->cl.getOptionValue("dstdata"))
    cmdMap += ("numTrees"->cl.getOptionValue("numTrees", "100"))
    cmdMap += ("learningRate"->cl.getOptionValue("learningRate", "0.1"))
    cmdMap += ("loss"->cl.getOptionValue("loss", "SquaredError"))
    cmdMap += ("impurity"->cl.getOptionValue("impurity", "LambdaVariance"))
    cmdMap += ("treeMaxDepth"->cl.getOptionValue("treeMaxDepth", "5"))
    cmdMap += ("treeMaxBins"->cl.getOptionValue("treeMaxBins", "32"))
    cmdMap += ("treeSubsamplingRate"->cl.getOptionValue("treeSubsamplingRate", "1"))
    cmdMap += ("treeMinInstancesPerNode"->cl.getOptionValue("treeMinInstancesPerNode", "100"))
    cmdMap += ("treeMinInfoGain"->cl.getOptionValue("treeMinInfoGain", "0.0"))
    cmdMap += ("treeMaxMemoryInMB"->cl.getOptionValue("treeMaxMemoryInMB", "256"))

    return  cmdMap
  }

  def saveModel( model:LambdaGradientBoostedTreesModel, outputPath:String, analyseLog:StringBuilder): Unit = {
    analyseLog.append("---------model info-----------\n")
    analyseLog.append("numTrees = "+model.numTrees+"\n")
    print("numTrees = "+model.numTrees+"\n")
    analyseLog.append("totalNumNodes = "+model.totalNumNodes+"\n")
    print("totalNumNodes = "+model.totalNumNodes+"\n")
    analyseLog.append("summary of the model = "+model.toString()+"\n")
    print("summary of the model = "+model.toString()+"\n")
    analyseLog.append("full model = "+model.toDebugString+"\n")
    print("full model = "+model.toDebugString+"\n")
    mutil.saveArray(fs,model.treeWeights,outputPath+"/treeweight")
    model.save(sc,outputPath+"/treemodel")
  }


  def predictData(testData:RDD[LambdaLabeledPoint], model:LambdaGradientBoostedTreesModel, outputPath:String, analyseLog:StringBuilder,datalabel:String): Unit = {
    val scoreAndLabels = testData.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }.cache()

    // Get evaluation.
    // recall precision
    scoreAndLabels.saveAsTextFile(outputPath+"/"+datalabel+"_scoreAndLabels")
    val r1 = scoreAndLabels.filter{case(score,label) => score>0.5 && label>=1}.count()
    val r2 = scoreAndLabels.filter{case(score,label) => score>0.5 && label==0}.count()
    val r3 = scoreAndLabels.filter{case(score,label) => score<=0.5 && label>=1}.count()
    val r4 = scoreAndLabels.filter{case(score,label) => score<=0.5 && label==0}.count()
    analyseLog.append("recall = " + r1.toFloat/(r1.toFloat+r3.toFloat) + " = " + r1 + "/(" + r1 + "+" + r3 + ")\n")
    print("recall = " + r1.toFloat/(r1.toFloat+r3.toFloat) + " = " + r1 + "/(" + r1 + "+" + r3 + ")\n")
    analyseLog.append("precision = " + r1.toFloat/(r1.toFloat+r2.toFloat) + " = " + r1 + "/(" + r1 + "+" + r2 + ")\n")
    print("precision = " + r1.toFloat/(r1.toFloat+r2.toFloat) + " = " + r1 + "/(" + r1 + "+" + r2 + ")\n")
    analyseLog.append("accuracy = " + (r1.toFloat+r4.toFloat)/(r1.toFloat+r2.toFloat+r3.toFloat+r4.toFloat) + " = " + "(" + r1 + "+" + r4 + ")" + "/(" + r1 + "+" + r2 + "+" + r3 + "+" + r4 + ")\n")
    print("accuracy = " + (r1.toFloat+r4.toFloat)/(r1.toFloat+r2.toFloat+r3.toFloat+r4.toFloat) + " = " + "(" + r1 + "+" + r4 + ")" + "/(" + r1 + "+" + r2 + "+" + r3 + "+" + r4 + ")\n")

    // auc
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("AUC = " + auROC)
    analyseLog.append("AUC = " + auROC+"\n")

    // pr curve
    val pr = metrics.pr()
    pr.saveAsTextFile(outputPath+"/"+datalabel+"_pr")

    scoreAndLabels.unpersist()

  }

  def deal(): Unit = {
    val outputPath = cmdMap("dstdata")
    val algo = cmdMap("algo")
    analyseLog.append("input params : " + cmdMap.toString() + "\n")

    // Load training data in LIBSVM format.
    val trainDataSor = TreeInput.makeLambdaLabeledPoint(sc,cmdMap("traindata")).cache()
    //trainData.map(_.toString()).saveAsTextFile(outputPath+"/traindata")
    val testDataSor =TreeInput.makeLambdaLabeledPoint(sc,cmdMap("testdata"))
    //testData.map(_.toString()).saveAsTextFile(outputPath+"/testdata")

    // filter
    //
    val trainFilter = trainDataSor.map{case lp=>((lp.qid),lp)}.groupByKey().filter{case (key,points)=>
      val allCount = points.toArray.length
      val positiveCount = points.toArray.filter(_.label>=1).length
      val negtiveCount = points.toArray.filter(_.label<1).length
      allCount>1&&negtiveCount>=1
    }.flatMap(_._2)
    val testFilter = testDataSor.map{case lp=>((lp.qid),lp)}.groupByKey().filter{case (key,points)=>
      val allCount = points.toArray.length
      val positiveCount = points.toArray.filter(_.label>=1).length
      val negtiveCount = points.toArray.filter(_.label<1).length
      allCount>1&&negtiveCount>=1
    }.flatMap(_._2)
    // distinct(qid,pos)
    val trainData = trainFilter.map{case lp=>((lp.qid,lp.pos),lp)}.groupByKey().map{ case (key,lps) =>
      val points = lps.toArray
      val positivePoint = points.filter{_.label>=1}
      val negtivePoint = points.filter{_.label<1}
      if(positivePoint.length>=1) positivePoint(0) else negtivePoint(0)
    }
    val testData = testFilter.map{case lp=>((lp.qid,lp.pos),lp)}.groupByKey().map{ case (key,lps) =>
      val points = lps.toArray
      val positivePoint = points.filter{_.label>=1}
      val negtivePoint = points.filter{_.label<1}
      if(positivePoint.length>=1) positivePoint(0) else negtivePoint(0)
    }

    println("************traindata line number = "+trainData.count())
    println("************testdata line number = "+testData.count())
    println("************traindata's partitions number = "+trainData.partitions.size)
    analyseLog.append("traindata number = "+trainData.count()+"\n")
    analyseLog.append("testdata line number = "+testData.count()+"\n")

    // Train a GradientBoostedTrees model.
    //  The defaultParams for Regression use SquaredError by default.
    val boostingStrategy = if(cmdMap("algo")=="41") LambdaBoostingStrategy.defaultParams("Regression") else LambdaBoostingStrategy.defaultParams("Classification")
    //val boostingStrategy = LambdaBoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = cmdMap("numTrees").toInt
    boostingStrategy.learningRate = cmdMap("learningRate").toDouble
    if(cmdMap("loss")=="SquaredError")
      boostingStrategy.loss = SquaredError
    else if(cmdMap("loss")=="AbsoluteError")
      boostingStrategy.loss = AbsoluteError

    boostingStrategy.treeStrategy.maxDepth = cmdMap("treeMaxDepth").toInt
    boostingStrategy.treeStrategy.maxBins = cmdMap("treeMaxBins").toInt
    boostingStrategy.treeStrategy.subsamplingRate = cmdMap("treeSubsamplingRate").toDouble
    boostingStrategy.treeStrategy.minInstancesPerNode = cmdMap("treeMinInstancesPerNode").toInt
    boostingStrategy.treeStrategy.minInfoGain = cmdMap("treeMinInfoGain").toDouble
    boostingStrategy.treeStrategy.maxMemoryInMB = cmdMap("treeMaxMemoryInMB").toInt
    boostingStrategy.treeStrategy.impurity = LambdaVariance


    val model = LambdaMART.train(trainData, boostingStrategy)
    saveModel(model,outputPath,analyseLog)



    // Predict
    analyseLog.append("---------predict train data-----------\n")
    predictData(trainData,model,outputPath,analyseLog,"train")
    analyseLog.append("---------predict test data-----------\n")
    predictData(testData,model,outputPath,analyseLog,"test")

    // analyse output
    val etime = Calendar.getInstance().getTimeInMillis
    val exeTime = etime-this.stime
    analyseLog.append("execution time = "+mutil.formatDuring(exeTime)+"\n")
    println("***********************execution time = "+mutil.formatDuring(exeTime)+"\n")
    val analysePath = new Path(outputPath+"/analyse")
    val outEval2 = fs.create(analysePath,true)
    outEval2.writeBytes(analyseLog.toString())
    outEval2.close()
  }


}
