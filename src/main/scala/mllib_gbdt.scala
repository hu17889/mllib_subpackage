/**
 * Created by 58 on 2015/4/11.
 */

import java.util.Calendar

import malgo.{LR_OWLQN, LR_LBFGS}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.Algo.Algo
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, GradientBoostedTreesModel}
import org.apache.spark.mllib.tree.loss.{Loss,AbsoluteError,SquaredError}
import org.apache.spark.SparkContext

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector=>LinalgVector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.{StringBuilder, Map}
import scala.util.Random

import java.lang.Runtime

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path

import org.apache.commons.cli._

import common.mutil

class mllib_gbdt(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) {

  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data path")
    options.addOption("te", "testdata", true, "test data path")
    options.addOption("va","validdata",true,"valid data path")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; 40. gbdt_regression")
    options.addOption("nt", "numTrees", true, "num of trees")
    options.addOption("lr", "learningRate", true, "learning rate")
    options.addOption("loss", "loss", true, "SquaredError or AbsoluteError")
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
    if(cl.getOptionValue("validdata") !=null){
      cmdMap += ("validdata"->cl.getOptionValue("validdata"))
    }

    cmdMap += ("dstdata"->cl.getOptionValue("dstdata"))
    cmdMap += ("numTrees"->cl.getOptionValue("numTrees", "100"))
    cmdMap += ("learningRate"->cl.getOptionValue("learningRate", "0.1"))
    cmdMap += ("loss"->cl.getOptionValue("loss", "SquaredError"))
    cmdMap += ("treeMaxDepth"->cl.getOptionValue("treeMaxDepth", "5"))
    cmdMap += ("treeMaxBins"->cl.getOptionValue("treeMaxBins", "32"))
    cmdMap += ("treeSubsamplingRate"->cl.getOptionValue("treeSubsamplingRate", "1"))
    cmdMap += ("treeMinInstancesPerNode"->cl.getOptionValue("treeMinInstancesPerNode", "100"))
    cmdMap += ("treeMinInfoGain"->cl.getOptionValue("treeMinInfoGain", "0.0"))
    cmdMap += ("treeMaxMemoryInMB"->cl.getOptionValue("treeMaxMemoryInMB", "256"))

    return  cmdMap
  }

  def saveModel( model:GradientBoostedTreesModel, outputPath:String, analyseLog:StringBuilder, analyseLog2:StringBuilder): Unit = {
    analyseLog2.append("---------model info-----------\n")
    analyseLog2.append("numTrees = "+model.numTrees+"\n")
    analyseLog2.append("totalNumNodes = "+model.totalNumNodes+"\n")
    analyseLog2.append("summary of the model = "+model.toString()+"\n")
    analyseLog.append("full model = "+model.toDebugString)
    mutil.saveArray(fs,model.treeWeights,outputPath+"/treeweight")
    model.save(sc,outputPath+"/treemodel")
  }


  def predictData(testData:RDD[LabeledPoint], model:GradientBoostedTreesModel, outputPath:String, analyseLog:StringBuilder,datalabel:String): Unit = {
    val scoreAndLabels = testData.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }.cache()

    // Get evaluation.
    // recall precision
    scoreAndLabels.saveAsTextFile(outputPath+"/"+datalabel+"_scoreAndLabels")
    val r1 = scoreAndLabels.filter{case(score,label) => score>0.5 && label==1}.count()
    val r2 = scoreAndLabels.filter{case(score,label) => score>0.5 && label==0}.count()
    val r3 = scoreAndLabels.filter{case(score,label) => score<=0.5 && label==1}.count()
    val r4 = scoreAndLabels.filter{case(score,label) => score<=0.5 && label==0}.count()
    analyseLog.append("recall = " + r1.toFloat/(r1.toFloat+r3.toFloat) + " = " + r1 + "/(" + r1 + "+" + r3 + ")\n")
    analyseLog.append("precision = " + r1.toFloat/(r1.toFloat+r2.toFloat) + " = " + r1 + "/(" + r1 + "+" + r2 + ")\n")
    analyseLog.append("accuracy = " + (r1.toFloat+r4.toFloat)/(r1.toFloat+r2.toFloat+r3.toFloat+r4.toFloat) + " = " + "(" + r1 + "+" + r4 + ")" + "/(" + r1 + "+" + r2 + "+" + r3 + "+" + r4 + ")\n")

    // auc
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("AUC = " + auROC)
    analyseLog.append("AUC = " + auROC+"\n")

    // pr curve
    val pr = metrics.pr()
    pr.saveAsTextFile(outputPath+"/"+datalabel+"_pr")


  }

  def deal() {
    val outputPath = cmdMap("dstdata")
    val algo = cmdMap("algo")
   // analyseLog.append("input params : " + cmdMap.toString() + "\n")

    sc.setCheckpointDir(outputPath)
    val analyseLog2 = new StringBuilder()
    analyseLog2.append("input params : " + cmdMap.toString() + "\n")
    // Load training data in LIBSVM format.
    val trainData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, cmdMap("traindata")).cache()
    trainData.map(_.toString()).saveAsTextFile(outputPath+"/traindata")
    val testData = MLUtils.loadLibSVMFile(sc, cmdMap("testdata"))
    testData.map(_.toString()).saveAsTextFile(outputPath+"/testdata")




    println("************traindata line number = "+trainData.count())
    println("************testdata line number = "+testData.count())
    println("************traindata's partitions number = "+trainData.partitions.size)
    analyseLog2.append("traindata number = "+trainData.count()+"\n")
    analyseLog2.append("testdata line number = "+testData.count()+"\n")

    //analyseLog2.append("validdata line number = "+validData.count()+"\n")

    // Train a GradientBoostedTrees model.
    //  The defaultParams for Regression use SquaredError by default.
    val boostingStrategy = if(cmdMap("algo")=="40") BoostingStrategy.defaultParams("Regression") else BoostingStrategy.defaultParams("Classification")
    //val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.setNumIterations(cmdMap("numTrees").toInt)
    boostingStrategy.setLearningRate(cmdMap("learningRate").toDouble)
    if(cmdMap("loss")=="SquaredError")
      boostingStrategy.setLoss(SquaredError)
    else if(cmdMap("loss")=="AbsoluteError")
      boostingStrategy.setLoss(AbsoluteError)

    boostingStrategy.getTreeStrategy().setMaxDepth(cmdMap("treeMaxDepth").toInt)
    boostingStrategy.getTreeStrategy().setMaxBins(cmdMap("treeMaxBins").toInt)
    boostingStrategy.getTreeStrategy().setSubsamplingRate(cmdMap("treeSubsamplingRate").toDouble)
    boostingStrategy.getTreeStrategy().setMinInstancesPerNode(cmdMap("treeMinInstancesPerNode").toInt)
    boostingStrategy.getTreeStrategy().setMinInfoGain(cmdMap("treeMinInfoGain").toDouble)
    boostingStrategy.getTreeStrategy().setMaxMemoryInMB(cmdMap("treeMaxMemoryInMB").toInt)

     val numIterations = boostingStrategy.numIterations
     val baseLearners = new Array[DecisionTreeModel](numIterations)
     val bestM = 1
    val baseLearnerWeights = new Array[Double](numIterations)
    var model = new GradientBoostedTreesModel(boostingStrategy.treeStrategy.algo, baseLearners.slice(0, bestM), baseLearnerWeights.slice(0, bestM))
    if(cmdMap.contains("validdata")){
      val validData = MLUtils.loadLibSVMFile(sc, cmdMap("validdata")).cache()
      validData.map(_.toString()).saveAsTextFile(outputPath+"/validdata")
      analyseLog2.append("validdata line number = "+validData.count()+"\n")
      model = new GradientBoostedTrees(boostingStrategy).runWithValidation(trainData,validData)

      analyseLog2.append("---------predict valid data-----------\n")
      predictData(validData,model,outputPath,analyseLog2,"valid")
    }else{
      model = GradientBoostedTrees.train(trainData, boostingStrategy)
      //saveModel(model,outputPath,analyseLog,analyseLog2)
    }
    saveModel(model,outputPath,analyseLog,analyseLog2)
   // val model = new GradientBoostedTrees(boostingStrategy).runWithValidation(trainData,validData)
   //val model =  new GradientBoostedTrees(boostingStrategy).runWithValidation(trainData,validData,boostingStrategy)
    //val model = GradientBoostedTrees.train(trainData, boostingStrategy)


    /*
    val updater =
    if (cmdMap("regtype")=="1") {
      new L1Updater()
    } else if (cmdMap("regtype")=="2") {
      new SquaredL2Updater()
    }
    */



    // Predict
    analyseLog2.append("---------predict train data-----------\n")
    predictData(trainData,model,outputPath,analyseLog2,"train")

    analyseLog2.append("---------predict test data-----------\n")
    predictData(testData,model,outputPath,analyseLog2,"test")

    // analyse output
    val etime = Calendar.getInstance().getTimeInMillis
    val exeTime = etime-this.stime
    analyseLog2.append("execution time = "+mutil.formatDuring(exeTime)+"\n")
    println("***********************execution time = "+mutil.formatDuring(exeTime)+"\n")
    val modelPath = new Path(outputPath+"/model")
    val outEval2 = fs.create(modelPath,true)
    outEval2.writeBytes(analyseLog.toString())
    outEval2.close()
    val analysisPath = new Path(outputPath+"/analysis")
    val outEval3 = fs.create(analysisPath,true)
    outEval3.writeBytes(analyseLog2.toString());
    outEval3.close()


  }
}
