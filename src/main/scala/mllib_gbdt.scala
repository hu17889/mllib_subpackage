/**
 * Created by 58 on 2015/4/11.
 */

import java.util.Calendar

import common.{MBLAS, mutil}
import org.apache.commons.cli._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.tree.loss.{AbsoluteError, LogLoss, SquaredError}
//import org.apache.spark.mllib.tree.model.{Split, Node}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, GradientBoostedTreesModel, Node, Split}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.{Map, StringBuilder}

class mllib_gbdt(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args)  with Logging{

  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("pre", "predict", true, "predict model")
    options.addOption("mf","modelpath",true,"model path")
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
    options.addOption("scal", "useFeatureScaling", true, "")
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
    cmdMap +=("predict"->cl.getOptionValue("predict","0"))
    cmdMap +=("model"->cl.getOptionValue("modelpath"))
    cmdMap +=("useFeatureScaling"->cl.getOptionValue("useFeatureScaling","0"))
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
   // GradientBoostedTreesModel.load()

    //model.trees(1)

  }

  def subtreeToString(node:Node,indentFactor: Int = 0,vectorstd:Vector): String = {

    def splitToString(split: Split, left: Boolean,vectorstd:Vector): String = {
      split.featureType match {
        case Continuous =>
         // val mean = vectormean(split.feature).toString
          val std = vectorstd(split.feature).toString
          val threshold = split.threshold
          val thresholdmiddle = threshold * std.toDouble
          val recoverthreshold = f"$thresholdmiddle%1.4f".toDouble
          if (left) {
          s"(feature ${split.feature} <= ${recoverthreshold})"
        } else {
          s"(feature ${split.feature} > ${recoverthreshold})"
        }
      }
    }
    val prefix: String = " " * indentFactor
    if (node.isLeaf) {
      prefix + s"Predict: ${node.predict.predict}\n"
    } else {
      prefix + s"If ${splitToString(node.split.get, left = true,vectorstd)}\n" +
        subtreeToString(node.leftNode.get,indentFactor + 1,vectorstd) +
        prefix + s"Else ${splitToString(node.split.get, left = false,vectorstd)}\n" +
        subtreeToString(node.rightNode.get,indentFactor + 1,vectorstd)
    }
  }



  def saveScalModel(model:GradientBoostedTreesModel, outputPath:String,analyseLog:StringBuilder,analyseLog2:StringBuilder,vectorstd:Vector):Unit = {

    val trees :Array[DecisionTreeModel] = model.trees
    val toScalString: String = {
      val header = toString + "\n"
      header + trees.zipWithIndex.map { case (tree, treeIndex) =>
        s"  Tree $treeIndex:\n" + subtreeToString( tree.topNode,4,vectorstd)
      }.fold("")(_ + _)
    }

    analyseLog.append("full model = "+toScalString)
    analyseLog2.append("---------model info-----------\n")
    analyseLog2.append("numTrees = "+model.numTrees+"\n")
    analyseLog2.append("totalNumNodes = "+model.totalNumNodes+"\n")
    analyseLog2.append("summary of the model = "+model.toString()+"\n")
   // analyseLog.append("full model = "+model.toDebugString)
    mutil.saveArray(fs,model.treeWeights,outputPath+"/treeweight")
   // model.save(sc,outputPath+"/treemodel")
      //p.topNode.split = Some(new Split(p.topNode.split.get.feature-1, s, Continuous, List()))
  }


  //useFeatureScaling=1 Normalization
  def scalData(data:RDD[LabeledPoint]):(RDD[LabeledPoint],Vector) = {
    val scaler = if (cmdMap("useFeatureScaling") == "1") {
      println("useFeatureScaling=1")
      //not use mean  x=>x/std
      new StandardScaler(false, true).fit(data.map(_.features))
    } else {
      null
    }
    //val mean = scaler.mean
    val std:Vector = if(scaler != null){
      scaler.std
    }else{
      null
    }

    //val std = scaler.std
    // println("mean:"+mean)
    //println("std:"+std)
    val scal_data =
      if (cmdMap("useFeatureScaling") == "1") {
        // println("useFeatureScaling:"+trainData.count())
        //trainData.map(labeledPoint => (labeledPoint.label, scaler.transform(labeledPoint.features)))
        data.map(lp => new LabeledPoint(lp.label, scaler.transform(lp.features))).cache()
      } else {
        //trainData.map(labeledPoint => (labeledPoint.label, labeledPoint.features))
        data.map(lp => (new LabeledPoint(lp.label, lp.features))).cache()
      }
    (scal_data,std)
  }


  def predictDiffTrees(testData:RDD[LabeledPoint], model:GradientBoostedTreesModel, outputPath:String, analyseLog:StringBuilder,datalabel:String): Unit = {
    println("hahahahahahhhhhhhhhhhhhhhhhhhhhh")
    val scoreAndLabels = testData.map { point =>
      val trees = model.trees
     // println("hahahahaha")
      val nodeinfo = trees.map { p =>
        //val predict = predictDTree(point, p)

       val nodeidAndPre = MBLAS.test(point,p.topNode)
       //val nodeidAndPre = test(point,p.topNode)


       val list :List[Double] = List(nodeidAndPre(0),nodeidAndPre(1))

       (list)
        //predictDTree(point, p)
      }
      val mutablePonitInfoMap = new mutable.HashMap[Int, Int]()
      var count = 1
      for(i <- 0 until nodeinfo.length){
        mutablePonitInfoMap.put(count, (nodeinfo(i))(0).toInt)
        count = count + 1
      }

      val sampleMapInfo = {

        val finalInfo = Array.fill[Int](trees.length)(0)
        for((k,v) <- mutablePonitInfoMap){
          finalInfo(k-1) = (k.toString + v.toString).toInt +1000
        }
        var begin = ""
        for(i <- 0 until finalInfo.length){
          begin = begin +" "+ finalInfo(i).toString() +":1"
      }
        //point.features
        val feaArray = point.features.toSparse.toArray
        var orifea = ""
        for(i <- 0 until feaArray.length){
          orifea = orifea + (i+1).toString +":"+feaArray(i).toString+" "
        }
        (orifea + begin)
      }

     // println("###################################################")
      //println("nodeidAndPre:" + nodeidAndPre)


      //val nodeids = nodeidAndPre.sortBy(_0).toArray
      //nodeidAndPre

      //nodeidAndPre.foldLeft()

      // nodeidAndPre.
      // val treePredictions = trees.map(_.predict(point.features))
      // treePredictions
     // val score = model.predict(point.features)
      (sampleMapInfo)
    }.cache()

    println("length:"+scoreAndLabels.count())

    //analyseLog.append(point.label.toString + sampleMapInfo)
    scoreAndLabels.saveAsTextFile(outputPath+"/"+datalabel+"_testtest")

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
def predict(): Unit ={
  logDebug("###################################################")
  logDebug("start to predict********************")
  val outputPath = cmdMap("dstdata")
  val modelpath = cmdMap("model")
  val testData = MLUtils.loadLibSVMFile(sc, cmdMap("testdata"))
  testData.map(_.toString()).saveAsTextFile(outputPath+"/testdata")
  val analyseLog2 = new StringBuilder()
  logDebug("input params : " + cmdMap.toString() + "\n")
  logDebug("testdata line number = "+testData.count()+"\n")
  analyseLog2.append("input params : " + cmdMap.toString() + "\n")
  analyseLog2.append("testdata line number = "+testData.count()+"\n")
  analyseLog2.append("---------predict test data-----------\n")
  //val modelpath = outputPath+"/treemodel"
  logDebug("begin to load model")
  val model = GradientBoostedTreesModel.load(sc,modelpath)
  predictDiffTrees(testData,model,outputPath,analyseLog2,"test")
  //predictData(testData,model,outputPath,analyseLog2,"test")
  val analysisPath = new Path(outputPath+"/predictAnalysis")
  val outEval3 = fs.create(analysisPath,true)
  outEval3.writeBytes(analyseLog2.toString())
  outEval3.close()
}
  def deal() {
    val outputPath = cmdMap("dstdata")
    val algo = cmdMap("algo")
   // analyseLog.append("input params : " + cmdMap.toString() + "\n")
    sc.setCheckpointDir(outputPath)
    val analyseLog2 = new StringBuilder()
    //val analyseLog3 = new StringBuilder()
    analyseLog2.append("input params : " + cmdMap.toString() + "\n")
    // Load training data in LIBSVM format.
    val trainData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, cmdMap("traindata")).cache()

    val scalinfo = scalData(trainData)
    val scal_traindata = scalinfo._1
    val std = scalinfo._2
    scal_traindata.map(_.toString()).saveAsTextFile(outputPath+"/traindata")
    val testData = MLUtils.loadLibSVMFile(sc, cmdMap("testdata"))
    val scal_testdata = scalData(testData)._1
    scal_testdata.map(_.toString()).saveAsTextFile(outputPath+"/testdata")
    println("************traindata line number = "+scal_traindata.count())
    println("************testdata line number = "+scal_testdata.count())
    println("************traindata's partitions number = "+scal_traindata.partitions.size)
    analyseLog2.append("traindata number = "+scal_traindata.count()+"\n")
    analyseLog2.append("testdata line number = "+scal_testdata.count()+"\n")

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
    else if(cmdMap("loss")=="logloss")
      boostingStrategy.setLoss(LogLoss)

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
      val scalerValid = if (cmdMap("useFeatureScaling") == "1") {
        println("useFeatureScaling=1")
        //not use mean  x=>x/std
        new StandardScaler(false, true).fit(validData.map(_.features))
      } else {
        null
      }
      val scal_validdata = scalData(validData)._1
      scal_validdata.map(_.toString()).saveAsTextFile(outputPath+"/validdata")
      analyseLog2.append("validdata line number = "+scal_validdata.count()+"\n")
      model = new GradientBoostedTrees(boostingStrategy).runWithValidation(scal_traindata,scal_validdata)

      analyseLog2.append("---------predict valid data-----------\n")
      predictData(scal_validdata,model,outputPath,analyseLog2,"valid")
    }else{
      //Logger.debug("strategy.categoricalFeaturesInfo:"+boostingStrategy.categoricalFeaturesInfo)

      model = GradientBoostedTrees.train(scal_traindata, boostingStrategy)
      //saveModel(model,outputPath,analyseLog,analyseLog2)
    }

    if (cmdMap("useFeatureScaling") == "1"){
     // saveScalModel(model,analyseLog,std)
      saveScalModel(model, outputPath,analyseLog,analyseLog2,std)
    }
   // saveScalModel(model,analyseLog,std)
   else{
      saveModel(model,outputPath,analyseLog,analyseLog2)
    }
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
//generate features
    println("begin to generate fature")
    //predictDiffTrees(trainData,model,outputPath,analyseLog2,"train")


    // Predict
    analyseLog2.append("---------predict train data-----------\n")
    predictData(scal_traindata,model,outputPath,analyseLog2,"train")

    analyseLog2.append("---------predict test data-----------\n")
    predictData(scal_testdata,model,outputPath,analyseLog2,"test")

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
    outEval3.writeBytes(analyseLog2.toString())
    outEval3.close()
    //val outorimodelpath = new Path(outputPath+"/model_ori")
   // val outEval4 = fs.create(outorimodelpath,true)
   // outEval4.writeBytes(analyseLog3.toString())
    //outEval4.close()

  }
}
