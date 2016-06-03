/**
 * Created by 58 on 2015/4/11.
 */

import java.util.Calendar

import malgo.{LR_OWLQN, LR_LBFGS}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.optimization.Updater
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

class mllib_lr(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) {

  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data path")
    options.addOption("te", "testdata", true, "test data path")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; 10. sgd 11. lbfgs  ")
    options.addOption("iw", "initweight", true, "init weight; 0. 0-weights 1. +-rand(0-1) 2. +-rand(0-10)  Default 0")
    options.addOption("rt", "regtype", true, "Regularization type; 1. L1 2. L2   Default 2")
    options.addOption("ip", "iternum", true, "Set the number of iterations.. Default 100.")
    options.addOption("sc", "scale", true, "features will be scaled or not. Default 1.")
    options.addOption("sp", "stepsize", true, "Set the initial step size of SGD for the first step. Default 1.0. In subsequent steps, the step size will decrease with stepSize/sqrt(t)")
    options.addOption("bp", "batchp", true, "Set fraction of data to be used for each SGD iteration. Default 1.0 (corresponding to deterministic/classical gradient descent)")
    options.addOption("rp", "regp", true, "Set the regularization parameter. Default 0.01.")
    options.addOption("tp", "tolerance", true, "Set the convergence tolerance of iterations for L-BFGS. Default 1E-4. Smaller value will lead to higher accuracy with the cost of more iterations. This value must be nonnegative. Lower convergence values are less tolerant and therefore generally cause more iterations to be run.")
    options.addOption("cp", "correction", true, "Set the number of corrections used in the LBFGS update. Default 10. Values of numCorrections less than 3 are not recommended; large values of numCorrections will result in excessive computing time. 3 < numCorrections < 10 is recommended. Restriction: numCorrections > 0")
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
    cmdMap += ("regtype"->cl.getOptionValue("regtype", "2"))
    cmdMap += ("initweight"->cl.getOptionValue("initweight", "0"))
    cmdMap += ("iternum"->cl.getOptionValue("iternum", "1"))
    cmdMap += ("regp"->cl.getOptionValue("regp", "0"))
    cmdMap += ("scale"->cl.getOptionValue("scale", "1"))
    if(algo=="10") {
      cmdMap += ("stepsize"->cl.getOptionValue("stepsize", "1"))
      cmdMap += ("batchp"->cl.getOptionValue("batchp", "1"))
    } else if(algo=="11"||algo=="12"||algo=="13") {
      cmdMap += ("tolerance"->cl.getOptionValue("tolerance", "0.001"))
      cmdMap += ("correction"->cl.getOptionValue("correction", "7"))
    }
    return  cmdMap
  }


  def initWeight(traininput: RDD[LabeledPoint],testinput: RDD[LabeledPoint], wtype:String): LinalgVector = {
    val numFeatures1 = traininput.map(_.features.size).max()
    val numFeatures2 = testinput.map(_.features.size).max()
    val numFeatures = math.max(numFeatures1,numFeatures2)
    println("************weight size = "+numFeatures.toString)
    //analyseLog.append("runtime2 = "+runtimeInfo().toString()+"\n")
    val w1 = new Array[Double](numFeatures)
    //analyseLog.append("runtime3 = "+runtimeInfo().toString()+"\n")
    val w2 = if(wtype=="0") {
      w1.map(_=>0.toDouble)
    } else if (wtype=="1") {
      val p = Random.nextBoolean()
      w1.map{case x => if(p) Random.nextDouble; else -Random.nextDouble }
    } else if (wtype=="2") {
      val p = Random.nextBoolean()
      w1.map{case x => if(p) Random.nextInt(10).toDouble; else -Random.nextInt(10).toDouble }
    }
    //analyseLog.append("runtime4 = "+runtimeInfo().toString()+"\n")
    Vectors.dense(w2.asInstanceOf[Array[Double]])
  }



  def analysisWeight(weights:LinalgVector): Unit = {
    val nonzeroWeiLen = weights.toArray.filter(_!=0).length
    val totalWeiLen = weights.toArray.length
    analyseLog.append("---------weight analyse-----------\n")
    analyseLog.append("nonzero weight length = " + nonzeroWeiLen +"\n")
    analyseLog.append("total weight length = " + totalWeiLen +"\n")
    analyseLog.append("nonzero weight length/total weight length = " + nonzeroWeiLen.toFloat/totalWeiLen.toFloat +"\n")
    val positiveWeights = weights.toArray.filter(_!=0).map(_.abs)
    analyseLog.append("count(|weight|>=10) = " +  positiveWeights.count(_>=10) + "(" + positiveWeights.count(_>=10).toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(1<=|weight|<10) = " +  positiveWeights.count{case x => x>=1 && x<10} + "(" + positiveWeights.count{case x => x>=1 && x<10}.toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(1E-1<=|weight|<1) = " +  positiveWeights.count{case x => x>=1E-1 && x<1} + "(" + positiveWeights.count{case x => x>=1E-1 && x<1}.toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(1E-2<=|weight|<1E-1) = " +  positiveWeights.count{case x => x>=1E-2 && x<1E-1} + "(" + positiveWeights.count{case x => x>=1E-2 && x<1E-1}.toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(1E-3<=|weight|<1E-2) = " +  positiveWeights.count{case x => x>=1E-3 && x<1E-2} + "(" + positiveWeights.count{case x => x>=1E-3 && x<1E-2}.toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(1E-4<=|weight|<1E-3) = " +  positiveWeights.count{case x => x>=1E-4 && x<1E-3} + "(" + positiveWeights.count{case x => x>=1E-4 && x<1E-3}.toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(1E-5<=|weight|<1E-4) = " +  positiveWeights.count{case x => x>=1E-5 && x<1E-4} + "(" + positiveWeights.count{case x => x>=1E-5 && x<1E-4}.toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(1E-6<=|weight|<1E-5) = " +  positiveWeights.count{case x => x>=1E-6 && x<1E-5} + "(" + positiveWeights.count{case x => x>=1E-6 && x<1E-5}.toFloat/nonzeroWeiLen.toFloat +")\n")
    analyseLog.append("count(|weight|<1E-6) = " +  positiveWeights.count{case x => x<1E-6} + "(" + positiveWeights.count{case x => x<1E-6}.toFloat/nonzeroWeiLen.toFloat +")\n")
  }

  def predictData(testData:RDD[LabeledPoint], model:LogisticRegressionModel, outputPath:String, analyseLog:StringBuilder,datalabel:String): Unit = {
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
    analyseLog.append("input params : " + cmdMap.toString() + "\n")

    // Load training data in LIBSVM format.
    val exenum = sc.getConf.get("spark.executor.instances").toInt
    val trainDatain = MLUtils.loadLibSVMFile(sc, cmdMap("traindata")).cache()
    val exenumNow = trainDatain.partitions.size
    val trainData = if(exenumNow<exenum-5) {
      trainDatain.repartition(exenum.toInt-2)
    } else {
      trainDatain
    }
    //val trainData = MLUtils.loadLibSVMFile(sc, cmdMap("traindata")).repartition(100).cache()
    trainData.map(_.toString()).saveAsTextFile(outputPath+"/traindata")
    val testData = MLUtils.loadLibSVMFile(sc, cmdMap("testdata"))
    testData.map(_.toString()).saveAsTextFile(outputPath+"/testdata")

    println("************traindata line number = "+trainData.count())
    println("************testdata line number = "+testData.count())
    println("************traindata's partitions number = "+trainData.partitions.size)
    analyseLog.append("traindata number = "+trainData.count()+"\n")
    analyseLog.append("testdata line number = "+testData.count()+"\n")


    // Run training algorithm to build the model
    //analyseLog.append("runtime1 = "+runtimeInfo().toString()+"\n")
    val initwei = initWeight(trainData, testData, cmdMap("initweight"))

    mutil.saveVector(fs, initwei, outputPath+"/initweight")

    val updater =
    if (cmdMap("regtype")=="1") {
      new L1Updater()
    } else if (cmdMap("regtype")=="2") {
      new SquaredL2Updater()
    }

    val modeltmp:Any =
    if (algo == "10") {
      // sgd
      val lralg = new LogisticRegressionWithSGD()
      lralg.optimizer
        .setNumIterations(cmdMap("iternum").toInt)
        .setStepSize(cmdMap("stepsize").toDouble)
        .setMiniBatchFraction(cmdMap("batchp").toDouble)
        .setRegParam(cmdMap("regp").toDouble)
        .setUpdater(updater.asInstanceOf[Updater])
      lralg.run(trainData,initwei)
    } else if (algo == "11") {
      // lbfgs
      val lralg = new LogisticRegressionWithLBFGS()
      lralg.optimizer
        .setNumIterations(cmdMap("iternum").toInt)
        .setRegParam(cmdMap("regp").toDouble)
        .setConvergenceTol(cmdMap("tolerance").toDouble)
        .setNumCorrections(cmdMap("correction").toInt)
        .setUpdater(updater.asInstanceOf[Updater])
      lralg.run(trainData,initwei)
    } else if (algo == "12") {
      // my lbfgs
      val lralg = new LR_LBFGS()
      lralg.optimizer
        .setNumIterations(cmdMap("iternum").toInt)
        .setRegParam(cmdMap("regp").toDouble)
        .setConvergenceTol(cmdMap("tolerance").toDouble)
        .setNumCorrections(cmdMap("correction").toInt)
        .setUpdater(updater.asInstanceOf[Updater])
      lralg.run(trainData,initwei)
    } else if (algo == "13") {
      // my owlqn
      val lralg = new LR_OWLQN(cmdMap("scale")=="1")
      lralg.optimizer
        .setNumIterations(cmdMap("iternum").toInt)
        .setRegParam(cmdMap("regp").toDouble)
        .setConvergenceTol(cmdMap("tolerance").toDouble)
        .setNumCorrections(cmdMap("correction").toInt)
        .setUpdater(updater.asInstanceOf[Updater])
      lralg.run(trainData,initwei)
    }

    val model:LogisticRegressionModel = modeltmp.asInstanceOf[LogisticRegressionModel]
    // save weight info
    mutil.saveVector(fs, model.weights, outputPath+"/weight")
    analysisWeight(model.weights)


    // Predict
    // Clears the threshold so that predict will output raw prediction scores.
    model.clearThreshold()
    analyseLog.append("numFeatures of model = "+model.numFeatures+"\n")
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
