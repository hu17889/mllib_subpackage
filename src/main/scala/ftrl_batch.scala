/**
 * Created by 58 on 2015/4/11.
 */

import java.util.Calendar

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD


import scala.collection.mutable.{ArrayBuffer, StringBuilder, Map}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.commons.cli._

import common.mutil



class ftrl_batch(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) with  Serializable  {
  var weightNum = 0



  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data path")
    options.addOption("te", "testdata", true, "test data path")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; 21. ftrl")
    options.addOption("pa", "pa", true, "parameter pa for learning rate")
    options.addOption("pb", "pb", true, "parameter pb for learning rate")
    options.addOption("pr1", "pr1", true, "parameter pr1 for sparsing")
    options.addOption("pr2", "pr2", true, "parameter pr2 for sparsing")
    options.addOption("ps", "parts", true, "number of partitions")
    val cl = parser.parse( options, args )

    if( cl.hasOption('h') ) {
      val f:HelpFormatter = new HelpFormatter()
      f.printHelp("OptionsTip", options)
    }

    val cmdMap = Map[String, String]()
    val algo = cl.getOptionValue("algo")
    cmdMap += ("algo" -> algo)
    cmdMap += ("traindata"->cl.getOptionValue("traindata"))
    cmdMap += ("dstdata"->cl.getOptionValue("dstdata"))
    cmdMap += ("testdata"->cl.getOptionValue("testdata"))
    cmdMap += ("pa"->cl.getOptionValue("pa"))
    cmdMap += ("pb"->cl.getOptionValue("pb"))
    cmdMap += ("pr1"->cl.getOptionValue("pr1"))
    cmdMap += ("pr2"->cl.getOptionValue("pr2"))
    cmdMap += ("parts"->cl.getOptionValue("parts"))
    return  cmdMap
  }


  /*
        def calWei(pa:Double,pb:Double,pr1:Double,pr2:Double,ni:Double,zi:Double):Double = {
          val x = if(math.abs(zi)<=pr1) {
            0
          } else {
            (math.signum(zi)*pr1 - zi) / ((pb+math.sqrt(ni))/pa + pr2)
          }
          x.toDouble
        }*/

  def analysisWeight(weights:Array[Double]): Unit = {
    val nonzeroWeiLen = weights.filter(_!=0).length
    val totalWeiLen = weights.length
    analyseLog.append("---------weight analyse-----------\n")
    analyseLog.append("nonzero weight length = " + nonzeroWeiLen +"\n")
    analyseLog.append("total weight length = " + totalWeiLen +"\n")
    analyseLog.append("nonzero weight length/total weight length = " + nonzeroWeiLen.toFloat/totalWeiLen.toFloat +"\n")
    val positiveWeights = weights.filter(_!=0).map(_.abs)
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

  def predictData(testData:RDD[(Int,Array[(Int,Double)])], weight:Array[Double], outputPath:String, analyseLog:StringBuilder, datalabel:String): Unit = {

    val scoreAndLabels = testData.map { point =>
      val x = point._2
      val margin = -1.0 * x.map { case (index, value) => {
        val w = weight(index)
        value * w
      } }.sum
      val score = 1/(1+math.exp(margin))
      (score, point._1.toDouble)
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

  def prepareInput(lines:RDD[String],analyseLog:StringBuilder,outputPath:String,label:String): RDD[(Int,Array[(Int,Double)])] = {
    val linesFormat = lines.map{case line => {
      val parts = line.split(" ")
      val label = parts(0).toInt
      val x = parts.drop(1).map{case p=>{
        val a = p.split(":")
        // Convert 1-based indices to 0-based
        (a(0).toInt-1,a(1).toDouble)
      }}
      (label,x)
    }}.cache()

    /*
    linesFormat.flatMap{case (y,x)=>
        x.map{x=>(x._1,y)}
    }.groupByKey().map{case (x,y)=>
      val z0 = y.filter(_==0).toArray.length
      val z1 = y.filter(_==1).toArray.length
      (x,z0,z1,z0+z1,z1.toDouble/(z0+1).toDouble)
    }.sortBy(_._4,false).saveAsTextFile(outputPath+"/"+label+"_sample_distribution")
    */

    val weightNum = linesFormat.map(_._2).flatMap(x => x).map(_._1).max+1
    this.weightNum = math.max(weightNum,this.weightNum)
    println("**************"+label+" weight length : " + weightNum.toString() + "\n")
    analyseLog.append(label+" weight length : " + weightNum.toString() + "\n")
    val sampleNum = linesFormat.count()
    println("**************"+label+" sample number : " + sampleNum.toString() + "\n")
    analyseLog.append(label+" sample number : " + sampleNum.toString() + "\n")

    linesFormat
  }

  def deal() {
    val cmdP = this.cmdMap
    println("**************input params : " + cmdP.toString() + "\n")
    analyseLog.append("input params : " + cmdP.toString() + "\n")

    val trainlines = sc.textFile(cmdP("traindata"))
    val testlines = sc.textFile(cmdP("testdata"))
    val outputPath = cmdP("dstdata")
    val algo = cmdP("algo")

    // to (Int,Array)
    val trainLinesFormat = prepareInput(trainlines,analyseLog,outputPath,"train").cache()
    val testLinesFormat = prepareInput(testlines,analyseLog,outputPath,"test").cache()

    val weightLength = this.weightNum

    val myfunc = (iter: Iterator[(Int,Array[(Int,Double)])]) => {
      val w = new Array[Double](weightLength)
      val z = new Array[Double](weightLength)
      val n = new Array[Double](weightLength)
      val pa = cmdP("pa").toDouble
      val pb = cmdP("pb").toDouble
      val pr1 = cmdP("pr1").toDouble
      val pr2 = cmdP("pr2").toDouble
      /*
      val pa = 1.0
      val pb = 1.0
      val pr1 = 1.0
      val pr2 = 0.05
      */

      while (iter.hasNext)
      {
        val cur = iter.next
        val y = cur._1
        val x = cur._2
        // cal weight
        x.foreach(txi => {
          val i = txi._1
          //w(i) = calWei(pa,pb,pr1,pr2,n(i),z(i))
          w(i) = if (math.abs(z(i)) <= pr1) {
            0
          } else {
            (math.signum(z(i)) * pr1 - z(i)) / ((pb + math.sqrt(n(i))) / pa + pr2)
          }
        })

        // cal pt
        //val margin = -1.0 * ( SparseVector(x) dot DenseVector(w) )
        val margin = -1.0 * x.map { case (index, value) => {
          value * w(index)
        } }.sum

        // cal z and n
        x.foreach(txi => {
          val i = txi._1
          val xi = txi._2
          val gi = ((1.0 / (1.0 + math.exp(margin))) - y) * xi
          val sigma = (math.sqrt(n(i) + math.pow(gi, 2)) - math.sqrt(n(i))) / pa
          z(i) = z(i) + gi - sigma * w(i)
          n(i) = n(i) + math.pow(gi, 2)
        })
      }

      w.zipWithIndex.map{case (x,y)=>(y,x)}.iterator
    }



    val retrainLinesFormat = trainLinesFormat.coalesce(cmdP("parts").toInt,true)
    analyseLog.append("number of train data partitions = "+retrainLinesFormat.partitions.length+"\n")

    val weight = retrainLinesFormat.mapPartitions(myfunc).groupByKey().map{case (key, values) =>
      val num = values.toArray.length
      val all = values.toArray.sum
      val mean = all/num
      (key,f"$mean%.6f".toDouble)
    }.cache()
    weight.saveAsTextFile(outputPath+"/weight1")

    val weightDriven = new ArrayBuffer[Double]()
    val weightMap = weight.collect().toMap
    var i=0
    while(i<=weightLength){
      val elem = weightMap.applyOrElse(i,(x:Int)=>0.toDouble)
      weightDriven += elem
      i+=1
    }
    mutil.saveArray(fs,weightDriven.toArray,outputPath+"/weight")
    analysisWeight(weightDriven.toArray)

    analyseLog.append("---------predict train data-----------\n")
    predictData(trainLinesFormat,weightDriven.toArray,outputPath,analyseLog,"train")
    analyseLog.append("---------predict test data-----------\n")
    predictData(testLinesFormat,weightDriven.toArray,outputPath,analyseLog,"test")



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
