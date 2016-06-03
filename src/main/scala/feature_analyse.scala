/**
 * Created by 58 on 2015/4/11.
 */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD


import scala.collection.mutable.{ArrayBuffer, StringBuilder, Map}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.commons.cli._

import common.mutil



class feature_analyse(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) with  Serializable  {
  var weightNum = 0



  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data path")
    options.addOption("te", "testdata", true, "test data path")
    options.addOption("fn", "feaname", true, "feature name")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; 21. ftrl")
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
    cmdMap += ("feaname"->cl.getOptionValue("feaname"))
    cmdMap += ("dstdata"->cl.getOptionValue("dstdata"))
    return  cmdMap
  }


  def prepareInput(lines:RDD[String],names:RDD[(Int,String)],analyseLog:StringBuilder,outputPath:String,label:String): RDD[(Int,Array[(Int,Double)])] = {
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


    linesFormat.flatMap{case (y,x)=>
        x.map{x=>(x._1,y)}
    }.groupByKey().map{case (x,y)=>
      val z0 = y.filter(_==0).toArray.length
      val z1 = y.filter(_==1).toArray.length
      (x,(z0,z1,z0+z1,z1.toDouble/(z0+1).toDouble))
    }.leftOuterJoin(names).map{case(x,y)=>
      val y1 = y._1
      (x,y._2.toString,y1._1,y1._2,y1._3,y1._4)
    }.sortBy(_._5,false).saveAsTextFile(outputPath+"/"+label+"_sample_distribution")


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
    val namelines = sc.textFile(cmdP("feaname")).map{case x=>
      val y = x.replaceAll("\t|\n","").split(" ")
      if(y.length==2) (y(1).toInt,y(0))
    }.asInstanceOf[RDD[(Int,String)]]
    val outputPath = cmdP("dstdata")
    val algo = cmdP("algo")

    // to (Int,Array)
    val trainLinesFormat = prepareInput(trainlines,namelines,analyseLog,outputPath,"train").cache()
    val testLinesFormat = prepareInput(testlines,namelines,analyseLog,outputPath,"test").cache()



    // analyse output
    val analysePath = new Path(outputPath+"/analyse")
    val outEval2 = fs.create(analysePath,true)
    outEval2.writeBytes(analyseLog.toString())
    outEval2.close()

  }
}
