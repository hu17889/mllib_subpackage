/**
 * Created by 58 on 2015/4/11.
 */
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.SparkContext
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.streaming.dstream



import scala.collection.mutable.{StringBuilder, Map}
import scala.Option
import scala.util.Random
import collection.mutable.ArrayBuffer

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.commons.cli._

import breeze.linalg.{DenseVector,SparseVector}


import common.mutil

case class Status(w:ArrayBuffer[Double]=ArrayBuffer[Double](), z:ArrayBuffer[Double]=ArrayBuffer[Double](), n:ArrayBuffer[Double]=ArrayBuffer[Double]()) extends Serializable {
  def emptyStatus(n:Int)={
    new Status(new ArrayBuffer[Double](n),new ArrayBuffer[Double](n),new ArrayBuffer[Double](n))
  }

}

class ftrl(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) with  Serializable  {



  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data path")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; 21. ftrl")
    options.addOption("ts", "timeslice", true, "time slice")
    options.addOption("pa", "pa", true, "parameter pa for learning rate")
    options.addOption("pb", "pb", true, "parameter pb for learning rate")
    options.addOption("pr1", "pr1", true, "parameter pr1 for sparsing")
    options.addOption("pr2", "pr2", true, "parameter pr2 for sparsing")
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
    cmdMap += ("ts"->cl.getOptionValue("ts"))
    cmdMap += ("pa"->cl.getOptionValue("pa"))
    cmdMap += ("pb"->cl.getOptionValue("pb"))
    cmdMap += ("pr1"->cl.getOptionValue("pr1"))
    cmdMap += ("pr2"->cl.getOptionValue("pr2"))
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



  def deal() {
    val cmdP = this.cmdMap
    val ssc = new StreamingContext(sc, Seconds(cmdP("ts").toInt))
    val lines = ssc.textFileStream(cmdP("traindata"))

    val outputPath = cmdP("dstdata")
    val algo = cmdP("algo")
    println("input params+ : " + cmdP.toString() + "\n")
    analyseLog.append("input params+ : " + cmdP.toString() + "\n")

    // to (Int,Array)
    val linesFormat = lines.map{case line => {
      val parts = line.split(" ")
      val label = parts(0).toInt
      val x = parts.drop(1).map{case p=>{
        val a = p.split(":")
        (a(0).toInt,a(1).toDouble)
      }}
      (1,(label,x))
    }}
    /*
    linesFormat.print()
    ssc.checkpoint("/secure/search/sort/dir/tmp/sparkcheckpoint")
    ssc.start()
    ssc.awaitTermination()
    */

    val literateWindow1 = (currValues: Seq[(Int,Array[(Int,Double)])], prevValueState: Option[Int]) => {
      val prev = prevValueState.getOrElse(0)
      val dataArray = currValues.toArray
      var cur = prev
      if(dataArray.length>0) {
        cur = currValues.toArray.map(_._2).flatMap(x=>x).map(_._1).max
        //cur=200
      }
      //val cur = currValues.toArray.map(_._2).flatMap(x=>x).head._1//map(_._1).max
      Some(cur)
    }

    val literateWindow = (currValues: Seq[(Int,Array[(Int,Double)])], prevValueState: Option[Status]) => {
      // get max length of weight
      //println("*****************************num of records in 10 seconds + : " + currValues.length + "\n")
      val dataArray = currValues.toArray
      if(dataArray.length<0) {
        val num = dataArray.map(_._2).flatMap(x => x).map(_._1).max
        val prev = prevValueState.getOrElse(Status().emptyStatus(num))
        val w: ArrayBuffer[Double] = if (num > prev.w.length) prev.w.padTo(num, 0).asInstanceOf[ArrayBuffer[Double]] else prev.w
        val z: ArrayBuffer[Double] = if (num > prev.z.length) prev.z.padTo(num, 0).asInstanceOf[ArrayBuffer[Double]] else prev.z
        val n: ArrayBuffer[Double] = if (num > prev.n.length) prev.n.padTo(num, 0).asInstanceOf[ArrayBuffer[Double]] else prev.n

        val pa = cmdP("pa").toDouble
        val pb = cmdP("pb").toDouble
        val pr1 = cmdP("pr1").toDouble
        val pr2 = cmdP("pr2").toDouble

        dataArray.foreach { case (y, x) => {
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
          }
          }.sum

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
        }
        //Some(currValues.length)

        // output

        val nstatus = new Status(w, z, n)

        //sc.parallelize(w.toList).saveAsTextFile(outputPath+"/weight"+System.currentTimeMillis())
        //println("************************"+nstatus.toString)
        //saveArray(fs,w.toArray,outputPath+"/weight")
        Some(nstatus)
      } else {
        prevValueState
      }

    }

    //linesFormat.updateStateByKey(literateWindow1).saveAsTextFiles(outputPath+"/weight")
    //linesFormat.updateStateByKey(literateWindow).flatMap(_._2.w.zipWithIndex).print()
    linesFormat.updateStateByKey(literateWindow).print()

    ssc.checkpoint("/secure/search/sort/dir/tmp/sparkcheckpoint")
    ssc.start()
    ssc.awaitTermination()

    // analyse output
    val analysePath = new Path(outputPath+"/analyse")
    val outEval2 = fs.create(analysePath,true)
    outEval2.writeBytes(analyseLog.toString())
    outEval2.close()

  }
}
