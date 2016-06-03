package common

import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.mllib.linalg.{Vector => LinalgVector, Vectors, DenseVector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.{ArrayBuffer, Map}
import scala.util.Random
import util.control.Breaks._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}



/**
 * Created by 58 on 2015/6/3.
 */
object mutil {

  def vectorfromBreeze(breezeVector: BV[Double]): LinalgVector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  def vectorToBreeze(v:LinalgVector): BV[Double] = {
    new BDV(v.toArray)
  }

  def formatDuring(mss:Long):String = {
    val days = mss / (1000 * 60 * 60 * 24)
    val hours = (mss % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)
    val minutes = (mss % (1000 * 60 * 60)) / (1000 * 60)
    val seconds = (mss % (1000 * 60)) / 1000
    days + " days " + hours + " hours " + minutes + " minutes " + seconds + " seconds "
  }

  // 记录运行时信息
  def runtimeInfo(): Map[String,String] = {
    val cmdMap = Map[String, String]()
    val rt:Runtime = Runtime.getRuntime()
    cmdMap += ("freeMemory" -> rt.freeMemory().toString)
    cmdMap += ("totalMemory" -> rt.totalMemory().toString)
    return cmdMap
  }

  // 保存Vector结构数据到hdfs中
  def saveVector(fs: FileSystem, input:LinalgVector, path:String): Unit = {
    val weightPath = new Path(path)
    val outEval1 = fs.create(weightPath,true)
    val weightLog = new StringBuilder()
    //input.toArray.map{ case w:Double => outEval1.writeUTF(w.toString + "\n")}
    input.toArray.map{ case w:Double => weightLog.append(w.toString + "\n")}
    outEval1.writeBytes(weightLog.toString())
    outEval1.close()
  }

  // 保存Array结构数据到hdfs中
  def saveArray(fs: FileSystem, input:Array[_], path:String): Unit = {
    val weightPath = new Path(path)
    val outEval1 = fs.create(weightPath,true)
    val weightLog = new StringBuilder()
    input.map {
      case w:Double => outEval1.writeUTF(w.toString + "\n")//weightLog.append(w.toString + "\n")
      case w:(_,_) => outEval1.writeUTF(w.toString + "\n")//weightLog.append(w.toString + "\n")
    }
    //outEval1.writeBytes(weightLog.toString())
    //outEval1.writeUTF(weightLog.toString())
    outEval1.close()
  }

  // 初始化权重
  def initWeight(input: RDD[LabeledPoint], wtype:String): LinalgVector = {
    val numFeatures = input.map(_.features.size).first()
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

  // Top n 选择
  def topn(input:Array[(String,Float)],n:Int,top_type:String): Array[(String,Float)] = {
    val ret = new ArrayBuffer[(String,Float)]()
    for(i <- 0 until input.length) {
      if(ret.length==0) {
        ret.append(input(i))
      } else {
        val t = ret.length
        breakable {
          for(j <- 0 until ret.length) {
            if ((top_type=="desc"&&input(i)._2 > ret(j)._2)||(top_type=="asc"&&input(i)._2 < ret(j)._2)) {
              if (ret.length >= n) ret.trimEnd(1)
              ret.insert(j, input(i))
              break
            }
          }
        }
        if(t==ret.length&&ret.length<n) ret.append(input(i))

      }
    }
    ret.toArray
  }
}
