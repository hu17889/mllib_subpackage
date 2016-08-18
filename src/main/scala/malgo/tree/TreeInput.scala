package org.apache.spark.mllib.tree

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

object TreeInput {
  def makeLambdaLabeledPoint( sc: SparkContext,inputpath:String): RDD[LambdaLabeledPoint] = {
    val parsed = sc.textFile(inputpath)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
      var items = line.split(' ')
      val label = items.head.toDouble
      items = items.tail
      val qid = items.head.split(":")(1)
      items = items.tail
      val pos = items.head.split(":")(1)
      val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
        val indexAndValue = item.split(':')
        val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
      val value = indexAndValue(1).toDouble
        (index, value)
      }.unzip

      // check if indices are one-based and in ascending order
      var previous = -1
      var i = 0
      val indicesLength = indices.length
      while (i < indicesLength) {
        val current = indices(i)
        require(current > previous, "indices should be one-based and in ascending order" )
        previous = current
        i += 1
      }

      (label,qid, pos, indices.toArray, values.toArray)
    }

    // Determine number of features.
    parsed.persist(StorageLevel.MEMORY_ONLY)
    val d:Int = parsed.map { case (label, qid, pos, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1

    parsed.map { case (label, qid, pos, indices, values) =>
      LambdaLabeledPoint(label,qid,pos.toDouble.toInt, Vectors.sparse(d, indices, values),0,0)
    }
  }

  // Array[(label,pred)]
  // return ndcg.sum
  private def NDCG(input:Array[(Double,Double)]): Double = {
    // ((label,pred),index)
    val dcg = input.zipWithIndex.scanLeft(0.0){case (a,b)=>
      val index = b._2
      val label = b._1._1
      a+(math.pow(2,label)-1)/math.log(2+index)
    }.drop(1)
    val maxdcg = input.sortBy(-_._1).zipWithIndex.scanLeft(0.0){case (a,b)=>
      val index = b._2
      val label = b._1._1
      a+(math.pow(2,label)-1)/math.log(2+index)
    }.drop(1)
    val re = dcg.zip(maxdcg).map{case (a,b)=>a/b}
    re.sum/re.length
  }

  // Array[((label,pred),pos)]
  private def changePosNDCG(input:Array[((Double,Double),Int)],posx:Int,posy:Int): Double = {
    val inputMap = input.map(x=>(x._2,x._1)).toMap
    val temp = inputMap(posx)
    val input2 = inputMap.updated(posx,inputMap(posy)).updated(posy,temp)
    //inputMap.updated(posx,inputMap(posy))
    //inputMap.updated(posy,temp)

    NDCG(input2.toArray.map(x=>x._2))
  }

  def firstCal(input:RDD[LambdaLabeledPoint]): RDD[LambdaLabeledPoint] = {
    calre(input.map{x=>
      x.lamda = 0.0
      x.w = 0.0
      ((x.qid,x.pos),(x,(-x.pos.toDouble,0.0)))
    })
  }

  def cal(input:RDD[LambdaLabeledPoint],predError: RDD[((String,Int),(Double, Double))]): RDD[LambdaLabeledPoint] = {
    calre(input.map{case x=>((x.qid,x.pos),x)}.join(predError))
  }

  // ((qid,pos),(LambdaLabeledPoint,(pred,error)))
  private def calre(input: RDD[((String,Int),(LambdaLabeledPoint,(Double,Double)))]): RDD[LambdaLabeledPoint] = {
    println("begin to calre")
    input.map{case (key,(point,prederr))=>(point.qid,(point,prederr))}.groupByKey.filter(_._2.size > 1).flatMap{case (_,data) =>{
      val ds = data.toArray
      val positives = ds.filter(_._1.label>=1)
      val negtives = ds.filter(_._1.label<1)

      if(positives.length<1 || (negtives.length <1)) {
        Array().toIterable
      } else {
        val pointPairs = positives.flatMap{case x=>{
          val pairs = ArrayBuffer[((LambdaLabeledPoint,(Double,Double)),(LambdaLabeledPoint,(Double,Double)))]()
          negtives.foreach{case y=>
            pairs += ((x,y),(y,x))
          }
          pairs
        }}

        val ndcgInput = ds.map{case (point,(pred,error))=>(point.label,pred)}.sortBy(-_._2)
        // (label,pred) order by pred desc and cal ndcg
        val sorNDCG = NDCG(ndcgInput)
        val rhos = pointPairs.map{case (pointx,pointy)=>((pointx._1.pos,pointy._1.pos), (pointx._1.label, pointy._1.label, 1/(1+math.exp(pointx._2._1-pointy._2._1))))}.toMap
        val ndcgPosInput = ds.map{case (point,(pred,error))=>((point.label,pred),point.pos)}.sortBy(-_._1._2)
        val dNDCGs = pointPairs.map{case (pointx,pointy)=>((pointx._1.pos,pointy._1.pos), math.abs(changePosNDCG(ndcgPosInput,pointx._1.pos,pointy._1.pos)-sorNDCG))}
        // key=(posx,posy)
        //val lambdaIJ = rhos.zip(dNDCGs).map{case ((key1,rho),(key2,ndcg))=>((key1,rho*ndcg))}
        val lambdaIJ = dNDCGs.map{case (key,ndcg)=>
          val data = rhos.apply(key)
          val rho = data._3
          // posx,posy,labelx,labely
          ((key._1,key._2,data._1,data._2),rho*ndcg)
        }
        val lambda = lambdaIJ.groupBy(_._1._1).map{case (keyi,elems)=>{
          val lambdai = elems.filter{case ((posx,posy,labelx,labely),v)=>labelx>labely}.map(_._2).sum - elems.filter{case ((posx,posy,labelx,labely),v)=>labelx<labely}.map(_._2).sum
          (keyi,lambdai)
        }}


        val lambdaIJP = dNDCGs.map{case (key,ndcg)=>
          val data = rhos.apply(key)
          val rho = data._3
          ((key._1,key._2,data._1,data._2),rho*(1-rho)*ndcg)
        }
        //val lambdaIJP = rhos.zip(dNDCGs).map{case ((key1,rho),(key2,ndcg))=>((key1,rho*(1-rho)*ndcg))}
        val w = lambdaIJP.groupBy(_._1._1).map{case (keyi,elems)=>{
          val wi = elems.map(_._2).sum
          (keyi,wi)
        }}
        val pointMap = ds.map{case (point,(pred,error))=>(point.pos,point)}.toMap
        println("pointMap:"+pointMap)
          val result = pointMap.map{case (pos,point)=>{
            println("input feature:"+point.toString)
              point.lamda = lambda.apply(pos)
              point.w = w.apply(pos)
              new LambdaLabeledPoint(point.lamda,point.qid,point.pos,point.features,point.lamda,point.w)


          }

      }
        result
      }

    }}.filter(!_.qid.equals("error"))
  }


}


case class LambdaLabeledPoint(label:Double,qid:String,pos:Int,features:Vector, var lamda:Double=0.0, var w:Double=0.0) {

  override def toString: String = {
    s"($label,$qid,$pos,$lamda,$w)"
  }
}