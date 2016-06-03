/**
 * Created by 58 on 2015/4/11.
 */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD


import scala.collection.mutable
import scala.collection.mutable.{StringBuilder, Map}
import scala.Option
import scala.util.Random
import collection.mutable.ArrayBuffer

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.commons.cli._

import breeze.linalg.{DenseVector,SparseVector}


import common.mutil

class relative(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) {

  val executerNum = sc.getConf.get("spark.executor.instances").toInt
  val outputPath = cmdMap("dstdata")


  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data dir path")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; 31. relative")
    options.addOption("dt", "dtype", true, "distance type. 1 cos 2 cospure")
    options.addOption("af", "allfloat", true, "all data is float?. true: data without type and data's segments number is three; false: data has four segments and has type(float, bool), cal for each type is different.")

    options.addOption("tt", "toptype", true, "1 (a,b:sort,d) 2 (a,b_c:sort,d)")
    options.addOption("ts", "topn_sep", true, "Number of parts. Sort by parts and merge the result of each part")
    options.addOption("td", "topn_dir", true, "get large top(desc) or small top(asc)")
    options.addOption("tn", "topn", true, "0 all k topk")

    options.addOption("pn", "pre_norm", true, "Normalization true false")
    options.addOption("ps", "pre_sep", true, "Seperate Data to parts by user frequence. Function opened when value bigger than 0.")
    options.addOption("pska", "pre_sep_keys_acc", true, "segment name for seperatting, like 116,120,121")
    options.addOption("pskm", "pre_sep_keys_map", true, "segment name for seperatting, like 116,120,121")
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
    cmdMap += ("dtype"->cl.getOptionValue("dtype","1"))
    cmdMap += ("allfloat"->cl.getOptionValue("allfloat","true"))

    cmdMap += ("toptype"->cl.getOptionValue("toptype", "1"))
    cmdMap += ("topn"->cl.getOptionValue("topn", "0"))
    cmdMap += ("topn_sep"->cl.getOptionValue("topn_sep", "10"))
    cmdMap += ("topn_dir"->cl.getOptionValue("topn_dir", "desc"))

    cmdMap += ("pre_norm"->cl.getOptionValue("pre_norm", "false"))
    cmdMap += ("pre_sep"->cl.getOptionValue("pre_sep", "0"))
    cmdMap += ("pre_sep_keys_acc"->cl.getOptionValue("pre_sep_keys_acc", ""))
    cmdMap += ("pre_sep_keys_map"->cl.getOptionValue("pre_sep_keys_map", ""))

    return  cmdMap
  }

  def euro(lines:RDD[String],allfloat:Boolean): RDD[String] = {
    println("----------------------------------------- start relative -----------------------------------------")
    /*
        ret0.saveAsTextFile(outputPath+"/ret0")
        val ret =  ret0
        */

    // merge sum(score1*score2) number(user with float) number(user with bool)
    // value (item1##item2,score1*score2##float)
    // value (item1##item2,bool)
    val seqop = (merge:(Float,Int,Int),value:Float) => {
      var sumScore = merge._1
      var numFloat = merge._2
      var numBool = merge._3
      if (value == -1) {
        numBool += 1
      } else {
        sumScore += value
        numFloat += 1
      }
      (sumScore, numFloat, numBool)
    }

    val combop = (m1:(Float,Int,Int),m2:(Float,Int,Int)) => {
      (m1._1+m2._1,m1._2+m2._2,m1._3+m2._3)
    }

    val ret = lines.filter{ case x=>{
      val t = x.split("\t")
      val num = t.length
      if(num==4&&t(3)=="bool") {
        true
      } else if(num==4&&t(3)=="float") {
        true
      } else if(num==3) {
        true
      } else {
        false
      }
    }}.map{case x:String=> {
      val parts = x.split("\t")
      if(allfloat) {
        // (user,item##score)
        (parts(0),parts(1)+"##"+parts(2))
      } else {
        // (user,item##score##type)
        (parts(0),parts(1)+"##"+parts(2)+"##"+parts(3))
      }
    }}.groupByKey().flatMap{case (fromid,idsWithValue)=>{
      val parts = idsWithValue.toArray
      val ret = new ArrayBuffer[(String,Float)]
      for(i <- 0 until parts.length) {
        for(j <- i until parts.length){
          val v1 = parts(i).split("##")
          val v2 = parts(j).split("##")

          if(allfloat) {
            // (item1##item2,(score1-score2)^2##float)
            ret.append((v1(0) + "##" + v2(0), math.pow(v1(1).toFloat - v2(1).toFloat,2).toFloat))
            ret.append((v2(0) + "##" + v1(0), math.pow(v1(1).toFloat - v2(1).toFloat,2).toFloat))
          } else {
            if(v1(2)==v2(2)&&v1(2)=="float") {
              // (item1##item2,(score1-score2)^2##float)
              ret.append((v1(0) + "##" + v2(0), math.pow(v1(1).toFloat - v2(1).toFloat,2).toFloat))
              ret.append((v2(0) + "##" + v1(0), math.pow(v1(1).toFloat - v2(1).toFloat,2).toFloat))
            } else if(v1(2)==v2(2)&&v1(1)==v2(1)&&v1(2)=="bool") {
              // (item1##item2,bool)
              ret.append((v1(0) + "##" + v2(0), (-1).toFloat))
              ret.append((v2(0) + "##" + v1(0), (-1).toFloat))
            }
          }
        }
      }
      ret
    }}.aggregateByKey[(Float,Int,Int)](0.toFloat,0,0)(seqop,combop).map{ case (itemToItem,(sumScore:Float,numFloat:Int,numBool:Int)) => {
      // bool type's score replace with avg score
      val avgScore = if(sumScore==0||numFloat==0) 0 else sumScore/numFloat
      var retScore = sumScore
      var i=0
      while(i<numBool) {
        retScore += avgScore
        i+=1
      }
      val parts = itemToItem.split("##")
      val p = math.sqrt(retScore).toFloat
      parts(0)+"\t"+parts(1)+"\t"+f"$p%.8f"
    }}

    ret
  }

  // nouse
  def cospure(lines:RDD[String]): RDD[String] = {
    println("----------------------------------------- start relative -----------------------------------------")

    val ret = lines.filter(_.split('\t')(2).toFloat!=0).map{case x:String=> {
      val parts = x.split("\t")
        // (user,item##score)
      (parts(0),parts(1)+"##"+parts(2))
    }}.groupByKey().flatMap{case (fromid,idsWithValue)=>{
      val parts = idsWithValue.toArray
      val ret = new ArrayBuffer[(String,Float)]
      for(i <- 0 until parts.length) {
        for(j <- i until parts.length){
          val v1 = parts(i).split("##")
          val v2 = parts(j).split("##")
          // (item1##item2,score1*score2)
          ret.append((v1(0) + "##" + v2(0), v1(1).toFloat * v2(1).toFloat))
          ret.append((v2(0) + "##" + v1(0), v1(1).toFloat * v2(1).toFloat))
        }
      }
      ret
    }}.reduceByKey(_+_).map{case(key:String,value:Float)=>{
      val parts = key.split("##")
      // (item1,item2##score)
      (parts(0),parts(1)+"##"+value.toString)
    }}.groupByKey().flatMap{case(key:String,values)=>{
      val vs = values.toArray
      val ret = new ArrayBuffer[(String,String,Float)]
      var d:Float = 0
      for(i <- 0 until vs.length) {
        val parts = vs(i).split("##")
        if(key==parts(0)) d = parts(1).toFloat
        ret.append((key,parts(0),parts(1).toFloat))
      }
      val ret1 = new ArrayBuffer[(String,String)]
      if(d!=0) {
        for(i <- 0 until ret.length) {
          // (item2,item1##score)
          if(ret(i)._1==ret(i)._2) ret1.append((ret(i)._1,ret(i)._2+"##"+ret(i)._3.toString))
          else ret1.append((ret(i)._2,ret(i)._1+"##"+(ret(i)._3/d).toString))
        }
      }
      ret1
    }}.groupByKey().flatMap{case(key,values)=>{
      val vs = values.toArray
      val ret = new ArrayBuffer[(String,String,Float)]
      var d:Float = 0
      for(i <- 0 until vs.length) {
        val parts = vs(i).split("##")
        if(key==parts(0)) d = parts(1).toFloat
        ret.append((key,parts(0),parts(1).toFloat))
      }
      val ret1 = new ArrayBuffer[String]
      if(d!=0) {
        for(i <- 0 until ret.length) {
          // (item1 item2 score)
          if(ret(i)._1==ret(i)._2) ret1.append(ret(i)._1+"\t"+ret(i)._2+"\t"+ret(i)._3.toString)
          else {
            val p = ret(i)._3/d
            ret1.append(ret(i)._2+"\t"+ret(i)._1+"\t"+f"$p%.8f")
          }
        }
      }
      ret1
    }}

    ret
  }

  // nouse
  def cos_pre(lines:RDD[String],allfloat:Boolean): RDD[String] = {
    println("----------------------------------------- start relative -----------------------------------------")

    // merge sum(score1*score2) number(user with float) number(user with bool)
    // value (item1##item2,score1*score2##float)
    // value (item1##item2,bool)
    val seqop = (merge:(Float,Int,Int),value:String) => {
      val parts = value.split("##")
      var sumScore = merge._1
      var numFloat = merge._2
      var numBool = merge._3
      if(parts.length==2&&parts(1)=="float") {
        sumScore += parts(0).toFloat
        numFloat += 1
      } else if(parts.length==1&&parts(0)=="bool") {
        numBool += 1
      }
      (sumScore,numFloat,numBool)
    }

    val combop = (m1:(Float,Int,Int),m2:(Float,Int,Int)) => {
      (m1._1+m2._1,m1._2+m2._2,m1._3+m2._3)
    }
    /*
        ret0.saveAsTextFile(outputPath+"/ret0")
        val ret =  ret0
        */

    val ret = lines.filter{ case x=>{
      val t = x.split("\t")
      val num = t.length
      if(num==4&&t(3)=="bool") {
        true
      } else if(num==4&&t(3)=="float"&&t(2).toFloat!=0) {
        true
      } else if(num==3&&t(2).toFloat!=0) {
        true
      } else {
        false
      }
    }}.map{case x:String=> {
      val parts = x.split("\t")
      if(allfloat) {
        // (user,item##score)
        (parts(0),parts(1)+"##"+parts(2))
      } else {
        // (user,item##score##type)
        (parts(0),parts(1)+"##"+parts(2)+"##"+parts(3))
      }
    }}.groupByKey().flatMap{case (fromid,idsWithValue)=>{
      val parts = idsWithValue.toArray
      val ret = new ArrayBuffer[(String,String)]
      for(i <- 0 until parts.length) {
        for(j <- i until parts.length){
          val v1 = parts(i).split("##")
          val v2 = parts(j).split("##")
          if(allfloat) {
            // (item1##item2,score1*score2##float)
            ret.append((v1(0) + "##" + v2(0), v1(1).toFloat * v2(1).toFloat+"##float"))
            ret.append((v2(0) + "##" + v1(0), v1(1).toFloat * v2(1).toFloat+"##float"))
          } else {
            if(v1(2)==v2(2)&&v1(2)=="float") {
              // (item1##item2,score1*score2##float)
              ret.append((v1(0) + "##" + v2(0), v1(1).toFloat * v2(1).toFloat+"##"+v1(2)))
              ret.append((v2(0) + "##" + v1(0), v1(1).toFloat * v2(1).toFloat+"##"+v1(2)))
            } else if(v1(2)==v2(2)&&v1(1)==v2(1)&&v1(2)=="bool") {
              // (item1##item2,bool)
              ret.append((v1(0) + "##" + v2(0), v1(2)))
              ret.append((v2(0) + "##" + v1(0), v1(2)))
            }
          }
        }
      }
      ret
    }}.aggregateByKey[(Float,Int,Int)](0.toFloat,0,0)(seqop,combop).map{ case (itemToItem,(sumScore:Float,numFloat:Int,numBool:Int)) => {
      // bool type's score replace with avg score
      val avgScore = if(sumScore==0||numFloat==0) 0 else sumScore/numFloat
      var retScore = sumScore
      var i=0
      while(i<numBool) {
        retScore += avgScore
        i+=1
      }
      (itemToItem,retScore)
    }}.map{case(key:String,value:Float)=>{
      val parts = key.split("##")
      // (item1,item2##score)
      (parts(0),parts(1)+"##"+value.toString)
    }}.groupByKey().flatMap{case(key:String,values)=>{
      def output(a:String,b:String,score:Float,d:Float): (String,String) = {
        if(a==b) (a,b+"##"+score.toString)
        else (b,a+"##"+(score/d).toString)
      }

      val vs = values.toArray
      val ret = new ArrayBuffer[(String,String,Float)]
      val ret1 = new ArrayBuffer[(String,String)]
      var d:Float = 0
      for(i <- 0 until vs.length) {
        val parts = vs(i).split("##")
        if(key==parts(0)) {
          d = math.sqrt(parts(1).toDouble).toFloat
          for(i <- 0 until ret.length) {
            // (item2,item1##score)
            if(d!=0) {
              ret1.append(output(ret(i)._1, ret(i)._2, ret(i)._3, d))
            }
          }
          ret.clear()
        }
        if(d==0) ret.append((key,parts(0),parts(1).toFloat))
        else {
          ret1.append(output(key,parts(0),parts(1).toFloat,d))
        }
      }
      if(d!=0) {
        for(i <- 0 until ret.length) {
          // (item2,item1##score)
          ret1.append(output(ret(i)._1,ret(i)._2,ret(i)._3,d))
        }
      }
      ret1
    }}.groupByKey().flatMap{case(key,values)=>{
      def output(a:String,b:String,score:Float,d:Float): String = {
        if(a==b) a+"\t"+b+"\t"+score.toString
        else {
          val p = score/d
          b+"\t"+a+"\t"+(score/d).toString+"\t"+f"$p%.8f"
        }
      }

      val vs = values.toArray
      val ret = new ArrayBuffer[(String,String,Float)]
      val ret1 = new ArrayBuffer[String]
      var d:Float = 0
      for(i <- 0 until vs.length) {
        val parts = vs(i).split("##")
        if(key==parts(0)) {
          d = math.sqrt(parts(1).toDouble).toFloat
          for(i <- 0 until ret.length) {
            // (item1,item2##score)
            if(d!=0) {
              ret1.append(output(ret(i)._1, ret(i)._2, ret(i)._3, d))
            }
          }
          ret.clear()
        }
        if(d==0) ret.append((key,parts(0),parts(1).toFloat))
        else {
          ret1.append(output(key,parts(0),parts(1).toFloat,d))
        }
      }
      if(d!=0) {
        for(i <- 0 until ret.length) {
          // (item1 item2 score)
          ret1.append(output(ret(i)._1,ret(i)._2,ret(i)._3,d))
        }
      }
      ret1
    }}

    ret
  }

  def cos(lines:RDD[String],allfloat:Boolean): RDD[String] = {
    println("----------------------------------------- start relative -----------------------------------------")

    // merge sum(score1*score2) number(user with float) number(user with bool)
    // value (item1##item2,score1*score2##float)
    // value (item1##item2,bool)
    val seqop = (merge:(Float,Int,Int),value:Float) => {
      var sumScore = merge._1
      var numFloat = merge._2
      var numBool = merge._3
      if(value == -1) {
        numBool += 1
      } else {
        sumScore += value
        numFloat += 1
      }
      (sumScore,numFloat,numBool)
    }

    val combop = (m1:(Float,Int,Int),m2:(Float,Int,Int)) => {
      (m1._1+m2._1,m1._2+m2._2,m1._3+m2._3)
    }
    /*
        ret0.saveAsTextFile(outputPath+"/ret0")
        val ret =  ret0
        */

    val ret = lines.filter{ case x=>{
      val t = x.split("\t")
      val num = t.length
      if(num==4&&t(3)=="bool") {
        true
      } else if(num==4&&t(3)=="float"&&t(2).toFloat!=0) {
        true
      } else if(num==3&&t(2).toFloat!=0) {
        true
      } else {
        false
      }
    }}.map{case x:String=> {
      val parts = x.split("\t")
      if(allfloat) {
        // (user,item##score)
        (parts(0),parts(1)+"##"+parts(2))
      } else {
        // (user,item##score##type)
        (parts(0),parts(1)+"##"+parts(2)+"##"+parts(3))
      }
    }}.groupByKey().flatMap{case (fromid,idsWithValue)=>{
      val parts = idsWithValue.toArray
      val ret = new ArrayBuffer[(String,Float)]
      for(i <- 0 until parts.length) {
        for(j <- i until parts.length){
          val v1 = parts(i).split("##")
          val v2 = parts(j).split("##")
          if(allfloat) {
            // (item1##item2,score1*score2##float)
            ret.append((v1(0) + "##" + v2(0), v1(1).toFloat * v2(1).toFloat))
            ret.append((v2(0) + "##" + v1(0), v1(1).toFloat * v2(1).toFloat))
          } else {
            if(v1(2)==v2(2)&&v1(2)=="float") {
              // (item1##item2,score1*score2##float)
              ret.append((v1(0) + "##" + v2(0), v1(1).toFloat * v2(1).toFloat))
              ret.append((v2(0) + "##" + v1(0), v1(1).toFloat * v2(1).toFloat))
            } else if(v1(2)==v2(2)&&v1(1)==v2(1)&&v1(2)=="bool") {
              // (item1##item2,bool)
              ret.append((v1(0) + "##" + v2(0), -1.toFloat))
              ret.append((v2(0) + "##" + v1(0), -1.toFloat))
            }
          }
        }
      }
      ret
    }}.aggregateByKey[(Float,Int,Int)](0.toFloat,0,0)(seqop,combop).map{ case (itemToItem,(sumScore:Float,numFloat:Int,numBool:Int)) => {
      // bool type's score replace with avg score
      val avgScore = if(sumScore==0||numFloat==0) 0 else sumScore/numFloat
      var retScore = sumScore
      var i=0
      while(i<numBool) {
        retScore += avgScore
        i+=1
      }
      (itemToItem,retScore)
    }}.map{case(key:String,value:Float)=>{
      val parts = key.split("##")
      // (item1,item2##score)
      (parts(0),parts(1)+"##"+value.toString)
    }}.cache()

    val itemToitem = ret.filter{case (item1,item2Score)=>{
      val ps = item2Score.split("##")
      item1 == ps(0)
    }}.map{case (item1,item2Score) =>{
      val ps = item2Score.split("##")
      (item1,ps(1))
    }}

    val ret1 = ret.groupByKey().join(itemToitem).flatMap{case x=>{
      val leftItem = x._1
      val values = x._2._1.toArray
      val rscore = x._2._2.toFloat
      val ret = values.map{case value=>{
        val ps = value.split("##")
        val rightItem = ps(0)
        val lscore = ps(1).toFloat
        if(leftItem==rightItem) {
          (leftItem,leftItem+"##"+lscore.toString)
        } else {
          (rightItem,leftItem+"##"+(lscore/rscore).toString)
        }
      }}
      ret
    }}.groupByKey().join(itemToitem).flatMap{case x=>{
      val leftItem = x._1
      val values = x._2._1.toArray
      val rscore = x._2._2.toFloat
      val ret = values.map{case value=>{
        val ps = value.split("##")
        val rightItem = ps(0)
        val lscore = ps(1).toFloat
        if(leftItem==rightItem) {
          leftItem+"\t"+leftItem+"\t"+f"$lscore%.8f"
        } else {
          val p = lscore/rscore
          rightItem+"\t"+leftItem+"\t"+f"$p%.8f"
        }
      }}
      ret
    }}

    ret1
  }

  // Type bool will not be normalized
  // input (user,item,score,type)
  def normalizationWithType(lines:RDD[(String)]): RDD[(String)] = {
    val lineS = lines.map{ case x=> {
      val parts = x.split("\t")
      (parts(0),parts(1),parts(2),parts(3))
    } }

    val maxValue =  lineS.filter(_._4=="float").map{case x=>{
      (x._1,x._3.toFloat)
    }}.reduceByKey(math.max(_,_))

    val minValue =  lineS.filter(_._4=="float").map{case x=>{
      (x._1,x._3.toFloat)
    }}.reduceByKey(math.min(_,_))

    val maxminValue = maxValue.join(minValue)

    lineS.map{case x=>{
      (x._1,x._2+"##"+x._3+"##"+x._4)
    }}.leftOuterJoin(maxminValue).map{case x=>{
      val user = x._1
      val p = x._2._1.split("##")
      val item = p(0)
      val score = p(1)
      val datatype = p(2)
      val maxminvalue = x._2._2
      maxminvalue match {
        case Some(t) => {
          val maxvalue = t._1
          val minvalue = t._2
          val retscore =
            if(maxvalue-minvalue>0) (score.toFloat-minvalue)/(maxvalue-minvalue)
            else 0
          user+"\t"+item+"\t"+f"$retscore%.8f"+"\t"+datatype
        }
        case None => user+"\t"+item+"\t"+score+"\t"+datatype
      }
    }}
  }

  // input (user,item,score)
  def normalization(lines:RDD[(String)]): RDD[(String)] = {
    val lineS = lines.map{ case x=> {
      val parts = x.split("\t")
      (parts(0),parts(1),parts(2).toFloat)
    } }

    val maxValue =  lineS.map{case x=>{
      (x._1,x._3)
    }}.reduceByKey(math.max(_,_)).cache()

    val minValue =  lineS.map{case x=>{
      (x._1,x._3)
    }}.reduceByKey(math.min(_,_)).cache()

    lineS.map{case x=>{
      (x._1,x._2+"##"+x._3)
    }}.join(maxValue).join(minValue).map{case x=>{
      val user = x._1
      val p = x._2._1._1.split("##")
      val item = p(0)
      val score = p(1).toFloat
      val maxvalue = x._2._1._2
      val minvalue = x._2._2
      val retscore =
        if(maxvalue-minvalue>0) (score.toFloat-minvalue)/(maxvalue-minvalue)
        else 0
      user+"\t"+item+"\t"+f"$retscore%.8f"
    }}
  }


  def sortn(sorRelative:RDD[String]): RDD[String] = {
    val cmdP = this.cmdMap
    val sortdef = (key:String,values:Iterable[String]) => {
      val vs = values.toArray.map{case x=>{
        val parts = x.split("##")
        (parts(0), parts(1).toFloat)
      }}

      val ret = if(cmdP("toptype")=="1") {
        mutil.topn(vs, cmdP("topn").toInt,cmdP("topn_dir"))
      }else {
        vs.map{case (key, value)=>{
          val parts = key.split("_")
          // (item2, score) => (item21, item22, score)
          (parts(0),parts(1),value)
        }}.groupBy(_._1).toArray.flatMap{case (key, values)=>{
          val ret = values.map(x=>(x._1+"_"+x._2,x._3.toFloat))
          mutil.topn(ret, cmdP("topn").toInt,cmdP("topn_dir"))
        }}
      }

      val ret0 = ret.asInstanceOf[Array[(String,Float)]]

      val ret1 = new ArrayBuffer[(String,String)]()
      for(i <- 0 until ret0.length) {
        val keyseps = key.split("##")
        ret1.append((keyseps(0),ret0(i)._1+"##"+ret0(i)._2))
      }
      ret1

    }

    /*
        ret0.saveAsTextFile(outputPath+"/ret0")
        val ret =  ret0
        */
    val retSorted = sorRelative.map{case x => {
      val parts = x.split("\t")
      // (item1, item2##score)
      val sepnum = cmdP("topn_sep").toInt
      (parts(0)+"##"+sepnum,parts(1)+"##"+parts(2))
    }}.groupByKey().flatMap{case(key,values)=> {
      sortdef(key,values)
    }}.groupByKey().flatMap{case(key,values)=> {
      val p = key.split("##")
      sortdef(p(0),values)
    }}.map{case (x:String,y:String)=>{
      val p = y.split("##")
      x+"\t"+p(0)+"\t"+p(1)
    }}
    retSorted
  }

  // before tuning
  def sortn_pre(sorRelative:RDD[String]): RDD[String] = {
    val cmdP = this.cmdMap
    val retSorted = sorRelative.map{case x => {
      val parts = x.split("\t")
      // (item1, item2##score)
      (parts(0),parts(1)+"##"+parts(2))
    }}.groupByKey().flatMap{case(key,values)=> {
      val vs = values.toArray.map{case x=>{
        val parts = x.split("##")
        (parts(0), parts(1).toFloat)
      }}

      val ret = if(cmdP("toptype")=="1") {
        mutil.topn(vs, cmdP("topn").toInt,cmdP("topn_dir"))
      }else {
        vs.map{case (key, value)=>{
          val parts = key.split("_")
          // (item2, score) => (item21, item22, score)
          (parts(0),parts(1),value)
        }}.groupBy(_._1).toArray.flatMap{case (key, values)=>{
          val ret = values.map(x=>(x._1+"_"+x._2,x._3.toFloat))
          mutil.topn(ret, cmdP("topn").toInt,cmdP("topn_dir"))
        }}
      }

      val ret0 = ret.asInstanceOf[Array[(String,Float)]]

      val ret1 = new ArrayBuffer[String]()
      for(i <- 0 until ret0.length) {
        ret1.append(key+"\t"+ret0(i)._1+"\t"+ret0(i)._2)
      }
      ret1
    }}
    retSorted
  }

  // seperate input users to different parts by item's score.
  def seperate(lines:RDD[String]): RDD[String] = {
    val len = lines.map(x=>x.split("\t")(1)).distinct().count()
    val cmdP = this.cmdMap
    val all = cmdP("pre_sep_keys_acc")=="all"
    val choose1 = cmdP("pre_sep_keys_acc").split(",").zipWithIndex.toMap
    val choose2 = cmdP("pre_sep_keys_map").split(",").zipWithIndex.toMap
    val lineS = lines.map{ case x=> {
      val parts = x.split("\t")
      (parts(0),x)
      /*
      if(cmdP("allfloat").toBoolean)  (parts(0),parts(1),parts(2))
      else  (parts(0),parts(1),parts(2),parts(3))
      */
    } }.groupByKey().flatMap{case (key,line)=>{
      line.toList.map(_.split("\t")).sortWith{ case (a,b) =>{
        if(cmdP("allfloat").toBoolean)  a(2).toFloat>b(2).toFloat
        else {
          if(a(3)=="float") a(2).toFloat>b(2).toFloat
          else a(2)>b(2)
        }
      }}.zipWithIndex.map{case (parts,key)=>{
        // user item score type
        if(all||choose1.contains(parts(0))) parts(0) = parts(0)+"#"+(key*cmdP("pre_sep").toInt/len).toString
        else if(choose2.contains(parts(0))) parts(0) = parts(0)+"#"+parts(2)
        else parts(0) = parts(0)+"#0"
        parts.mkString("\t")
      }}
    }}
    lineS
  }

  def dealBySeperate(lines:RDD[String]): RDD[String] = {
    println("----------------------------------------- eruo algo by seperate -----------------------------------------")
    val cmdP = this.cmdMap
    val data = lines.map{case x=>{
      val ps = x.split("\t")
      val us = ps(0).split("#")
      (us(0),us(1).toInt)
    }}.distinct().collect()
    val labels = data.groupBy(_._1).map{ case (key,values)=>{
      (key,values.map(_._2))
    }}
    val maplabels = labels.map(_._1).zipWithIndex.map{case x=>x._2->x._1}.toMap
    val deeplevel = maplabels.size-1
    val maxv = labels.map{case x=>x._1->x._2.max}
    val indexv = scala.collection.mutable.Map[String,Int](labels.map{case x=>x._1->0}.toList: _*)

    /*
    println(maplabels)
    println(maxv)
    println(indexv)
    println("---------------------------")
    */

    var lineret:RDD[String] = sc.emptyRDD[String]
    var npart = 1
    def nodev(lines:RDD[String],level:Int) {
      val key = maplabels.apply(level)
      val ns = labels.apply(key)
      ns.foreach{case n=>{
        indexv(key) = n
        if(level==deeplevel) {
          val index1 = indexv
          val input = lines.map{case x=>{
            val ps = x.split("\t")
            (ps(1),x)
          }}.groupByKey().filter{case (key,values)=>{
            var ret = true
            values.foreach{case x=>{
              val ps = x.split("\t")
              val us = ps(0).split("#")
              if(!(index1.contains(us(0))&&(index1.apply(us(0))==us(1).toInt))) ret = false
            }}
            ret
          }}.flatMap(_._2)
          //input.saveAsTextFile(outputPath+"/input")
          println("***********************************sep_part "+npart+" = "+index1.toString())
          npart += 1
          analyseLog.append("--------------------------part "+npart+" : " + index1.toString() + "\n")
          val lens = analyseSegment(input)
          if(lens(1)>1) lineret = lineret.union(euro(input,cmdP("allfloat").toBoolean))
        } else {
          nodev(lines,level+1)
        }
      }}
    }
    nodev(lines,0)

    lineret
  }

  def dealalgo(lines:RDD[String]): RDD[String] = {
    val cmdP = this.cmdMap
    val ret = if (cmdP("dtype") == "1") {
      cos(lines,cmdP("allfloat").toBoolean)
    } else if (cmdP("dtype") == "2") {
      cospure(lines)
    } else if (cmdP("dtype") == "3") {
      if(cmdP("pre_sep").toInt>0) dealBySeperate(lines.cache())
      else euro(lines,cmdP("allfloat").toBoolean)
    }

    ret.asInstanceOf[RDD[String]]
  }

  def analyseSegment(lines:RDD[String]): Array[Long] = {
    val lenx = lines.map(x=>x.split("\t")(0)).distinct().count()
    val leny = lines.map(x=>x.split("\t")(1)).distinct().count()
    val len = lines.count()
    println("**************************distinct number of segment 1 : " + lenx.toString())
    println("**************************distinct number of segment 2 : " + leny.toString())
    println("**************************number of all lines : " + len.toString())
    analyseLog.append("all distinct number of segment 1 : " + lenx.toString() + "\n")
    analyseLog.append("all distinct number of segment 2 : " + leny.toString() + "\n")
    analyseLog.append("number of all lines : " + len.toString() + "\n")
    Array(lenx,leny,len)
  }


  def deal() {
    val cmdP = this.cmdMap

    val algo = cmdP("algo")
    println("input params : " + cmdP.toString() + "\n")
    analyseLog.append("input params : " + cmdP.toString() + "\n")
    sc.getConf.getAll.foreach(println)


    println("----------------------------------------- input file -----------------------------------------")
    println("**************************executerNum = "+executerNum)
    analyseLog.append("----------------------------------------- input file -----------------------------------------\n")
    analyseLog.append("executerNum : " + executerNum + "\n")
    val lineSor = sc.wholeTextFiles(cmdP("traindata"),executerNum).flatMap{case(path,content)=>content.split("\n")}.filter(_.split("\t").length>=3).repartition(executerNum).cache()
    analyseSegment(lineSor)

    /*
    lineSor.filter{ case x=> {
      val t = x.split("\t")
      if (t(1) == "23099200988963" || t(1) == "21883636740746") {
        true
      } else false
    }}.saveAsTextFile(outputPath+"/sor")
    */

    val lineSep = if(cmdP("pre_sep").toInt>0) {
      seperate(lineSor)
    } else {
      lineSor
    }
    println("----------------------------------------- after seperate -----------------------------------------")
    analyseLog.append("----------------------------------------- after seperate -----------------------------------------\n")
    val maxItems = lineSep.map{x=>{
      val ps = x.split("\t")
      (ps(0),1)
    }}.reduceByKey(_+_).map(_._2).max()//.saveAsTextFile(outputPath+"/maxitems")
    println("**************************number of max items : " + maxItems)
    analyseSegment(lineSep)
    //lineSep.saveAsTextFile(outputPath+"/linesep")

    val lines = if(cmdP("pre_norm").toBoolean) {
      if(cmdP("allfloat").toBoolean) {
        normalization(lineSep)
      } else {
        normalizationWithType(lineSep)
      }
    } else {
      lineSor
    }
    //lines.saveAsTextFile(outputPath+"/linenorm")

    /*
    lines.filter{ case x=> {
      val t = x.split("\t")
      if (t(1) == "23099200988963" || t(1) == "21883636740746") {
        true
      } else false
    }}.saveAsTextFile(outputPath+"/norm")
    */


    val sorRelative = dealalgo(lines)

    // get top n
    println("----------------------------------------- top n -----------------------------------------")
    analyseLog.append("----------------------------------------- top n -----------------------------------------\n")
    if(cmdP("topn").toInt>0) {
      val retSorted = sortn(sorRelative)
      retSorted.saveAsTextFile(outputPath+"/relative_topn")
    } else {
      sorRelative.saveAsTextFile(outputPath+"/relative_all")
    }


    // analyse output
    val analysePath = new Path(outputPath+"/analyse")
    val outEval2 = fs.create(analysePath,true)
    outEval2.writeBytes(analyseLog.toString())
    outEval2.close()


  }
}
