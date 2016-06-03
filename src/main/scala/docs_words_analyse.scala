/**
 * Created by 58 on 2015/4/11.
 */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD


import scala.collection.mutable.{ArrayBuffer, StringBuilder, Map}

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.commons.cli._

import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.IDF

import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors

import common.mutil


class docs_words_analyse(sc:SparkContext, fs:FileSystem, args:Array[String]) extends malgo(args) with  Serializable  {
  var outputPath = ""
  var weightNum = 0

  val executerNum = sc.getConf.get("spark.executor.instances").toInt



  def parseCMD(args: Array[String]): Map[String, String] = {
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("h", "help", false, "Print this usage information")
    options.addOption("tr", "traindata", true, "train data path")
    options.addOption("d", "dstdata", true, "output data path")
    options.addOption("a", "algo", true, "algo type; tfidf")
    options.addOption("t_n", "topic_num", true, "default 10")
    options.addOption("t_ni", "topic_iternum", true, "default 20")
    options.addOption("t_nw", "topic_wordnum", true, "the number of top words of each topic; default 20")
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
    cmdMap += ("topic_num"->cl.getOptionValue("topic_num","10"))
    cmdMap += ("topic_iternum"->cl.getOptionValue("topic_iternum","20"))
    cmdMap += ("topic_wordnum"->cl.getOptionValue("topic_wordnum","20"))
    return  cmdMap
  }

  //  words  每个元素是一个文档 (docid,words)
  def anaTFIDF(words:RDD[(String,Array[String])], label:String): Unit = {
    val nd = words.count()
    // wc 每个单词的出现次数
    val wc = words.map(_._2).flatMap(x=>x).map(x=>(x,1)).reduceByKey(_+_)
    // tf 单词在文档中的词频
    val tf = words.map(x=>(x._1, x._2.groupBy(y=>y).map{case (word,words)=>(word,words.length)})).cache()
    // df 单词的文档频率
    val df = words.map(_._2).flatMap(x=>x.distinct).map(x=>(x,1)).reduceByKey(_+_).cache()
    val dfbroad = sc.broadcast(df.collect().toMap)

    // tfidf
    /*
    val tfidf = tf.map(x=>(x._2, x._1)).flatMap{case (words,index)=>
      // (word,(tf,docid))
      words.toArray.map(x=>(x._1,(x._2,index)))
    }.groupByKey().join(df)
      .flatMap{case (word,(tf_docids,df))=>{
      tf_docids.toArray.map{case (tf,docid)=>(docid,(word,tf*math.log((nd+1)/(df+1))))}
    }}
      .aggregateByKey(ArrayBuffer[(String,Double)]()) ((arr,ele)=>arr+=ele,(arr1,arr2)=>arr1++=arr2)
      .map{case (docid,words)=>(docid,words.toArray.sortBy(-_._2))}
      */
    val tfidf = tf.map{case (docid,word_tfs)=>{
      val new_word_tfs = word_tfs.map{case (word,tf)=>(word,tf.toFloat/dfbroad.value.apply(word))}
      (docid,new_word_tfs)
    }}

    analyseLog.append(label+" : num of all docs = " + tf.count() + "\n")
    analyseLog.append(label+" : num of all words = " + wc.count() + "\n")
    wc.map(x=>(x._2,x._1)).saveAsTextFile(outputPath+"/mwc_"+label)
    df.map(x=>(x._2,x._1)).saveAsTextFile(outputPath+"/mdf_"+label)
    tfidf.map{case (docid,words)=>docid+"\t"+words.map(_.toString()).mkString(" ")}.saveAsTextFile(outputPath+"/mtfidf_"+label)

    tf.unpersist()
    df.unpersist()
  }

  def anaInput(trainlines:RDD[(String,Array[String])]): (RDD[(String,Array[String])],RDD[(String,Array[String])]) = {
    // RDD[(docid,Array[(word,Array[tag])])]
    val docs = trainlines.map{case (docid,words)=>
      def dealPart(word:String): (String,Array[String]) = {
        val p = word.split("_",2)
        if(p.length!=2) return (word,Array.empty)

        val regex = """\[(.*)\]""".r
        val m = regex.findAllIn(p(1))
        if(m.isEmpty) return (word,Array.empty)
        //val regex(tagstr) = p(1)
        val tags = m.group(1).split(",")
        (p(0),tags)
      }
      // (word,Array[tag])
      val ret = Array.ofDim[(String,Array[String])](words.length)
      for( i <- 0 to words.length-1) {
        ret(i) = dealPart(words(i))
      }
      (docid,ret)
    }.cache()
    //docs.saveAsTextFile(outputPath+"/docs")

    // RDD[(docid,Array[word])]

    val allwords = docs.map{case (docid,line) =>
      (docid,line.map(_._1))
    }



    // RDD[Array[word]]
    val tagwords = docs.map{case (docid,line) =>
      (docid, line.filter(x=>x._2.length>0).map(_._1))
    }
    anaTFIDF(allwords,"allwords")
    anaTFIDF(tagwords,"tagwords")
    anaTFIDF(docs.map(x=>(x._1,x._2.flatMap(_._2))),"tags")
    docs.unpersist()

    (allwords,tagwords)
  }

  // 没用
  def anaTfIdf(input:RDD[Array[String]],label:String): Unit = {
    val wordsDir = input.flatMap(x=>x).distinct.map(x=>(x.##,x)).reduceByKey((a,b)=>a)

    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(input.map(_.toSeq))
    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
    val tfidfWords = tfidf.flatMap{case x=>
      val input = x.toSparse
      val ret = Array.ofDim[(Int,Double)](input.numNonzeros)

      for( i <- 0 to input.numNonzeros-1) {
        ret(i) = (input.indices(i),input.values(i))
      }
      // Array[(hashkey,tfidf)]
      ret
    }.distinct.join(wordsDir).map{case (hashkey,(value,word))=>(value,word)}.sortByKey(false)

    tf.saveAsTextFile(outputPath+"/tf_"+label)
    tfidf.saveAsTextFile(outputPath+"/tfidf_"+label)
    tfidfWords.saveAsTextFile(outputPath+"/tfidfwords_"+label)
  }



  def dealLDAModel(ldaModel:DistributedLDAModel): Unit = {

    analyseLog.append("topics num : "+ldaModel.k+"\n")
    analyseLog.append("logLikelihood : "+ldaModel.logLikelihood+"\n")
    analyseLog.append("logPrior : "+ldaModel.logPrior+"\n")
    analyseLog.append("topicConcentration : "+ldaModel.topicConcentration+"\n")
    analyseLog.append("vocabSize : "+ldaModel.vocabSize+"\n")
    mutil.saveVector(fs,ldaModel.docConcentration,outputPath+"/docConcentration")

  }

  def anatopic(tagwords:RDD[(String,Array[String])], label:String): Unit = {
    analyseLog.append("========================topic of "+label+"======================\n")
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(tagwords.map(_._2.toSeq))
    val corpus = tf.zipWithIndex.map(_.swap).cache()

    val ldaModel = new LDA().setK(cmdMap("topic_num").toInt).setMaxIterations(cmdMap("topic_iternum").toInt).run(corpus).asInstanceOf[DistributedLDAModel]
    corpus.unpersist()
    analyseLog.append("========================lda model data of "+label+"======================\n")
    dealLDAModel(ldaModel)

    // 词表：word_hashcode word
    val vocabSize = ldaModel.vocabSize
    val wordsDic = tagwords.flatMap(x=>x._2).distinct.map{x=>
      def nonNegativeMod(x: Int, mod: Int): Int = {
        val rawMod = x % mod
        rawMod + (if (rawMod < 0) mod else 0)
      }
      (nonNegativeMod(x.##,vocabSize),x)
    }.reduceByKey((a,b)=>a).collect.toMap
    analyseLog.append("tags : num of tag words dic = " + wordsDic.toArray.length + "\n")


    // 每个topic的高权重词列表
    analyseLog.append("========================topic top words of "+label+"======================\n")
    val topicsTerms = ldaModel.describeTopics(cmdMap("topic_wordnum").toInt)
    for (topic <- Range(0,cmdMap("topic_num").toInt)) {
      val data = topicsTerms(topic)
      analyseLog.append("Topic " + topic + ":"+"\n")
      data._1.zip(data._2).map{x=>analyseLog.append(x._1+"\t"+wordsDic(x._1)+"\t"+x._2+"\n")}
    }
  }

  def deal() {
      val cmdP = this.cmdMap
      println("**************input params : " + cmdP.toString() + "\n")
      analyseLog.append("input params : " + cmdP.toString() + "\n")
      outputPath = cmdP("dstdata")
      val algo = cmdP("algo")
      analyseLog.append("executerNum : " + executerNum + "\n")

      println("++++++++++++++++++++++"+executerNum)
      //val trainlines = sc.wholeTextFiles(cmdP("traindata"),2000).flatMap{x=>x._2.split("\n")}.map(_.split("\t")).map(x=>(x(0),x(1).split(" "))).cache()
      val trainlines = sc.textFile(cmdP("traindata"),2*executerNum).map(_.split("\t")).map(x=>(x(0),x(1).split(" "))).cache()
      // 分析文档和单词的普通统计信息以及tfidf数据
      val (allwords,tagwords) = anaInput(trainlines)
      // (docid,tagwords)
      //tagwords.map(x=>x._1+"\t"+x._2.mkString("_")).saveAsTextFile(outputPath+"/mid_tagwords")

      // 分析topic
      anatopic(tagwords,"tagwords")
      anatopic(allwords,"allwords")




    // Save and load model.
    //ldaModel.save(sc, outputPath+"/LDAModel")
    //val sameModel = DistributedLDAModel.load(sc, outputPath+"/LDAModel")


    // analyse output
    val analysePath = new Path(outputPath+"/analyse")
    val outEval2 = fs.create(analysePath,true)
    //outEval2.writeBytes(analyseLog.toStrin
    // g())
    outEval2.writeUTF(analyseLog.toString())
    outEval2.close()
  }
}
