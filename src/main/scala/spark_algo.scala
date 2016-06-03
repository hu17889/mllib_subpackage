/**
 * Created by 58 on 2015/4/11.
 */

import org.apache.commons.cli.{Options, PosixParser}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
//import org.apache.hadoop.fs
import java.util.Date
import java.util.Calendar



import org.apache.hadoop.fs.FileSystem
//import sun.management.FileSystem


object spark_algo {
  def main(args: Array[String]) {

    // Input Params
    val parser = new PosixParser( )
    val options = new Options( )
    options.addOption("a", "algo", true, "algo type; 10. sgd 11. lbfgs")
    val cl = parser.parse( options, args, true )
    val algo = cl.getOptionValue("algo")

    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    sc.getConf.getAll.foreach(println)

    val configuration = sc.hadoopConfiguration
    configuration.setBoolean("mapreduce.output.fileoutputformat.compress", false)
    val fs = FileSystem.get(configuration)
    val modeltmp = if(algo=="10" || algo=="11" || algo=="12" || algo=="13") {
      new mllib_lr(sc, fs, args)
    } else if(algo=="21") {
      new ftrl(sc, fs, args)
    } else if(algo=="22") {
      new ftrl_batch(sc, fs, args)
    } else if(algo=="31") {
      new relative(sc, fs, args)
    } else if(algo=="40") {
      new mllib_gbdt(sc, fs, args)
    } else if(algo=="41") {
      new lambda_mart(sc, fs, args)
    } else if(algo=="91") {
      new feature_analyse(sc, fs, args)
    } else if(algo=="docs_words_analyse") {
      new docs_words_analyse(sc, fs, args)
    }

    val model = modeltmp.asInstanceOf[malgo]
    model.deal()
  }



}
