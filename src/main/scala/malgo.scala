import java.util.Calendar


import scala.collection.mutable.Map

/**
 * Created by 58 on 2015/6/3.
 */
abstract class malgo(args:Array[String])  {
  val cmdMap = parseCMD(args)
  val analyseLog = new StringBuilder()
  val stime = Calendar.getInstance().getTimeInMillis
  def deal()
  def parseCMD(args: Array[String]): Map[String, String]
}
