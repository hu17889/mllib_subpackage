/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.{DeveloperApi, Experimental, Since}
import org.apache.spark.mllib.tree.impl.LambdaTreePoint


/**
 * Interface for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 * @param statsSize  Length of the vector of sufficient statistics for one bin.
 */
private[spark] abstract class LambdaImpurityAggregator(val statsSize: Int) extends Serializable {

  /**
   * Merge the stats from one bin into another.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for (node, feature, bin) which is modified by the merge.
   * @param otherOffset  Start index of stats for (node, feature, other bin) which is not modified.
   */
  def merge(allStats: Array[Double], offset: Int, otherOffset: Int): Unit = {
    var i = 0
    while (i < statsSize) {
      allStats(offset + i) += allStats(otherOffset + i)
      i += 1
    }
  }

  /**
   * Update stats for one (node, feature, bin) with the given label.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def update(allStats: Array[Double], offset: Int, label: Double,  point: LambdaTreePoint, instanceWeight: Double): Unit

  /**
   * Get an [[ImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): ImpurityCalculator

}




