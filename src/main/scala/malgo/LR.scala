package malgo

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.{SparkException, Logging}
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.{Optimizer, SquaredL2Updater, LogisticGradient, LBFGS}
import org.apache.spark.mllib.regression.{LabeledPoint, GeneralizedLinearModel}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.mllib.classification.{LogisticRegressionModel=>LRModel}

import common.mutil
import org.apache.spark.rdd.RDD


/**
 * Classification model trained using Logistic Regression.
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 */
class LogisticRegressionModel  (
                                override val weights: Vector,
                                override val intercept: Double)
  extends LRModel(weights, intercept) with Serializable {

  private var threshold: Option[Double] = Some(0.5)

  /**
   * :: Experimental ::
   * Sets the threshold that separates positive predictions from negative predictions. An example
   * with prediction score greater than or equal to this threshold is identified as an positive,
   * and negative otherwise. The default value is 0.5.
   */
  @Experimental
  override def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }

  /**
   * :: Experimental ::
   * Clears the threshold so that `predict` will output raw prediction scores.
   */
  @Experimental
  override def clearThreshold(): this.type = {
    threshold = None
    this
  }

  override protected def predictPoint(dataMatrix: Vector, weightMatrix: Vector,
                                         intercept: Double) = {
    val margin = mutil.vectorToBreeze(weightMatrix).dot(mutil.vectorToBreeze(dataMatrix)) + intercept
    val score = 1.0 / (1.0 + math.exp(-margin))
    threshold match {
      case Some(t) => if (score < t) 0.0 else 1.0
      case None => score
    }
  }
}

/**
 * :: DeveloperApi ::
 * GeneralizedLinearAlgorithm implements methods to train a Generalized Linear Model (GLM).
 * This class should be extended with an Optimizer to create a new GLM.
 */
@DeveloperApi
abstract class GeneralizedLinearAlgorithm[M <: GeneralizedLinearModel]
  extends Logging with Serializable {

  protected val validators: Seq[RDD[LabeledPoint] => Boolean] = List()

  /** The optimizer to solve the problem. */
  def optimizer: Optimizer

  /** Whether to add intercept (default: false). */
  protected var addIntercept: Boolean = false

  protected var validateData: Boolean = true

  /**
   * Whether to perform feature scaling before model training to reduce the condition numbers
   * which can significantly help the optimizer converging faster. The scaling correction will be
   * translated back to resulting model weights, so it's transparent to users.
   * Note: This technique is used in both libsvm and glmnet packages. Default false.
   */
  private var useFeatureScaling = false

  /**
   * Set if the algorithm should use feature scaling to improve the convergence during optimization.
   */
  protected def setFeatureScaling(useFeatureScaling: Boolean): this.type = {
    this.useFeatureScaling = useFeatureScaling
    this
  }

  /**
   * Create a model given the weights and intercept
   */
  protected def createModel(weights: Vector, intercept: Double): M

  /**
   * Set if the algorithm should add an intercept. Default false.
   * We set the default to false because adding the intercept will cause memory allocation.
   */
  def setIntercept(addIntercept: Boolean): this.type = {
    this.addIntercept = addIntercept
    this
  }

  /**
   * Set if the algorithm should validate data before training. Default true.
   */
  def setValidateData(validateData: Boolean): this.type = {
    this.validateData = validateData
    this
  }

  /**
   * Run the algorithm with the configured parameters on an input
   * RDD of LabeledPoint entries.
   */
  def run(input: RDD[LabeledPoint]): M = {
    val numFeatures: Int = input.first().features.size
    val initialWeights = Vectors.dense(new Array[Double](numFeatures))
    run(input, initialWeights)
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD
   * of LabeledPoint entries starting from the initial weights provided.
   */
  def run(input: RDD[LabeledPoint], initialWeights: Vector): M = {

    // Check the data properties before running the optimizer
    if (validateData && !validators.forall(func => func(input))) {
      throw new SparkException("Input validation failed.")
    }

    /**
     * Scaling columns to unit variance as a heuristic to reduce the condition number:
     *
     * During the optimization process, the convergence (rate) depends on the condition number of
     * the training dataset. Scaling the variables often reduces this condition number
     * heuristically, thus improving the convergence rate. Without reducing the condition number,
     * some training datasets mixing the columns with different scales may not be able to converge.
     *
     * GLMNET and LIBSVM packages perform the scaling to reduce the condition number, and return
     * the weights in the original scale.
     * See page 9 in http://cran.r-project.org/web/packages/glmnet/glmnet.pdf
     *
     * Here, if useFeatureScaling is enabled, we will standardize the training features by dividing
     * the variance of each column (without subtracting the mean), and train the model in the
     * scaled space. Then we transform the coefficients from the scaled space to the original scale
     * as GLMNET and LIBSVM do.
     *
     * Currently, it's only enabled in LogisticRegressionWithLBFGS
     */
    val scaler = if (useFeatureScaling) {
      (new StandardScaler).fit(input.map(x => x.features))
    } else {
      null
    }

    // Prepend an extra variable consisting of all 1.0's for the intercept.
    val data = if (addIntercept) {
      if(useFeatureScaling) {
        input.map(labeledPoint =>
          (labeledPoint.label, appendBias(scaler.transform(labeledPoint.features))))
      } else {
        input.map(labeledPoint => (labeledPoint.label, appendBias(labeledPoint.features)))
      }
    } else {
      if (useFeatureScaling) {
        input.map(labeledPoint => (labeledPoint.label, scaler.transform(labeledPoint.features)))
      } else {
        input.map(labeledPoint => (labeledPoint.label, labeledPoint.features))
      }
    }

    val initialWeightsWithIntercept = if (addIntercept) {
      appendBias(initialWeights)
    } else {
      initialWeights
    }

    val weightsWithIntercept = optimizer.optimize(data, initialWeightsWithIntercept)

    val intercept = if (addIntercept) weightsWithIntercept(weightsWithIntercept.size - 1) else 0.0
    var weights =
      if (addIntercept) {
        Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1))
      } else {
        weightsWithIntercept
      }

    /**
     * The weights and intercept are trained in the scaled space; we're converting them back to
     * the original scale.
     *
     * Math shows that if we only perform standardization without subtracting means, the intercept
     * will not be changed. w_i = w_i' / v_i where w_i' is the coefficient in the scaled space, w_i
     * is the coefficient in the original space, and v_i is the variance of the column i.
     */
    if (useFeatureScaling) {
      weights = scaler.transform(weights)
    }

    createModel(weights, intercept)
  }
}

/**
 * Created by 58 on 2015/11/9.
 */
class LR_LBFGS extends GeneralizedLinearAlgorithm[LRModel] with Serializable {

    this.setFeatureScaling(true)

    override val optimizer = new LBFGS(new LogisticGradient, new SquaredL2Updater)

    override protected val validators = List(DataValidators.binaryLabelValidator)

    override protected def createModel(weights: Vector, intercept: Double) = {
      new LRModel(weights, intercept)
    }
}
