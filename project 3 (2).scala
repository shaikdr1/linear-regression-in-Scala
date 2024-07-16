// Databricks notebook source
// MAGIC %md
// MAGIC **Main Project (100 pts)** \
// MAGIC Implement closed-form solution when m(number of examples is large) and n(number of features) is small:
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]
// MAGIC Here, X is a distributed matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC Steps:
// MAGIC 1. Create an example RDD for matrix X and vector y
// MAGIC 2. Compute \\[ \scriptsize \mathbf{(X^TX)}\\]
// MAGIC 3. Convert the result matrix to a Breeze Dense Matrix and compute pseudo-inverse
// MAGIC 4. Compute \\[ \scriptsize \mathbf{X^Ty}\\] and convert it to Breeze Vector
// MAGIC 5. Multiply \\[ \scriptsize \mathbf{(X^TX)}^{-1}\\] with \\[ \scriptsize \mathbf{X^Ty}\\]

// COMMAND ----------



// COMMAND ----------

import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.pinv
import breeze.linalg.DenseVector
val sc = SparkContext.getOrCreate()
val mat_data = Array(
  Array(1.0, 2.0),
  Array(3.0, 4.0)

)
val y_data = Array(5.0, 6.0)
val mat: RDD[Vector] = sc.parallelize(mat_data.map(row => Vectors.dense(row)))
val mul = new RowMatrix(mat)
val XTX: Matrix = mul.computeGramianMatrix()
val re_dense = new breeze.linalg.DenseMatrix(XTX.numRows, XTX.numCols, XTX.toArray)
val inver = pinv(re_dense)


// COMMAND ----------


val M_Small = sc.parallelize(mat_data.zipWithIndex.flatMap{case (row, i) => row.zipWithIndex.map { case (value, j) => ((i.toInt, j.toInt), value)}})
val N_Small = sc.parallelize(y_data.zipWithIndex.map{case(value, i) => (i.toInt, value)})
val shar = M_Small.map{case ((i, k), valuem) => ( (k, i), valuem)}
val first = shar.map {case ((i, k), valuem) => (k, (i, valuem))}
val second = N_Small.map{ case (j, valuen) => (j, (valuen))}
val result = first.join(second).map{case(_, ((i, valuem),(valuen)))=> ((i),valuem * valuen)}
            .reduceByKey(_+_)
             result
val sdr = result.collect()
val rac = sdr.map{ case (i, value) => (i, value) }.sortBy(_._1).map(_._2)
val na = new DenseVector(rac)

// COMMAND ----------

val c = inver * na

// COMMAND ----------

// MAGIC %md
// MAGIC Alternatively, you can implement \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\] using Spark DataFrame

// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 1(10 pts)** \
// MAGIC Implement \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\] where you compute \\[ \scriptsize \mathbf{\theta=(X^TX)}\\] using outer-product technique. You can use either Scala or DataFame to implement this.

// COMMAND ----------

// MAGIC %md
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 2(20 pts)** \
// MAGIC Run your algorithm on Boston Housing Dataset: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices?resource=download
