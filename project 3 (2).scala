

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


val c = inver * na
