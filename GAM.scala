import scala.util.Random

object Main {
  // Eksik verileri doldurmak için fonksiyonlar
  def mean(values: List[Double]): Double = {
    values.sum / values.length
  }

  def fillMissingValues(data: List[Double], missingIndices: List[Int]): List[Double] = {
    val validData = data.zipWithIndex.filterNot { case (value, index) =>
      missingIndices.contains(index)
    }.map(_._1)

    val missingValue = mean(validData)
    data.zipWithIndex.map { case (value, index) =>
      if (missingIndices.contains(index)) missingValue else value
    }
  }

  // Cox-de Boor algoritması ile B-spline dönüşümü
  def coxDeBoor(x: Double, knots: List[Double], degree: Int, index: Int): Double = {
    if (degree == 0) {
      if (knots(index) <= x && x <= knots(index)+1) 1.0
      else 0.0
    }
    else {
      val denominator1 = knots(index + degree) - knots(index)
      val denominator2 = knots(index + degree ) - knots(index + 1)

      val term1 = if (denominator1 != 0) ((x - knots(index)) / denominator1) * coxDeBoor(x, knots, degree - 1, index)
      else 0.0

      val term2 = if (denominator2 != 0) ((knots(index + degree ) - x) / denominator2) * coxDeBoor(x, knots, degree - 1, index + 1)
      else 0.0

      term1 + term2
    }
  }
  // x1 için B-spline dönüşümü
  def s1(x1: Double): Double = {
    val xknots1 = List(2.01, 2.3, 3.2, 3.6, 3.9)
    xknots1.indices.map(index => coxDeBoor(x1, xknots1, degree=0, index)).sum
  }

  // x2 için B-spline dönüşümü
  def s2(x2: Double): Double = {
    val xknots2 = List(1.0, 2.0, 3.0, 4.0, 5.0)
    xknots2.indices.map(index => coxDeBoor(x2, xknots2, degree=0, index)).sum
  }

  // GAM modelinin hesaplanması
  def gamModel(x1: Double, x2: Double, y: Double): Double = {
    // Model parametreleri
    val beta0 = 1.0 // İntersept
    val beta1 = 2.0 // x1
    val beta2 = 3.0 // x2

    // GAM modelinin hesaplanması
    val gamResult = beta0 + beta1 * x1 + beta2 * x2 + s1(x1) + s2(x2)
    gamResult // Sonucu döndürür
  }

  //  // Lasso regresyonu
  //  def lassoRegression(x1: Double, x2: Double, y: Double, lambda: Double): Double = {
  //    // Model parametreleri
  //    val beta0 = 1.0 // İntersept
  //    val beta1 = 2.0 // x1
  //    val beta2 = 3.0 // x2
  //
  //    // Model hesaplaması
  //    val lassoResult = beta0 + beta1 * x1 + beta2 * x2 + lambda * (math.abs(beta1) + math.abs(beta2))
  //    lassoResult
  //  }
  //
  //  // Ridge regresyonu
  //  def ridgeRegression(x1: Double, x2: Double, y: Double, lambda: Double): Double = {
  //    // Model parametreleri
  //    val beta0 = 1.0 // İntersept
  //    val beta1 = 2.0 // x1
  //    val beta2 = 3.0 // x2
  //
  //    // Model hesaplaması
  //    val ridgeResult = beta0 + beta1 * x1 + beta2 * x2 + lambda * (beta1 * beta1 + beta2 * beta2)
  //    ridgeResult
  //  }
  //
  //  // Elastic Net regresyonu
  //  def elasticNetRegression(x1: Double, x2: Double, y: Double, lambda: Double, alpha: Double): Double = {
  //    // Model parametreleri
  //    val beta0 = 1.0 // İntersept
  //    val beta1 = 2.0 // x1
  //    val beta2 = 3.0 // x2
  //
  //    // Model hesaplaması
  //    val elasticNetResult = beta0 + beta1 * x1 + beta2 * x2 + lambda * alpha * (math.abs(beta1) + math.abs(beta2)) +
  //      0.5 * lambda * (1 - alpha) * (beta1 * beta1 + beta2 * beta2)
  //    elasticNetResult
  //  }

  def kNeighborsRegression(x1: Double, x2: Double, y: Double, x1Train: List[Double], x2Train: List[Double], yTrain: List[Double], k: Int): Double = {
    // K-Nearest Neighbors model hesaplaması
    val distances = x1Train.zip(x2Train).map { case (x1_train, x2_train) =>
      val distance = math.sqrt(math.pow(x1 - x1_train, 2) + math.pow(x2 - x2_train, 2))
      distance
    }

    // Mesafelere göre sıralama ve k en yakın komşunun belirlenmesi
    val sortedDistancesWithLabels = distances.zip(yTrain)
    val kNearestNeighbors = sortedDistancesWithLabels.sortBy(_._1).take(k)

    // K en yakın komşunun ortalama değerinin alınması
    val kNeighborsMean = kNearestNeighbors.map(_._2).sum / k
    kNeighborsMean
  }

  def main(args: Array[String]): Unit = {
    val x1 = List(1.0, 2.0, 3.0, 4.0, 5.0)
    val x2 = List(2.0, 4.0, 5.0, 7.0, 9.0)
    val y = List(3.0, 6.0, 0.0, 12.0, 15.0)

    val k = 5 // cross-validation kat sayısı

    val missingDataIndices = List(2) // eksik verinin indisleri
    val yWithMissingValues = fillMissingValues(y, missingDataIndices)
    val x1WithMissingValues = fillMissingValues(x1, missingDataIndices)
    val x2WithMissingValues = fillMissingValues(x2, missingDataIndices)

    val shuffledIndices = Random.shuffle(y.indices.toList)

    val foldSize = yWithMissingValues.length / k
    val cvResults = (0 until k).map { i =>
      val validationIndices = shuffledIndices.slice(i * foldSize, (i + 1) * foldSize)
      val trainingIndices = shuffledIndices.diff(validationIndices)

      val x1Train = trainingIndices.map(x1WithMissingValues)
      val x2Train = trainingIndices.map(x2WithMissingValues)
      val yTrain = trainingIndices.map(yWithMissingValues)

      val x1Test = validationIndices.map(x1WithMissingValues)
      val x2Test = validationIndices.map(x2WithMissingValues)
      val yTest = validationIndices.map(yWithMissingValues)

      // Modellerin eğitimi ve test veri seti üzerinde tahmin yapma
      val gamPrediction = gamModel(x1Test.head, x2Test.head, yTest.head)
      //val elasticNetPrediction = elasticNetRegression(x1Test.head, x2Test.head, yTest.head, lambda = 1.0, alpha = 0.5)
      //val ridgePrediction = ridgeRegression(x1Test.head, x2Test.head, yTest.head, lambda = 1.0)
      //val lassoPrediction = lassoRegression(x1Test.head, x2Test.head, yTest.head, lambda = 1.0)
      val kNeighborsPrediction = kNeighborsRegression(x1Test.head, x2Test.head, yTest.head, x1Train, x2Train, yTrain, k = 3)

      // Hataların hesaplanması ve sonuçların yazdırılması
      val cvError = gamPrediction - yTest.head
      val mse = (cvError * cvError) / x1.length
      println(s"Fold $i - GAM Prediction: $gamPrediction, k-Neighbors Prediction: $kNeighborsPrediction, MSE: $mse")

      // MSE (Mean Squared Error) hesaplanması
      mse
    }

    // Cross-validation sonuçlarının hesaplanması
    val averageMSE = cvResults.sum / cvResults.length
    println(s"Average MSE: $averageMSE")
  }
}