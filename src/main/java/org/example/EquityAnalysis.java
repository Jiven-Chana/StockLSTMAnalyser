package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.ArrayList;
import java.util.List;

public class EquityAnalysis {

    // Custom Condition interface to apply conditions to INDArrays
    public interface ValueCondition {
        boolean apply(double value);
    }

    public DataContainer processData() {
        // Step 1: Fetch stock data from CSV
        CsvDataFetcher csvDataFetcher = new CsvDataFetcher();
        String fileName = "TrimmedSPX.csv";  // Your actual file name
        INDArray stockData = csvDataFetcher.fetchStockDataFromCsv(fileName);

        double initialPrice = csvDataFetcher.getInitialPrice();

        // Ensure stockData is valid and contains expected arrays
        if (stockData == null || stockData.isEmpty()) {
            System.out.println("Error: Stock data is missing or incomplete. Please check the CSV file.");
            return null;
        }

        // Step 2: Cast stockData to DOUBLE and scale using MinMaxScaler
        stockData = stockData.castTo(DataType.DOUBLE);
        INDArray scaledData = MinMaxScalerND4J.minMaxScale(stockData);
        System.out.println("Scaled Data:\n" + scaledData);

        // New Step: Replace NaN values with 0 in scaled data
        scaledData = scaledData.replaceWhere(Nd4j.zerosLike(scaledData), Conditions.isNan());
        System.out.println("Scaled Data (NaN values replaced with 0):\n" + scaledData);

        // Filter out rows that are completely empty (all values zero or missing)
        INDArray validRowsMask = scaledData.sum(1).gt(0);  // Valid rows have sum > 0
        INDArray filteredData = scaledData.get(flattenToINDArrayIndex(indicesWhere(validRowsMask, value -> value > 0)), NDArrayIndex.all());

        System.out.println("Filtered Scaled Data (Invalid rows removed):\n" + filteredData);

        // Step 4: Calculate indicators (RSI, EMA) using the adjusted close prices (column 3)
        TechnicalIndicators ti = new TechnicalIndicators();
        INDArray rsi = ti.calculateRSI(filteredData.getColumn(3), 15).castTo(DataType.DOUBLE);
        INDArray emaf = ti.calculateEMA(filteredData.getColumn(3), 20).castTo(DataType.DOUBLE);
        INDArray emam = ti.calculateEMA(filteredData.getColumn(3), 100).castTo(DataType.DOUBLE);
        INDArray emas = ti.calculateEMA(filteredData.getColumn(3), 150).castTo(DataType.DOUBLE);

        // Step 6: Add indicators as columns to the data
        filteredData = Nd4j.hstack(filteredData, rsi.reshape(-1, 1), emaf.reshape(-1, 1), emam.reshape(-1, 1), emas.reshape(-1, 1));

        // Step 7: Compute Target (Adj Close - Open)
        INDArray adjClosePrices = filteredData.get(NDArrayIndex.all(), NDArrayIndex.point(3));
        INDArray openPrices = filteredData.get(NDArrayIndex.all(), NDArrayIndex.point(0));
        INDArray target = adjClosePrices.sub(openPrices);

        // Step 7.1: Shift Target backward by one period
        INDArray shiftedTarget = Nd4j.concat(0, Nd4j.zeros(1).castTo(DataType.DOUBLE), target.get(NDArrayIndex.interval(0, target.size(0) - 1)));
        System.out.println("Target (shifted by -1):\n" + shiftedTarget);

        // Step 8: Compute TargetClass (1 if Target > 0, else 0)
        INDArray targetClass = shiftedTarget.gt(0).castTo(DataType.DOUBLE);
        System.out.println("TargetClass (binary):\n" + targetClass);

        // Step 9: Prepare the final data for training
        INDArray finalData = Nd4j.hstack(filteredData, target.reshape(-1, 1), targetClass.reshape(-1, 1));

        // Prepare features for model input
        INDArray X = prepareFeatures(finalData, Config.BACKCANDLES);

        // Ensure X has enough data
        if (X.size(0) == 0) {
            System.out.println("Error: Not enough data to prepare features.");
            return null;
        }

        // Align target `y` with `X`
        INDArray y = shiftedTarget.get(NDArrayIndex.interval(Config.BACKCANDLES, shiftedTarget.size(0)));

        // Ensure y has enough data
        if (y.size(0) == 0) {
            System.out.println("Error: Not enough data in target variable y.");
            return null;
        }

        // Split data into training and test sets (80% training, 20% testing)
        int splitLimit = (int) (X.size(0) * 0.8);
        INDArray X_train = X.get(NDArrayIndex.interval(0, splitLimit));
        INDArray X_test = X.get(NDArrayIndex.interval(splitLimit, X.size(0)));
        INDArray y_train = y.get(NDArrayIndex.interval(0, splitLimit));
        INDArray y_test = y.get(NDArrayIndex.interval(splitLimit, y.size(0)));

        // Print details
        System.out.println("Train data (X_train) shape: (" + X_train.size(0) + ", " + X_train.size(1) + ", " + X_train.size(2) + ")");
        System.out.println("y_train shape: (" + y_train.size(0) + ")");

        // Return X_train and y_train in a DataContainer object
        return new DataContainer(X_train, y_train, initialPrice);
    }

    // Helper method to prepare features for model input
    public static INDArray prepareFeatures(INDArray data, int backcandles) {
        int numFeatures = (int) data.size(1);  // Number of features
        int numSamples = (int) data.size(0) - backcandles;

        if (numSamples <= 0) {
            System.out.println("Error: Not enough samples to prepare features.");
            return Nd4j.empty();
        }

        INDArray X = Nd4j.create(numSamples, backcandles, numFeatures);

        for (int i = backcandles; i < data.size(0); i++) {
            INDArray slice = data.get(NDArrayIndex.interval(i - backcandles, i), NDArrayIndex.all());
            X.put(new INDArrayIndex[]{NDArrayIndex.point(i - backcandles)}, slice);
        }

        return X;
    }

    // Method to get indices where the condition is true
    public static INDArrayIndex[] indicesWhere(INDArray array, ValueCondition condition) {
        List<Integer> indicesList = new ArrayList<>();

        // Loop through all elements of the array
        for (int i = 0; i < array.size(0); i++) {
            if (condition.apply(array.getDouble(i))) {
                indicesList.add(i);  // Collect indices where the condition is true
            }
        }

        // Convert List<Integer> to long[]
        long[] indicesArray = indicesList.stream().mapToLong(i -> i).toArray();

        // Return an array of NDArrayIndex for indexing
        return new INDArrayIndex[]{NDArrayIndex.indices(indicesArray)};
    }

    // New method to flatten the result of indicesWhere() into a single INDArrayIndex
    public static INDArrayIndex flattenToINDArrayIndex(INDArrayIndex[] indices) {
        // If the indicesWhere call returns multiple indices, we need to merge them into a single one
        // For now, we return the first one assuming it's a single index array
        return indices[0];
    }
}