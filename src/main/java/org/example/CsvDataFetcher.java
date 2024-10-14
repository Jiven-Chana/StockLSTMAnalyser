package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CsvDataFetcher {

    private double initialPrice;

    public double getInitialPrice() {
        return initialPrice;
    }

    public INDArray fetchStockDataFromCsv(String fileName) {
        List<double[]> dataRows = new ArrayList<>();

        String filePath = Paths.get("src", "main", "data_resources", fileName).toString();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isHeader = true;

            while ((line = br.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;  // Skip the first header line
                    continue;
                }

                String[] values = line.split(",");
                try {
                    // Adjust the size of the array to match the number of columns needed
                    double[] row = new double[4];
                    row[0] = Double.parseDouble(values[1]); // Open
                    row[1] = Double.parseDouble(values[2]); // High
                    row[2] = Double.parseDouble(values[3]); // Low
                    row[3] = Double.parseDouble(values[5]); // Adj Close

                    dataRows.add(row);

                } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                    System.out.println("Error parsing line: " + line + ". Skipping this entry.");
                }
            }

            // Set initialPrice
            if (!dataRows.isEmpty()) {
                // Assuming the CSV file is ordered from most recent to oldest,
                // the last element in dataRows corresponds to the earliest date
                this.initialPrice = dataRows.get(dataRows.size() - 1)[3]; // Adj Close
            } else {
                this.initialPrice = 0.0;
            }

        } catch (IOException e) {
            System.out.println("Error reading CSV file: " + e.getMessage());
        }

        // Convert List<double[]> to INDArray
        int numRows = dataRows.size();
        int numCols = dataRows.get(0).length;

        double[][] dataArray = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            dataArray[i] = dataRows.get(i);
        }

        // Create and return INDArray
        return Nd4j.create(dataArray);
    }
}