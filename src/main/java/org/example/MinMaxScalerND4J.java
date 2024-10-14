package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

public class MinMaxScalerND4J {

    public static INDArray minMaxScale(INDArray data) {
        // Find minimum and maximum values for each feature (column)
        INDArray min = data.min(0);  // Column-wise minimum
        INDArray max = data.max(0);  // Column-wise maximum

        // Calculate the difference between max and min
        INDArray range = max.sub(min);

        // Handle cases where max equals min (constant columns) by setting range to 1
        range = range.replaceWhere(Nd4j.onesLike(range), Conditions.equals(0));

        // Apply MinMax Scaling: (x - min) / (max - min)
        return data.sub(min).div(range);
    }
}