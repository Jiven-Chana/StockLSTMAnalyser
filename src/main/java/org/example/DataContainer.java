package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;

public class DataContainer {
    private INDArray X_train;
    private INDArray y_train;

    private double initialPrice;

    public DataContainer(INDArray X_train, INDArray y_train, double initialPrice) {
        this.X_train = X_train;
        this.y_train = y_train;
        this.initialPrice = initialPrice;
    }

    public INDArray getX_train() {
        return X_train;
    }

    public INDArray getY_train() {
        return y_train;
    }

    public double getInitialPrice() {
        return initialPrice;
    }
}
