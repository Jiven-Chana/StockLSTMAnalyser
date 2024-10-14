package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TechnicalIndicators {

    // Calculate the Relative Strength Index (RSI)
    public INDArray calculateRSI(INDArray prices, int period) {
        INDArray rsi = Nd4j.create(prices.length());
        INDArray gains = Nd4j.zeros(period);
        INDArray losses = Nd4j.zeros(period);

        // Initialize gains and losses for the first period
        for (int i = 1; i < period; i++) {
            double diff = prices.getDouble(i) - prices.getDouble(i - 1);
            if (diff > 0) {
                gains.putScalar(i, diff);
            } else {
                losses.putScalar(i, -diff);
            }
        }

        double avgGain = gains.mean().getDouble(0);
        double avgLoss = losses.mean().getDouble(0);

        // Start calculating RSI from the period onward
        for (int i = period; i < prices.length(); i++) {
            double diff = prices.getDouble(i) - prices.getDouble(i - 1);
            if (diff > 0) {
                avgGain = (avgGain * (period - 1) + diff) / period;
                avgLoss = (avgLoss * (period - 1)) / period;
            } else {
                avgGain = (avgGain * (period - 1)) / period;
                avgLoss = (avgLoss * (period - 1) - diff) / period;
            }

            // Avoid division by zero
            if (avgLoss == 0) {
                rsi.putScalar(i, 100);
            } else {
                double rs = avgGain / avgLoss;
                rsi.putScalar(i, 100 - (100 / (1 + rs)));
            }
        }
        return rsi;
    }

    // Calculate the Exponential Moving Average (EMA)
    public INDArray calculateEMA(INDArray prices, int period) {
        INDArray ema = Nd4j.create(prices.length());
        double multiplier = 2.0 / (period + 1);

        // Initialize the first value as the first price
        ema.putScalar(0, prices.getDouble(0));

        // Calculate EMA for the rest of the prices
        for (int i = 1; i < prices.length(); i++) {
            double prevEma = ema.getDouble(i - 1);
            double price = prices.getDouble(i);
            ema.putScalar(i, ((price - prevEma) * multiplier) + prevEma);
        }

        return ema;
    }
}
