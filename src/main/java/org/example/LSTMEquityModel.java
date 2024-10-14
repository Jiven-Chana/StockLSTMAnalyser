package org.example;

import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import org.knowm.xchart.XYChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChartBuilder;

import java.awt.BasicStroke;

public class LSTMEquityModel {

    public static void main(String[] args) {

        // Set random seed for reproducibility
        Nd4j.getRandom().setSeed(10);

        int numFeatures = 8;   // Number of features
        int numOutputs = 1;    // Output layer has 1 unit (for regression)
        int numEpochs = 30;    // Number of epochs

        // Fetch processed data from EquityAnalysis
        EquityAnalysis equityAnalysis = new EquityAnalysis();
        DataContainer dataContainer = equityAnalysis.processData();  // Process the data

        // Extract X_train and y_train from DataContainer
        INDArray X_train = dataContainer.getX_train();
        INDArray y_train = dataContainer.getY_train();

        System.out.println("X_train shape before slicing: " + Arrays.toString(X_train.shape()));

        // Select only the first 8 features
        X_train = X_train.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, numFeatures)).dup();

        System.out.println("X_train shape after slicing: " + Arrays.toString(X_train.shape()));

        // --- ADD VALIDATION SPLIT HERE ---
        int validationSplitLimit = (int) (X_train.size(0) * 0.9);

        // Training set (90%)
        INDArray X_train_subset = X_train.get(NDArrayIndex.interval(0, validationSplitLimit));
        INDArray y_train_subset = y_train.get(NDArrayIndex.interval(0, validationSplitLimit));

        // Validation set (10%)
        INDArray X_validation = X_train.get(NDArrayIndex.interval(validationSplitLimit, X_train.size(0)));
        INDArray y_validation = y_train.get(NDArrayIndex.interval(validationSplitLimit, y_train.size(0)));

        // Reshape data to 2D for scaling
        INDArray X_train_subset_2D = X_train_subset.reshape(
                X_train_subset.size(0) * X_train_subset.size(1), X_train_subset.size(2));
        INDArray X_validation_2D = X_validation.reshape(
                X_validation.size(0) * X_validation.size(1), X_validation.size(2));

        // --- Scaling Features ---
        // Scaling using NormalizerMinMaxScaler for features
        NormalizerMinMaxScaler featureScaler = new NormalizerMinMaxScaler(0, 1);
        featureScaler.fit(new DataSet(X_train_subset_2D, null));  // Fit scaler on training data

        // Transform training data
        featureScaler.transform(X_train_subset_2D);
        // Transform validation data using the same scaler
        featureScaler.transform(X_validation_2D);

        // Reshape back to 3D after scaling
        X_train_subset = X_train_subset_2D.reshape(
                X_train_subset.size(0), X_train_subset.size(1), X_train_subset.size(2));
        X_validation = X_validation_2D.reshape(
                X_validation.size(0), X_validation.size(1), X_validation.size(2));

        // --- Scaling Labels ---
        // Reshape labels to 2D for scaling if necessary
        if (y_train_subset.rank() == 1) {
            y_train_subset = y_train_subset.reshape(y_train_subset.size(0), 1);
        }
        if (y_validation.rank() == 1) {
            y_validation = y_validation.reshape(y_validation.size(0), 1);
        }

        // Scaling labels using a separate NormalizerMinMaxScaler
        NormalizerMinMaxScaler labelScaler = new NormalizerMinMaxScaler(0, 1);
        labelScaler.fit(new DataSet(y_train_subset, y_train_subset));  // Fit scaler on training labels

        // Transform labels
        labelScaler.transform(y_train_subset);
        labelScaler.transform(y_validation);

        // Permute input data to match DL4J expectations
        X_train_subset = X_train_subset.permute(0, 2, 1); // Shape: [batchSize, features, timeSteps]
        X_validation = X_validation.permute(0, 2, 1);

        // Print shapes for verification
        System.out.println("X_train_subset shape: " + Arrays.toString(X_train_subset.shape()));
        System.out.println("y_train_subset shape: " + Arrays.toString(y_train_subset.shape()));

        // Set optimizer parameters
        double learningRate = 0.001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;

        // Build LSTM Model with LastTimeStep wrapping the LSTM layer
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(10)  // Set seed for model initialization
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate, beta1, beta2, epsilon))  // Adam optimizer
                .list()
                .layer(0, new LastTimeStep(new LSTM.Builder()
                        .nIn(numFeatures)  // Number of input features = 8
                        .nOut(150)         // Number of LSTM units = 150
                        .activation(Activation.TANH)  // LSTM uses Tanh activation
                        .build()))
                .layer(1, new OutputLayer.Builder(LossFunction.MSE)  // Mean Squared Error Loss
                        .nIn(150)
                        .nOut(numOutputs)
                        .activation(Activation.IDENTITY)  // Identity activation
                        .build())
                .build();

        // Initialize the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Create DataSet objects
        DataSet trainData = new DataSet(X_train_subset, y_train_subset);
        DataSet valData = new DataSet(X_validation, y_validation);

        // Train the model
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(trainData);

            // Evaluate the model on the validation set
            double validationScore = model.score(valData);
            System.out.println("Epoch " + (epoch + 1) + " complete. Validation Score: " + validationScore);
        }

        // After training, use the model to make predictions on training and validation data
        INDArray y_train_pred_scaled = model.output(X_train_subset);
        INDArray y_validation_pred_scaled = model.output(X_validation);

        // Combine predictions and actual values
        INDArray y_pred_scaled = Nd4j.concat(0, y_train_pred_scaled, y_validation_pred_scaled);
        INDArray y_actual_scaled = Nd4j.concat(0, y_train_subset, y_validation);

        // Inverse transform the predictions and actual values to original scale
        // labelScaler.revertLabels(y_pred_scaled);
        // labelScaler.revertLabels(y_actual_scaled);

        // Now y_pred_scaled and y_actual_scaled contain the inverse-transformed data

        // Convert to double arrays
        double[] y_pred_array = y_pred_scaled.data().asDouble();
        double[] y_actual_array = y_actual_scaled.data().asDouble();

        System.out.println("Predicted Prices (first 5 values):");
        System.out.println(Arrays.toString(Arrays.copyOfRange(y_pred_array, 0, 5)));
        System.out.println("Actual Prices (first 5 values):");
        System.out.println(Arrays.toString(Arrays.copyOfRange(y_actual_array, 0, 5)));

        // Plotting the predicted and actual prices using XChart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Actual vs Predicted Prices")
                .xAxisTitle("Time")
                .yAxisTitle("Price")
                .build();

        // Add series to the chart
        chart.addSeries("Actual Prices", y_actual_array);
        chart.addSeries("Predicted Prices", y_pred_array);

        // Adjust the line thickness
        chart.getStyler().setSeriesLines(new BasicStroke[] {
                new BasicStroke(1.0f), // Adjust the thickness as desired
                new BasicStroke(1.0f)
        });

        // Alternatively, adjust each series individually
        chart.getSeriesMap().get("Actual Prices").setLineStyle(new BasicStroke(1.0f));
        chart.getSeriesMap().get("Predicted Prices").setLineStyle(new BasicStroke(1.0f));

        // Display the chart
        new SwingWrapper<>(chart).displayChart();
    }
}