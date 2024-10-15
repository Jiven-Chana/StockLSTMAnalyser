# Stock LSTM Analyzer

A Java-based tool leveraging Long Short-Term Memory (LSTM) neural networks (using Deeplearning4j) to predict and smooth stock prices of the S&P 500 Index (SPX). This project aims to provide a solid basis for representing bigger underlying trends in the financial market by mitigating the volatility inherent in stock data.

---

## Table of Contents
- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Motivation
The stock market is inherently volatile, making it challenging to discern long-term trends from short-term fluctuations. This volatility can obscure the true underlying patterns, leading to difficulties in making informed investment decisions. Traders and analysts often struggle to separate noise from meaningful signals, which can result in suboptimal trading strategies and increased risk exposure.

This project seeks to smooth out the volatile data of the S&P 500 Index (SPX), providing a clearer representation of underlying market trends. By applying advanced data preprocessing techniques, including the integration of technical indicators such as Relative Strength Index (RSI) and Exponential Moving Average (EMA), as well as normalization methods like MinMax scaling, we aim to enhance the quality and consistency of the input data. This refined data serves as a more reliable foundation for training the LSTM neural networks, thereby improving the accuracy and robustness of the stock price predictions.

By utilizing Long Short-Term Memory (LSTM) neural networks, which are well-suited for time-series forecasting, we aim to predict stock prices more accurately and provide insights into market dynamics. LSTMs are capable of capturing temporal dependencies and long-term relationships within sequential data, making them ideal for modeling the complex and non-linear patterns present in financial markets. Our approach leverages the strengths of LSTM architectures to model the intricate behaviors of stock prices, thereby enhancing prediction accuracy and offering deeper insights into market movements and trends.

This project serves as a foundation for further exploration into complex data relationships. Understanding the multifaceted interactions within financial data is crucial for developing robust predictive models. By establishing a solid baseline with LSTM-based forecasting, we are poised to explore more sophisticated modeling techniques that can capture additional layers of complexity in the data, such as sentiment analysis from news articles or macroeconomic indicators.

Future enhancements include integrating Kalman filters to capture even more nuanced patterns in the data. Kalman filters are powerful tools for estimating the state of a dynamic system from noisy observations, and their integration with neural networks can provide a hybrid approach that leverages the strengths of both statistical and machine learning methods. This combination is expected to improve the model's ability to filter out noise, adapt to changing market conditions, and provide more precise and reliable predictions. Additionally, we plan to explore other advanced techniques, such as attention mechanisms and ensemble models, to further refine our predictive capabilities and better understand the underlying drivers of stock market behavior.


---

## Features
- **Data Preprocessing**: Collects historical stock data and applies technical indicators such as Relative Strength Index (RSI) and Exponential Moving Average (EMA). Handles missing values and normalizes data using MinMax scaling.
- **LSTM Neural Network**: Implements a multi-layer LSTM network with 150 units, optimized using the Adam optimizer and Mean Squared Error (MSE) loss function.
- **Train-Validation Split**: Splits the dataset into training and validation sets (90% training, 10% validation) to evaluate model performance on unseen data.
- **Visualization**: Provides real-time plots comparing predicted prices against actual prices, demonstrating the model’s effectiveness in smoothing and predicting stock trends.
- **Modular Design**: Structured codebase allowing easy integration of additional features like Kalman filters for future development.

---

## Installation

```bash
git clone https://github.com/Jiven-Chana/StockLSTMAnalyser.git
cd StockLSTMAnalyser
mvn install
```
- Setup your enviorment such that you have installed Java Development Kit (JDK) 8 or higher (preferrably 17) and installed IntelliJ IDEA or your preferred Java IDE

---

## Usage 
- You can run the LSTMEquityModel class from your IDE or use the command line:
  ```bash
  mvn exec:java -Dexec.mainClass="org.example.LSTMEquityModel"
  ```
- You can customise the parameters
- 	Modify the ```numFeatures```, ```numOutputs```, and ```numEpochs``` variables in ```LSTMEquityModel.java``` to tune the model.
-   Adjust the technical indicators in EquityAnalysis.java to experiment with different preprocessing techniques.

---

## Results 
- The model effectively smooths out the volatile stock price data, providing a clearer view of the underlying trends in the SPX. By comparing the predicted prices against the actual prices, you can observe the model’s capability in forecasting and trend analysis

-  **Effective Data Smoothing:**  
  The model successfully smooths out the inherent volatility of the S&P 500 Index (SPX) stock price data, providing a more lucid view of the underlying market trends. By mitigating short-term fluctuations, the LSTM neural network highlights long-term patterns, making it easier to identify and analyze significant market movements.

- **Accurate Forecasting and Trend Analysis:**  
  Comparing the predicted stock prices against the actual prices showcases the model’s robust forecasting capabilities. The alignment between predicted and actual values demonstrates the LSTM’s proficiency in capturing temporal dependencies and trend dynamics, validating its effectiveness in time-series forecasting for financial data.

- **Long-Term Prediction Focus:**  
  Designed primarily for long-term predictions, the model intentionally disregards short-term volatility and external factors that can introduce noise into the data. This focus ensures that the forecasts are more stable and reliable, providing a solid foundation for strategic investment decisions based on sustained market trends.

<img width="1703" alt="Screenshot 2024-10-14 at 22 11 03" src="https://github.com/user-attachments/assets/a05e4257-8c2c-47d1-a600-f2e7331389c4">

- Below I have attached an image showcasing the true scope of the project. This implementation used a kalman filter in substitution to the EMA and RSI combination used above. Here I also unscaled the predicted prices and mapped them to the original prices given by the input csv file. In this instance input parameters were tweaked effectively to ensure minimal underfitting and more sensitivity for the LSTM model to fit the shorter term movements of price. I have no doubt that, upon further refinement, the model would be suited for smaller timeframe analysis. 

<img width="1517" alt="Screenshot 2024-10-15 at 01 22 00" src="https://github.com/user-attachments/assets/cb15e14a-f1cf-4407-ac12-dfa9277dad0f">


---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any enhancements or bug fixes.

---

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the LICENSE file for details.

---

## References
To understand the methodologies and technologies used in this project, you may refer to the following academic resources:

1. **Long Short-Term Memory Networks**
   - *Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.*
     [Link](https://www.bioinf.jku.at/publications/older/2604.pdf)

2. **Adam Optimizer**
   - *Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR).*
     [Link](https://arxiv.org/pdf/1412.6980.pdf)

3. **Mean Squared Error Loss Function**
   - *Zhang, Z. (2017). Introduction to machine learning: k-nearest neighbors and linear regression. Annals of Translational Medicine, 5(7).*
     [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5406162/)

4. **Time-Series Forecasting with Neural Networks**
   - *Fu, T. C. (2011). A review on time series data mining. Engineering Applications of Artificial Intelligence, 24(1), 164-181.*
     [Link](https://www.sciencedirect.com/science/article/pii/S0952197610001437)

5. **Application of LSTM in Stock Market Prediction**
   - *Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.*
     [Link](https://www.sciencedirect.com/science/article/pii/S0377221717302909)

6. **Normalization Techniques**
   - *Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning.*
     [Link](http://proceedings.mlr.press/v37/ioffe15.pdf)

## Contact

For any questions or suggestions, feel free to open an issue or contact me at [your_email@example.com](jiven.chana@icloud.com).

---
