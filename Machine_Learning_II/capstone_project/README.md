# Stock Market Price Forecast capstone project

## Introduction
The Stock Market Price Prediction project aims to leverage historical stock data from Google, S&P 500, and NASDAQ, obtained through the Yahoo Finance API (yfinance), to develop a machine learning model capable of predicting future stock prices. The project is motivated by the desire to explore the predictive capabilities of machine learning algorithms in the financial domain and to assist investors and traders in making informed decisions.

## Key components
### Data Collection
The project utilizes the yfinance library to fetch historical stock data for Google, S&P 500, and NASDAQ. The dataset includes essential features such as opening prices, closing prices, high and low prices, trading volumes, and adjusted close prices.

### Data preprocessing
Before diving into preprocessing, a comprehensive exploration of the dataset is conducted to gain insights into its characteristics. This involves visualizations such as line charts, candlestick charts, and statistical summaries to identify trends, patterns, and potential outliers.
##### Volatility Check

Volatility is a crucial aspect of stock market data. High volatility can lead to unpredictable price movements. To address this:

###### Volatility Calculation:
Compute daily or rolling volatility metrics, such as standard deviation or average true range (ATR), to quantify the degree of price variation.

###### Volatility Thresholding:
Set a threshold to identify and potentially exclude periods of extreme volatility that might distort model training.

##### Stationarity Check

Stationarity is a desirable property for time series data, as non-stationary data can introduce challenges in model training. Key steps include:

###### Visual Inspection:
Examine time series plots to identify trends, seasonality, or irregularities.

###### Statistical Tests:
Conduct formal statistical tests, such as the Augmented Dickey-Fuller (ADF) test, to assess stationarity. If the data is non-stationary, consider differencing or transformations to induce stationarity.


##### Missing Values Handling

Addressing missing values is crucial for ensuring the completeness of the dataset:

###### Imputation:
Employ imputation techniques, such as forward-fill, backward-fill, or mean imputation, to handle missing values.

###### Impact Analysis:
Analyze the impact of missing values on key features and consider excluding or imputing them based on their significance.

## Model development
During the model development stage of stock market price forecasting projects, many time series forecasters such as ARMA (autoregressive moving average), ARIMA (autoregressive integrated moving average), SARIMA (seasonal ARIMA), SARIMAX (SARIMA with exogenous I considered as the model
variable) and the ARCH model (autoregressive conditional heteroscedasticity).
After extensive experiments, we found that the ARIMA(3,1,3) model is the best for predicting stock prices.

## Model evaluation
###### Mean Absolute Error (MAE): 65.7775
Mean Absolute Error (MAE) is a measure of the average absolute difference between the predicted stock price and the actual stock price.
For the ARIMA(3,1,3) model, the MAE is calculated as 65.7775.
This value indicates the average size of error in  stock price predictions, and lower values ​​are better.
MAE assumes that the model's predictions differ from the actual stock price by about 65.7775 units on average.
###### Mean Square Error (MSE): 73.0359
Mean Square Error (MSE) is the average of the squares of the difference between the predicted stock price and the actual stock price.
For the ARIMA(3,1,3) model, the MSE is calculated as 73.0359.
MSE penalizes larger errors  than MAE and provides a more nuanced view of model performance.
In this case,  MSE means that the squared deviation between the predicted price and the actual price is on average  73.0359.
###### Root Mean Square Error (RMSE): 8.5461
Root Mean Square Error (RMSE) is the square root of  MSE and is another metric often used to evaluate prediction accuracy.
The RMSE of our model is 8.
5461, which reflects the average size of the error in predicting stock prices.
A lower RMSE indicates better model performance.
In this case, RMSE assumes an average deviation of approximately 8.
5461 units from the actual stock price.
These evaluation results provide quantitative insights into the performance of the ARIMA(3,1,3) model.

Interpretation of these indicators must be done in the context of the specific requirements and goals of the stock forecasting project.
Further analysis, visual inspection, and possible model improvements are considered to improve the  predictive capabilities of the model.

## Conclusion

The Stock Market Price Prediction project seeks to harness the power of machine learning to forecast stock prices, providing valuable insights for market participants. By addressing challenges and incorporating continuous improvements, the project aims to contribute to the development of effective stock price prediction models.