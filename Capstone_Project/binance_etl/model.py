# %% [markdown]
# ## <h1>CAPSTONE PROJECT</h1>
# ## <h2>STOCK PRICE PREDICTION</h2>
# 
# <p>
# Welcome to this Jupyter Notebook, where we will explore the art of predicting Google's stock price. Google, part of Alphabet Inc., is a global tech giant, and understanding its stock trends is of great significance. We will employ the tools of data analysis and machine learning to decipher the intricate world of market dynamics. Through systematic analysis, we aim to uncover patterns and insights that will guide us in projecting the future course of Google's stock. Join us as we deconstruct the complexities of stock price prediction in the digital age.
# </p>
# <h2>Problem statement: </h2><p>We all try to make informed and low risk investments.</p>

# %%
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

# %%
#First we need to download the data and put it in a dataframe
#We will require google's data, S&P500 and the NASDAQ as exogenous data
start = '2005-01-01'
end = '2023-10-20'
google_data = yf.download('GOOGL',start=start,end=end)
SnP500_data = yf.download('SPY',start = start,end = end)
NAS_data = yf.download('^IXIC',start = start,end = end)

# %% [markdown]
# We have downloaded the data from yahoofinance and put them in their own dataframes

# %%
print(NAS_data.isnull().sum(),SnP500_data.isnull().sum(),google_data.isnull().sum())

# %% [markdown]
# ## Data cleaning
# Since we do not have any null values, let's get to the part of the data that we need.
# We need to use the closing price to predict our data so we will keep two columns, close and volume column

# %%
google_data.head()

# %%
google = google_data.copy()
snp500 = SnP500_data.copy()
nas = NAS_data.copy()
columns_to_drop = {'Open','High','Low','Adj Close'}
google = google.drop(columns=columns_to_drop)
nas = nas.drop(columns = columns_to_drop)
snp500 = snp500.drop(columns = columns_to_drop)

# %%
google.info()

# %%
# The data is in datetime index which is a good thing

# %% [markdown]
# ## Data analysis

# %%
google.describe()

# %%
google.shape , snp500.shape , nas.shape

# %%
#Lets calculate the correlation
correlation = google.corr()
correlation

# %%
google['Close'].plot()
plt.title("Google daily closing price")
plt.show()

# %%
#Let's add the returns column in all our data
google['returns'] = google['Close'].pct_change()*100
snp500['returns'] = snp500['Close'].pct_change()*100
nas['returns'] = nas['Close'].pct_change()*100


# %%
google['returns'].plot()
plt.title("Google daily returns")
plt.show()

# %% [markdown]
# First we need to fill the null with a value

# %%
google = google.fillna(method='bfill')
snp500 = snp500.fillna(method='bfill')
nas = nas.fillna(method='bfill')

# %% [markdown]
# Lets check for stationarity in our data

# %%
import statsmodels.tsa.stattools as sts

sts.adfuller(google.returns)

# %%
sts.adfuller(snp500.returns)

# %%
sts.adfuller(nas.returns)

# %% [markdown]
# The result of the Augmented Dickey-Fuller (ADF) test:
# 
# Interpreting these results:
# 
# 1. **ADF Statistic**: The ADF Statistic is a test statistic used in the ADF test. This statistic is used to assess whether a time series is stationary or non-stationary. The more negative the ADF Statistic, the stronger the evidence against the null hypothesis (i.e., the data is non-stationary).
# 
# 2. **p-value**: The p-value is a measure of the strength of evidence against the null hypothesis. In this case, the p-value is very close to zero or zero, indicating strong evidence to reject the null hypothesis. A small p-value suggests that the data is likely stationary.
# 
# 3. **Number of Lags Used**: lags were used in the regression when performing the ADF test. The number of lags can vary depending on the test setup.
# 
# 4. **Number of Observations Used**: The ADF test used 4716 observations in your time series data.
# 
# 5. **Critical Values**: These are the critical values for different levels of significance (1%, 5%, and 10%). They are used to compare with the ADF Statistic. If the ADF Statistic is more negative than the critical values, it provides further evidence that the data is stationary. In your case, the ADF Statistic is indeed more negative than all the critical values, further supporting stationarity.
# 
# 6. **Maximized Information Criterion (AIC)**: The AIC is a measure used in model selection. While it's not directly related to the stationarity test, it's sometimes included in the ADF test results. It's used to compare the quality of different models, and lower AIC values indicate a better model fit.
# 
# In summary, the small p-value and the fact that the ADF Statistic is more negative than the critical values indicate strong evidence against the null hypothesis of non-stationarity. This suggests that the time series data in question is likely stationary.

# %%
#Let's create a volatility column
google['volatility'] = google['returns'].rolling(window=5).std()
nas['volatility'] = nas['returns'].rolling(window=5).std()
snp500['volatility'] = snp500['returns'].rolling(window=5).std()

# %%
plt.figure(figsize=(20,10))
google['volatility'].plot()
google['Close'].plot()
plt.title("Google daily volatility against the closing price")
plt.show()

# %%
google.info()

# %% [markdown]
# Let's check for seasonality in our data

# %%
import statsmodels.graphics.tsaplots as sgt
plt.figure(figsize=(20,10))
sgt.plot_acf(google.returns,zero=False,lags = 40)

# %% [markdown]
# We can see by using the returns column, the first 5 lags are significant and can be set as the minimum number of lags used. this was also asserted in the adfuller test

# %%
plt.figure(figsize=(20,10))
sgt.plot_acf(google.Close,lags = 40,zero=False)

# %% [markdown]
# From using the closing prices, we can see that all the previous data is significant in predicting the next value. This is a common characteristic with non-stationary data and this implies that it cannot be used with algorithms such as AR, MA ,ARMA SARMA and such since these are algorithms for stationary data

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

google = google.asfreq('B')
snp500 = snp500.asfreq('B')
nas = nas.asfreq('B')
google = google.fillna(method='ffill')
nas = nas.fillna(method='ffill')
snp500 = snp500.fillna(method='ffill')
plt.figure(figsize=(20,10))
s_dec = seasonal_decompose(google.Close, model="additive")
s_dec.plot()

# %%
s_dec = seasonal_decompose(google.returns, model="additive")
s_dec.plot()

# %%
#Splitting the data
size = int(len(google)*0.8)
train_google ,test_google = google.iloc[:size],google.iloc[size:]
train_nas ,test_nas = nas.iloc[:size],nas.iloc[size:]
train_snp ,test_snp = snp500.iloc[:size],snp500.iloc[size:]

# %% [markdown]
# ## <h1>Modelling</h1>
# 
# From our analysis, we can see that the data we are using is more likely to be non-stationary. So we need to choose algorithms that can handle non-stationary data

# %% [markdown]
# <h1>1: ARIMA</h1>

# %%
google.shape , snp500.shape , nas.shape

# %%
google.head()

# %%
print(google.duplicated().sum())

# %%
google = google.drop_duplicates()
snp500 = snp500.drop_duplicates()
nas = nas.drop_duplicates()

# %%
google.shape , snp500.shape , nas.shape

# %%
train_google.shape,train_snp.shape,test_google.shape,test_snp.shape

# %%
nas = nas.fillna(method='bfill')
snp500 = snp500.fillna(method='bfill')

# %%
#We will perform an LLR test to see which model performs best
from scipy.stats.distributions import chi2
def llr_test(model_1, model_2,df=1):
    """df is the degree of freedom.The Likelihood Ratio Test (LLR) is a statistical test used to compare the goodness of fit of two nested models, typically in the context of maximum likelihood estimation. The primary purpose of the LLR test is to determine whether a more complex model provides a statistically significant improvement in fit compared to a simpler, nested model.
    @:param model_1: the simpler model
    """
    l1 = model_1.fit().llf
    l2 = model_2.fit().llf
    lr = (2*(l2-l1))
    p = chi2.sf(lr,df).round(3)
    return p

# %%
from statsmodels.tsa.arima.model import ARIMA

arima_AR1_MA1_D1 = ARIMA(endog=google.returns, order=(1, 1, 1),exog=snp500.returns)
result = arima_AR1_MA1_D1.fit()
result.summary()

# %%
arima_AR2_MA2_D1 = ARIMA(endog=google.returns, order=(2, 1, 2),exog=snp500.returns)
result_ar2_ma2 = arima_AR2_MA2_D1.fit()
result.summary()

# %%
arima_AR3_MA3_D1 = ARIMA(endog=google.returns, order=(3,1,3),exog=snp500.returns)
result_ar3_ma3 = arima_AR3_MA3_D1.fit()
result_ar3_ma3.summary()

# %%
arima_AR4_MA4_D1 = ARIMA(endog=google.returns, order=(4,1,4),exog=snp500.returns)
result = arima_AR4_MA4_D1.fit()
result.summary()

# %%
arima_AR5_MA5_D1 = ARIMA(endog=google.returns, order=(5, 1, 5),exog=snp500.returns)
result = arima_AR5_MA5_D1.fit()
result.summary()

# %%
arima_AR6_MA6_D1 = ARIMA(endog=google.returns, order=(6, 1, 6),exog=snp500.returns)
result = arima_AR6_MA6_D1.fit()
result.summary()

# %%
arima_AR7_MA7_D1 = ARIMA(endog=google.returns, order=(7, 1, 7),exog=snp500.returns)
result = arima_AR7_MA7_D1.fit()
result.summary()

# %%
arima_AR3_MA7_D1 = ARIMA(endog=google.returns, order=(3, 1, 7),exog=snp500.returns)
result = arima_AR3_MA7_D1.fit()
result.summary()

# %%
arima_AR3_MA8_D1 = ARIMA(endog=google.returns, order=(3, 1, 8),exog=snp500.returns)
result = arima_AR3_MA8_D1.fit()
result.summary()

# %%
arima_AR3_MA4_D1 = ARIMA(endog=google.returns, order=(3, 1, 4),exog=snp500.returns)
result = arima_AR3_MA4_D1.fit()
result.summary()

# %%
arima_AR3_MA5_D1 = ARIMA(endog=google.returns, order=(3, 1, 5),exog=snp500.returns)
result = arima_AR3_MA5_D1.fit()
result.summary()

# %%
arima_AR2_MA4_D1 = ARIMA(endog=google.returns, order=(2, 1, 4),exog=snp500.returns)
result = arima_AR2_MA4_D1.fit()
result.summary()

# %%
llr_test(arima_AR3_MA5_D1,arima_AR3_MA4_D1)

# %%
llr_test(arima_AR3_MA3_D1,arima_AR3_MA4_D1)

# %%
llr_test(arima_AR5_MA5_D1,arima_AR6_MA6_D1)

# %%
llr_test(arima_AR4_MA4_D1,arima_AR5_MA5_D1)

# %%
llr_test(arima_AR3_MA3_D1,arima_AR2_MA4_D1,1)

# %% [markdown]
# We can conclude that arima 3, 3 is the best model for this dataset that is if we include the returns from snp500

# %% [markdown]
# <h1>2: Auto Arima</h1>

# %%
from pmdarima.arima import auto_arima

model = auto_arima(y=google.returns,seasonal=True,stepwise=True,trace=True)
model.summary()

# %% [markdown]
# ARIMA performs better than auto arima

# %% [markdown]
# <h1>3: GARCH AND ARCH</h1>

# %%
from arch import arch_model

# %%
arch_mod_1 = arch_model(google.returns,x=snp500.returns,vol='GARCH',p=1,q=1)
result = arch_mod_1.fit()
result.summary()

# %%
arch_mod_2 = arch_model(google.returns,x=snp500.returns,vol='GARCH',p=2,q=2)
result = arch_mod_2.fit()
result.summary()

# %%
arch_mod_3 = arch_model(google.returns,x=snp500.returns,vol='GARCH',p=1,q=2)
result = arch_mod_3.fit()
result.summary()

# %% [markdown]
# <h1>Prediction</h1>

# %%
start_date = pd.to_datetime('2023-08-21')
end_date = pd.to_datetime('2023-10-19')
forecast = result_ar3_ma3.predict(start=start_date,end=end_date)
forecast.plot()

# %%
actual_returns = google['returns'].loc[start_date:end_date]
plt.figure(figsize=(10, 5))
actual_returns.plot(label='Actual Returns', color='blue')
forecast.plot(label='Forecasted Returns', color='red')
plt.legend()
plt.title('ARIMA Model Forecast vs. Actual Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.show()

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Assuming 'actual' and 'forecast' are the actual values and model predictions
mae = mean_absolute_error(actual_returns, forecast)*100
mse = mean_squared_error(actual_returns, forecast)*100
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# %% [markdown]
# We can safely say that the ARIMA(3,1,3) is a better forecaster for returns and can be used in real time market


