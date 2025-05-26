<mark>**Note:** Please see notebook for modeling approach, development, and justification

Resources:
* [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/docs/getting-started/introduction.html)
* [yfinance](https://pypi.org/project/yfinance/)
* [ray tune](https://docs.ray.io/en/latest/tune/index.html)

</br>

___

</br>

### NHITS and LSTM Performance per Stock

![image](https://github.com/user-attachments/assets/07b745c3-6483-4768-bab4-59caa17a9b29)

For most of the stocks, the NHITS model is—on average—off by less than $5, with only three tickers with a higher average error. Similarly, the LSTM is usually off by less than $10 dollars, with only two tickers with a higher average error. The MAPE represents the predictions' average percentage difference from the actual value; a score less than 10% is generally considered acceptable to good. Analyzing the percentages for each model, we can see that both models excelled when predicting JPM (off by 4-5%) and RTX (off by 2-3%), while struggling with INTC (off by 9%) and AMD (off by 6%) the most.

#### Plots of Forecasts for Each Individual Stock

![image](https://github.com/user-attachments/assets/274d3608-67e7-42fb-841c-c3123710716c)

</br>

### NHITS and LSTM Performance for Portfolio

In addition to predicting each stock, I also want to compute the predicted total value of my simulated portfolio. With both models' outputs, I first calculate how many shares of each stock I initially "bought" at the start of 2019. Then, I pivot the forecasts from the `NeuralForecast` long format—which consists of each stock ticker, the closing dates, the NHITS forecasted prices, and the LSTM forecasted prices—back into the original wide-format. Finally, to calculate the final porfolio value, I multiply the number of shares for each stock by the last price. I can calculate forecasted profit by subtracting the cumulation of predicted prices by the initial investment, $100,000.

```
NHITS Portfolio MAE: $13,630.83
NHITS Portfolio MAPE: 0.0370
LSTM  Portfolio MAE: $14,265.63
LSTM Portfolio MAPE: 0.0389

Final Portfolio Values:
Actual: $415,992.42
NHITS: $365,356.78
LSTM : $387,639.02
```

With MAPEs slightly below 0.04 or predictions off by only 3-4% on average from the true values, I believe that both models are fairly accurate. As expected, the NHITS model performs slightly better in terms of minimizing loss. While the LSTM final value is technically closer to the actual final value, I believe that this is not as important of an indicator compared to the MAPE or MAE. Furthermore, this is just one point-in-time value, so the LSTM model will not always be closer to the actual. While the absolute error is upwards of $14,000, I believe that this amount is fairly negligent considering my initial investment is $100,000 and the final output is nearly four times as much.  

#### Plot of Overall Portfolio Value Forecast Accuracy

![image](https://github.com/user-attachments/assets/35295be7-cb7f-4b43-8c45-886043a0d986)

</br>

### Potential Improvements

1. By only using neural forecasting models (LSTM and NHITS), this project is fairly limited in scope. In future iterations, I would like to explore the different types of time series models offered in the `NeuralForecast`, `MLForecast`, and `StatsForecast` packages. By incorporating contemporary machine learning models—such as Temporal Fusion Transformers (TFT) and Autoformer—alongside the traditional multivariate statistical models—such as Vector Autoregression and Multiple Regression—I can easily juxtapose each models' benefits or drawbacks. TFTs in particular are extremely popular right now and have demonstrated extremely high accuracy. 

   Furthermore, I can leverage external models to forecast a specific variable, such as volatility or volume, and implement them as an exogenous variable within `futr_exog_list`. For example, by leveraging the `StatsForecast` package, I am able to combine my current models with predictions from ARCH or GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models. These are ideal when forecasting non-constant volatility over time and computing risk estimations. GARCH assumes that the variance of the error term follows an autoregressive moving average (ARMA) process, allowing it to model how the volatility changes based on past information—most notably squared residuals (past price changes) and previous volatility estimates. As such, I can employ a GARCH model to predict each asset's conditional volatility, before feeding the outputs into my NHITS model as an input feature. 

2. Expanding on the benefits of using both NHITS and GARCH, I seek to utilize these models for portfolio optimization and risk management methods. As mentioned earlier, GARCH can be used to forecast volatility, while NHITS can be used to generate expected prices or returns ($\mu$) for each asset in my portfolio. However, in this context, GARCH is severly limited as it only works with a univariate time series (a single stock). Building upon the GARCH model, I can use DCC-GARCH (Dynamic Conditional Correlation GARCH), which models how volatilities and correlations between assets change over time. The DCC-GARCH will:

    >1. Compute each stock's conditional variance at time $t$ using a univariate GARCH
    >2. Standardize the residuals (shock in returns) from each GARCH model
    >3. Employ the Dynamic Conditional Correlation (DCC) process to construct the covariance matrix of standardized residuals.
   
   The covariance matrix is then converted into a time-varying correlaton matrix. Note that the GARCH residuals do not use NHITS forecasts as expected returns but GARCH’s own fitted mean. After wrapping the correlation matrix with two diagnonal matrices consisting of the univariate GARCH ouputs—which are the conditional standard deviations or volatilities for each stock I have calculated in Step A—I have the full conditional covariance matrix ($\Sigma$). This matrix accounts for how correlations between assets change over time. Thankfully, there are Python packages that simplify this process. 

   > **Note**: The covariance matrix of asset returns can also be calculated using historical prices, which only requires calculating the log returns for each stock and computing the covariance matrix. Because this method relies on stationary data, it is only well-suited for extremely stable markets and is not as accurate as using GARCH. However, this is a lot easier and simpler to implement, requiring less computational resources and time.

   Finally, dividing the expected portfolio returns from my NHITS model ($\mu$) by the covariance matrix of asset returns ($\Sigma$), I can calculate the individual asset weights that maximize the Sharpe ratio. Assuming it exists, I can also subtract the portfolio returns by the risk-free rate. This process computes the optimal portfolio weights for each stock depending on the model forecasts. Thus, it is extremely important the the model parameters are up-to-date.

3. Because historical stock data is frequently revised and updated retroactively—accounting for splits, dividends, and corrections—the training data quality often changes day-to-day, affecting hyperparameter tuned and feature engineered variables. As such, for fundamental-based strategies, portfolio managers tend to re-tune weekly or monthly to ensure parameters are up-to-date while preserving computational costs. For a real-life scenario, I believe frequent training and incorporating a model drift detection mechanism is necessary, such as monitoring performance metrics day-to-day to ensure the models are within a desired error range. 
