## Model Evalutation

To evaluate the two models, I will utilize the `NeuralForecast` package's built-in cross validation function. Because the data is sequential, I will backtest through a series of windows, defined in the function as `n_windows`. `step_size` controls the distance between each cross-validation window, so we set it equal to the forecast horizon. By doing so, I can perform chained cross-validation where the windows do not overlap. Emulating real life scenarios, we will retrain our model using new observed data (sequential segments of `test_df`) before making the next set of predictions.

### Rolling Window Backtests with Refitting

Since I am using the `NeuralForecast` package's cross validation function, I need to input the full dataframe; this is because the method is designed to simulate how my model would perform at sequential cutoff dates. The cutoff date is determined by the number windows and the forecast horizon for each window. For each window, the model is trained up to a cutoff date and then forecasts the next `forecast_horizon` steps. The window slides forward by `step_size`, and the process repeats `n_windows` times.

#### MAE for Each Individual Stock

![image](https://github.com/user-attachments/assets/07b745c3-6483-4768-bab4-59caa17a9b29)


Comparing the MAE between both models, we can see that NHINTS performs signficantly better than LSTM, with lower overall loss across all tickers. For most of the stocks, the NHITS model is—on average—off by less than $5, with only three tickers with a higher average error. Similarly, the LSTM is usually off by less than $10 dollars, with only two tickers with a higher average error. 

The MAPE represents the predictions' average percentage difference from the actual value; a score less than 10% is generally considered acceptable to good. Analyzing the percentages for each model, we can see that both models excelled when predicting JPM (off by 4-5%) and RTX (off by 2-3%), while struggling with INTC (off by 9%) and AMD (off by 6%) the most. Finally, comparing the actual dollar value with the two forecasted values, we see that both models tend to underestimate the final prices.

#### Forecasts for Each Individual Stock

![image](https://github.com/user-attachments/assets/274d3608-67e7-42fb-841c-c3123710716c)

Analyzing their performance through each window of the backtest, both the LSTM and NHITS models struggle to predict significant spikes or valleys. This is expected, because the model cannot see into the future and anticipate significant socioeconomic or market-moving events. Starting from February 2025, both models continue to predict a stable or upward trend at each window despite the actual prices decreasing (see JPM, WFC, BAC, TSM, and AMD). Thus, we see a step-like pattern—the models only "sees" a stock has actually decreased at the next forecast window, causing the model to rapidly adjust its forecast to match the last-known true price and creating a jump. This has occured the most for bank stocks, since they are historically quite stable. Having learned this stable pattern, the models frequently predict low movement at each window. Similarly, for stocks that spiked suddenly (such as BAESY and INTC), the models fail to predict a significant jump. Finally, we see that volatile tickers—most notably INTC—are the most challenging to predict, as expected.

### Computing Overall Portfolio Value

In addition to predicting each stock, I also want to compute the predicted total value of my simulated portfolio. With both models' outputs, I first calculate how many shares of each stock I initially "bought" at the start of 2019. Then, I pivot the forecasts from the `NeuralForecast` long format—which consists of each stock ticker, the closing dates, the NHITS forecasted prices, and the LSTM forecasted prices—back into the original wide-format. Finally, to calculate the final porfolio value, I multiply the number of shares for each stock by the last price. I can calculate forecasted profit by subtracting the cumulation of predicted prices by the initial investment, $100,000.

#### Overall Portfolio Value Forecast Accuracy

NHITS Portfolio MAE: $13,630.83

NHITS Portfolio MAPE: 0.0370

LSTM  Portfolio MAE: $14,265.63

LSTM Portfolio MAPE: 0.0389

Final Portfolio Values:
* Actual: $415,992.42
* NHITS: $365,356.78
* LSTM : $387,639.02

With MAPEs slightly below 0.04 or predictions off by only 3-4% on average from the true values, I believe that both models are fairly accurate. As expected, the NHITS model performs slightly better in terms of minimizing loss. While the LSTM final value is technically closer to the actual final value, I believe that this is not as important of an indicator compared to the MAPE or MAE. Furthermore, this is just one point-in-time value, so the LSTM model will not always be closer to the actual. While the absolute error is upwards of $14,000, I believe that this amount is fairly negligent considering my initial investment is $100,000 and the final output is nearly four times as much.  

#### Plot of Overall Portfolio Value Forecast Accuracy

![image](https://github.com/user-attachments/assets/35295be7-cb7f-4b43-8c45-886043a0d986)

Because the full portfolio is just an aggregation of the individual stocks earlier, the issues I have already discussed also apply here—the predictions are in a step-like pattern, where the beginning of each window "jumps" to re-oriente with the actual data. While their forecasts appear to roughly follow the shape of the actual portfolio quite well, I believe the NHITS is a bit more stable. Furthermore, I tested both models on different forecast windows, ranging from 7 to 56 days (1-week to 8-weeks). As the window size increases over 14 days, the LSTM suffers significantly while the NHITS MAE and MAPE remain fairly consistent. In future use-cases, I would probably stick with just NHITS.
