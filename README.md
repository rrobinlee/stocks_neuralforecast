## NHITS and LSTM Performance for Each Individual Stock

![image](https://github.com/user-attachments/assets/07b745c3-6483-4768-bab4-59caa17a9b29)

#### Plots of Forecasts for Each Individual Stock

![image](https://github.com/user-attachments/assets/274d3608-67e7-42fb-841c-c3123710716c)

</br>

## NHITS and LSTM Performance for Portfolio

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

