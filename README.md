### NHITS and LSTM Performance of Each Individual Stock

![image](https://github.com/user-attachments/assets/07b745c3-6483-4768-bab4-59caa17a9b29)


Comparing the MAE between both models, we can see that NHINTS performs signficantly better than LSTM, with lower overall loss across all tickers. For most of the stocks, the NHITS model is—on average—off by less than $5, with only three tickers with a higher average error. Similarly, the LSTM is usually off by less than $10 dollars, with only two tickers with a higher average error. 

The MAPE represents the predictions' average percentage difference from the actual value; a score less than 10% is generally considered acceptable to good. Analyzing the percentages for each model, we can see that both models excelled when predicting JPM (off by 4-5%) and RTX (off by 2-3%), while struggling with INTC (off by 9%) and AMD (off by 6%) the most. Finally, comparing the actual dollar value with the two forecasted values, we see that both models tend to underestimate the final prices.

#### Plots of Forecasts for Each Individual Stock

![image](https://github.com/user-attachments/assets/274d3608-67e7-42fb-841c-c3123710716c)

</br>

### NHITS and LSTM Performance of Overall Portfolio

* NHITS Portfolio MAE: $13,630.83
* NHITS Portfolio MAPE: 0.0370
* LSTM  Portfolio MAE: $14,265.63
* LSTM Portfolio MAPE: 0.0389

* Final Portfolio Values:
  * Actual: $415,992.42
  * NHITS: $365,356.78
  * LSTM : $387,639.02

With MAPEs slightly below 0.04 or predictions off by only 3-4% on average from the true values, I believe that both models are fairly accurate. As expected, the NHITS model performs slightly better in terms of minimizing loss. While the LSTM final value is technically closer to the actual final value, I believe that this is not as important of an indicator compared to the MAPE or MAE. Furthermore, this is just one point-in-time value, so the LSTM model will not always be closer to the actual. While the absolute error is upwards of $14,000, I believe that this amount is fairly negligent considering my initial investment is $100,000 and the final output is nearly four times as much.  

#### Plot of Overall Portfolio Value Forecast Accuracy

![image](https://github.com/user-attachments/assets/35295be7-cb7f-4b43-8c45-886043a0d986)

