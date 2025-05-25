# Predicting a Stock Portfolio using NeuralForecast

In this project, I seek to leverage historical stock market data to construct a small portfolio and predict its cumulative value over the course of a couple months. In other words, rather than forecasting an individual stock, I aim to build and compare two deep learning models to forecast 9 stocks in 3 unique industries: defense, banking, and semiconductor manufacturers. By incorporating a base investment amount and corresponding weights, my ultimate objective is to:
1. Produce individual forecasts for each stock
2. Compute an estimated final portfolio value

After constructing the models, I will backtest through 12 windows—each consisting of a pre-defined number of days—before refitting after each window. To measure accuracy and performance, I will compute the Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) of each individual asset as well as the total portfolio.

To accomplish these objectives, I select a **Long Short-Term Memory** (LSTM) Recurrent Neural Network and a **Neural Hierarchical Interpolation for Time Series** (NHITS). 

* **Long Short-Term Memory (LSTM) Recurrent Neural Network**

  LSTMs are an extension of Recurrent Neural Networks (RNN) that learn patterens over a time sequence. RNNs use recurrent loops to remember information from previous steps in the sequence. According to the documentation on [Nixtla](https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html),  this model improves the exploding and vanishing gradient problem we often face when using simple RNNs by incorporating gate layers. The model will remember long-term dependencies using "cell states" and handle the sequence step-by-step, giving it the ability to retain and discard information. For example, the "forget gate" controls the flow of information and prevents any gradients from vanishing during back propagation. However, LSTMs are slow to train and extremely sensitive to hyperparameters. Furthermore, due to their complexity, these models tend to overfit. 

* **Neural Hierarchical Interpolation for Time Series (NHITS)**

  NHITS builds upon the Neural Basis Expansion Analysis (NBEATS) model, an  MLP-based deep neural network specifically for time series forecasting. According to the documentation on [Nixtla](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html), this model is a feed-forward (no recurrence) deep learning model with hierarchical blocks. Unlike RNN/LSTM models, the NHITS model processes historical data all at once, rather than going step-by-step and looping through time. The NHITS model will break the forecasts into smaller chunks (blocks), before learning and refining the chunks using interpolation. Each block consists of input projection, MLP layers, and output projection. The model uses interpolation to focus on "important" parts of historical data, picking key points before forecasting. Instead of predicting all future values at once, NHITS creates smaller, layered models that make up parts of the forecast. Each block will combine to create the final output. As such, NHITS is better suited for the volatility we expect to see in long-horizon stock forecasting. Because it does not use sequential updates (no recurrence), NHITS are slightly faster to train. However, both methods are still computationally expensive.

### Stock Data

Using Yahoo Finance's `yfinance` package, I have imported the following stocks: J.P. Morgan (JPM), Wells Fargo (WFC), Bank of America (BAC), BAE Systems (BAESY), Leonardo DRS (DRS), RTX Corporation (RTX), Intel (INTC), Advanced Micro Devices (AMD), and Taiwan Semiconductor Manufacturing (TSM). 

```
# Initialize variables
tickers = ['JPM', 'WFC', 'BAC', 'BAESY', 'DRS', 'RTX', 'INTC', 'AMD', 'TSM']
# starting balance
initial_investment = 100000
# investment weights
weights = np.array([0.2, 0.05, 0.05, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1])
# look-ahead days
forecast_horizon = 14
# number of look-ahead windows
n_windows = 14
```

Investment weights are arbituarily selected. In order to incorporate the effects of COVID-19 within the models, I set the start-date to January 1, 2019. The end date is the most recent date as of working on this project.
I select a forecast horizon of 14 days and a look-ahead of 14 windows. These have been arbituarily selected as well, with the horizon representing just over two weeks of stock prices and the look-ahead window representing about half a year. I have also read in S&P 500 closing prices as a broad measure of the overall U.S. stock market. I will not be predicting the S&P 500, but rather, utilizing it as an exogenous variable. 

![image](https://github.com/user-attachments/assets/2e8a420a-5e11-40f3-8466-928da6d5cd63)

**Note:** `yfinance` pulls the latest data from Yahoo Finance, which updates and corrects prices day-to-day—this can be due to splits, dividends, or data corrections that retroactively alter historical prices. Furthermore, closing data for the most recent days are often revised after market close. Thus, I will download the most recent data as of 2025-05-20 locally and use that to tune/train. If I read the same data a week later, the prices may be slightly different even if my end date is still 2025-05-20. This can cause significant issues with hyperparameter tuning, especially since both models are fairly sensitive. 


### Data Transformation and Feature Engineering

The input to NeuralForecast is always a data frame in long format with three columns: `unique_id`, `ds` and `y`:

* The `unique_id` (string, int or category) represents an identifier for the series.
* The `ds` (datestamp or int) column should be either an integer indexing time or a datestampe, ideally like `YYYY-MM-DD` for a date or `YYYY-MM-DD HH:MM:SS` for a timestamp.
* The `y` (numeric) represents the measurement we wish to forecast.


**Historical Exogenous Variables**

From the [NIXTLA documentation](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html):
> This time-dependent exogenous variable is restricted to past observed values. Its predictive power depends on Granger-causality, as its past values can provide significant information about future values of the target variable
y.

* `is_covid`: flags all dates that are within the COVID-19 market crash window. This can help the model identify market regime changes similar to those during the pandemic. More specifically, I want the model to learn the relationship between variables during similar outlier events, improving generalization for future periods of economic downturn and subsequent recovery.
* `dow_sin`, `dow_sin`: Since market behavior can depend on the day (beginning vs. end of the week), I encode the day of the week using cyclical encoding (sin and cos), which preserves weekday continuity. If I had used `day_of_week` itself, the model will treat Sunday (6) and Monday (0) as far apart, rather than side-by-side. Neural networks can more easily learn the impact of weekly seasonality when given these cyclical features.
* `f'volatility_{forecast_horizon}d'`: In order to avoid "Look-Ahead Bias", I calculate each stock's volatility using only the lagged returns (`return_1d`), which prevents accidental leakage of future information. Furthermore, I have made sure to align the window with the pre-defined forecast horizon. This variable allows the models to identify whether high-risk periods (volatility) correlate with specific price patterns, and adjust predictions based on market conditions.
* The 7-day rolling average (`rolling_mean_7`) helps our models understand whether each series is currently trending up or down, which can improve short-term forecasts. This process smoothens any short-term fluctuations, uncovering the overall trend in the past week. The 7-day (`rolling_std_7`) and 14-day (`rolling_std_14`) rolling standard deviations measure the recent volatility in the past week and past 2 weeks, allowing the model to detect periods of high or low uncertainty.
* I have implemented 1-day, 2-day and 5-day lagged returns for the models to detect short-term and medium-term dependencies for each stock, capturing autocorrelation and other temporal patterns within the data.
* By including the S&P 500’s daily returns (`sp500_return_1d`), the models can learn how each stock moves in relation to the market. This is known as a stock's “beta exposure”, or sensitivity to market movements.
* The 14-day rolling volatility of the S&P 500 (`sp500_rolling_vol_14d`) allows our models to quantify the overall market risk or uncertainty. Periods of high overall market volatility typically coincide with large moves in individual stocks. By including this feature, I seek to adjust each model's forecasts depending on market regime.

**Static Exogenous Variables**

From the [NIXTLA documentation](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html):
> The static exogenous variables carry time-invariant information for each time series. When the model is built with global parameters to forecast multiple time series, these variables allow sharing information within groups of time series with similar static variable levels. Examples of static variables include designators such as identifiers of regions, groups of products, etc.

* `is_def`,`is_bank`, `is_chip`: The models can use these columns to learn industry-specific patterns.
* 
![image](https://github.com/user-attachments/assets/117b5a1f-48da-42f6-9ef5-302d890797a5)

**Train, Validation, Test Split**

I split the dataset into training, validation and testing segments. Rather than segment the data by percentages—such as 70%, 15%, and 15%—I determine the size of my validation and test splits by multiplying the forecast horizon (`forecast_horizon`) by the number of windows (`n_windows`). I have decided to do this, because the cross-validation function in `NeuralForecast` uses this product to determine what size to backtest the model. I do not want to set a percentage split, only for the backtesting to begin signficantly after the end of the validation dataset (which is where I train my model up to). This also makes the program more dynamic.

I will build the models using the train and validation datasets, before testings its accuracy on the unseen test data.

### Modeling Approach

Because of computational limitations and time constraints, I have split the model development process into two steps:

1. In order to hyperparameter tune the models, I have leveraged `NeuralForecast`'s AutoNHITS and AutoLSTM models, which build on top of [ray](https://docs.ray.io/en/latest/index.html)—a flexible distributed computing framework for accelerating ML workloads. With a wide variety of hyperparameter optimization tools to choose from in raytune, I have decided to use [HyperOptSearch](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.hyperopt.HyperOptSearch.html), which implements Bayesian optimization algorithms (Tree-structured Parzen Estimator) to effectively handle complex, high-dimensional, and conditional hyperparameter spaces. As such, the Auto models are much more user-friendly when tuning compared to the regular models, reducing computational expense.

    In both cases, I select the default configuration in order to simplify the process, though I have also included several extra parameters to tune (such as gradient clipping and dropout) to mitigate overfitting/underfitting. The default search space dictionary can be accessed through the `get_default_config` function of the Auto model. Finally, I use Mean Absolute Error (MAE) because it is easy to interpret and robust to outliers. This is crucial, because I am forecasting a dollar amount and the stock market tends to fluctuate frequently, creating outliers.

   **Note:** When I ran the hyperparameter tuning, I had not yet created the exogenous variables, so we can interpret the tuned-models as our "baseline" to build upon. With further time and computational resources, I seek to run the tuning again with these new parameters.
   
2. After tuning, I will retrieve both models' hyperparameters, properly formatting them before printing out each tuned value. Because I do not want to rerun the hyperparameter tuning to get the best values every time I work on this project, I manually input each value into a separate "final" model. This is acceptable, because I have already saved the dataset as of 2025-05-20. As mentioned in the Exploratory Data Analysis, data after 2025-05-20 may appear different due to revisions and updates, necessitating a new set of tuned parameters.

### Final Models

For both models, I use `scaler_type='standard'` to ensure inputs have zero mean and unit variance, which is useful when values vary in magnitude. Because I select a fairly high `max_steps` of 1000, I also incorporate validation checks every 50 steps to prevent overfitting, and I stop training if the validation loss remains constant for 200 steps. I use a relatively large `input_size`, `batch_size`, and `windows_batch_size` to improve model generalization. I select a learning_rate that is lower than the tuned value to ensure stable convergence, sacrificing some compute speed as a result. Finally, I implement `gradient_clip_val` to mitigate exploding gradients, helping stabilize the deep training process.

* For **NHITS**, I have selected and tuned these additional model-specific parameters:
    * `n_pool_kernel_size` controls the temporal resolution or hierarchical downsampling process of each block in the NHITS hierarchy. By using (2,2,2), I employ gradual pooling such that the model learns from local and global patterns without overly aggressive downsampling. NHITS will only aggregate every 2 time steps within each block, which can help model multi-scale patterns. In other words, block 1 will see 1/2 the data points, block 2 will see 1/4, and block 3 will see 1/8. This helps prevent overfitting by reducing the amount of "detail" in the input.
    * This corresponds well with an `n_freq_downsample` of (24, 12, 1), which controls the number of basis functions or frequency components within each hierarchical block. The first block uses 24 functions (broad seasonal patterns), the second uses 12 (refined details), and the final block uses just 1 function (final trend). NHITS predicts future values by estimating the basis coefficients—called theta—for each block, then reconstructing the output using basis functions. The basis functions can be Fourier, polynomial, sinusoid, etc. to fit data signals or trends—these are added together to form each window's prediction.
    * I utilize `dropout_prob_theta` to randomly zero out the theta layers or basis coefficients, improving regularization. Finally, the `'nearest'` interpolation method keeps values close to the original sampled values when reconstructing the full forecast.
 
* For **LSTM**, I have selected and tuned these additional model-specific parameters:
    * `encoder_hidden_size` represents the number of units within each LSTM cell of the encoder. Given the size of the dataset, I have selected a relatively high value to capture long-term dependencies within the sequence. Furthermore, I stack two LSTM layers using `encoder_n_layers` for the model to learn hierarchical features and identify patterns or trends at different levels.
    * Because my LSTM has a high tendency to overfit the training data—likely due to the two large encoder layers—I am forced to include a high `encoder_dropout` rate for strong regularization in between the LSTM layers. This will shut off about 50% of connections between LSTM layers. I further reduce overfitting by applying a smaller  LSTM decoder (`decoder_hidden_size`).

### NHITS and LSTM Performance for Each Individual Stock

![image](https://github.com/user-attachments/assets/07b745c3-6483-4768-bab4-59caa17a9b29)

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

