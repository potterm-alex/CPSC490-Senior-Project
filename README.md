# Yale Computer Science - CPSC490 - Senior Project
##### Alex Potter, Yale '24 - Advised by Stephen Slade

Welcome! 
This is my senior project that I completed for my BS in Computer Science whilst finishing my last semester at Yale. 

Intro:

This project was to create an LSTM based stock price prediction model. Some important things to know are:
1. It uses a single layer LSTM model, with 48 neurons in the hidden layer. 
2. Adam optimization
3. MSE loss
4. Keras provided the backbone for the model setup, metrics etc.
5. We used numpy and pandas for our data handling
6. Pyplot was used for plotting
7. We also used sklearn's MixMaxScaler for data scaling

Data:

The data used for this project came from FirstRateData.com, a data brokerage service. I bought their Stocks Complete (7000+ Tickers)
bundle, which can be found [here](https://firstratedata.com/b/22/stock-complete-historical-intraday). I then used Nasdaq's stock screener to narrow down which stocks I used for training of the model. 

Use Breakdown:

To use the model, 
