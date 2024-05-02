# Yale Computer Science - CPSC490 - Senior Project
### Alex Potter, Yale '24 - Advised by Stephen Slade

Welcome! 
This is my senior project that I completed for my BS in Computer Science whilst finishing my last semester at Yale. 

### Intro:

This project was to create an LSTM based stock price prediction model. Some important things to know are:
1. It uses a single layer LSTM model, with 48 neurons in the hidden layer. 
2. Adam optimization
3. MSE loss
4. Keras provided the backbone for the model setup, metrics etc.
5. We used numpy and pandas for our data handling
6. Pyplot was used for plotting
7. We also used sklearn's MixMaxScaler for data scaling

### Data:

The data used for this project came from FirstRateData.com, a data brokerage service. I bought their Stocks Complete (7000+ Tickers)
bundle, which can be found [here](https://firstratedata.com/b/22/stock-complete-historical-intraday). 

To use the data in my model, we recommend creating a /Data folder, and then separating all tickers into folders alphabetically. 
The current process data function is set up to work with it in that way.

I then used Nasdaq's stock screener to narrow down which stocks I used for training of the model.

### Use of Model:

The model takes in high, low, opening price, and volume every minute of a stock's price, and predicts 1 min ahead what the price will be.
It has achieved <$1 MSE across most tests live.

#### [lstm.py](lstm.py)

This file contains all of the functions involved with the functionality of the model. It includes training functions that work on 
the chunk method I created, data processing from the files we have, and contains the lstm model class that I created, which will 
be the container that saves and loads the models that we have stored.

#### [chunks.py](chunks.py)

This file contains the functionality for the ticker lists that I used to train the model. The functions will take any list of tickers,
convert them into chunks that can be trained on, and then return this to the functions in [lstm.py](lstm.py) which will train the model.

#### [live.py](live.py)

This file is the main workhorse if you want to use my model as is for stock price prediction. It contains the live_trade() function
which will actively take in new data from [AlphaVantage's](https://www.alphavantage.co/) API and make live stock predictions. It also
less importantly contains the pseudo_test() function which I used to test the model's live predicting ability as though it was live. 

### Future Extensions

1. Multi-layered lstm
2. Regularization of the layers
3. Further ticker training on more volatile stocks
4. Better live training protocols
