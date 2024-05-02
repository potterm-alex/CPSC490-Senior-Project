# importing stuff we need for the funcs
from yahooquery import Ticker
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import datetime

# importing key lstm model, and heuristic trading model
from heuristic import *
from lstm import *

# API key for alphavantage
APIKEY = 'PUT API KEY HERE'


################
# Pseudo Testing
# includes funcs to help and carry out pseudo testing of the lstm
################


def pseudo_test(num_datapoints, plot = False):
    '''Preps some test data and then fake feeds a given amount of it to the model as though it is new data'''

    #### SETTING UP THE TICKER AND MODEL ####
    # getting the pieces in place to start
    # get tickers list
    tickers_list = read_chunks('all_stocks_list')[0]

    # load model
    model = LSTM('2821_COMPLETE')

    # prompt for ticker
    while True:
        ticker = input('\nWhich ticker would you like to choose? ')
        if ticker in tickers_list:
            print('{} selected'.format(ticker))
            break
        else:
            print('\n{} has not been trained on, select another please'.format(ticker))

    #### PRE SESH TRAIN ####
    # in order to get the model pre fired, we run a quick training session for it
    # process data
    print("\nStarting training update\n")
    X_train, X_test, y_train, y_test, raw_scale_val, batch, mimx_scaler = process_data(ticker[0], ticker, True)

    # train
    h = model.train_model(X_train, y_train, ticker, batch, test = True)
    print('\nTraining done, starting pseudo session\n')

    #### PSEUDO TIME ####
    # basically feed last num_datapoints vals of X_test to it one step at a time and see what happens
    predictions = np.empty((1,1))
    iters = 0
    for i in range(X_test.shape[0] - num_datapoints, X_test.shape[0]):

        # show where we are
        iters += 1
        if iters % 100 == 0:
            print('Step {} of test'.format(iters))

        # get the individual step data
        step_data = X_test[i].reshape(X_test[i].shape[0], 1, X_test[i].shape[1])
        
        # give it to the model and add to list of predictions
        pred = model.test_model(step_data, batch, False)
        predictions = np.append(predictions, pred, axis = 0)

        # do a little micro train to get it more accurate, NB that we do this every 5 'mins'
        if i % 5 == 0:
            true = np.array([y_test[i]])
            h = model.model.fit(step_data, true, batch_size = 1, epochs = 10, verbose = 0)

    #### EVALUATION
    
    # MSE
    mse = MeanSquaredError()
    mse.update_state(y_test[-num_datapoints], predictions[1:])
    print('\nMSE for {} is {}\n'.format(ticker, mse.result().numpy()))

    # rescale for plotting
    predictions /= raw_scale_val
    y_test /= raw_scale_val

    # if we want to plot, do that
    if plot == True:
        model.plot_results(predictions[1:], y_test[-num_datapoints:], '{} Test'.format(ticker))

    # after done, just return the predictions as a 1D list so we have the option to look at them easily
    return np.squeeze(predictions[1:])


def pseudo_test_list(num_datapoints, ticker_list):
    '''Rips a lot of pseudo tests and puts their results into a csv'''

    # load model
    model = LSTM('2821_COMPLETE')

    # loop over tickers
    for ticker in ticker_list:

        # in order to get the model pre fired, we run a quick training session for it
        # process data
        print("\nStarting training update on {}\n".format(ticker))
        X_train, X_test, y_train, y_test, raw_scale_val, batch, mimx_scaler = process_data(ticker[0], ticker, True)

        # train
        h = model.train_model(X_train, y_train, ticker, batch, test = True)
        print('\nTraining done, starting pseudo session\n')

        # predict time
        predictions = np.empty((1,1))
        iters = 0
        for i in range(X_test.shape[0] - num_datapoints, X_test.shape[0]):

            # show where we are
            iters += 1
            if iters % 100 == 0:
                print('Step {} of test'.format(iters))

            # get the individual step data
            step_data = X_test[i].reshape(X_test[i].shape[0], 1, X_test[i].shape[1])
            
            # give it to the model and add to list of predictions
            pred = model.test_model(step_data, batch, False)
            predictions = np.append(predictions, pred, axis = 0)

            # do a little micro train to get it more accurate, NB that we do this every 5 'mins'
            if i % 5 == 0:
                true = np.array([y_test[i]])
                h = model.model.fit(step_data, true, batch_size = 1, epochs = 10, verbose = 0)
        
        # MSE
        mse = MeanSquaredError()
        mse.update_state(y_test[-num_datapoints:], predictions[1:])
        print('\nMSE for {} is {}\n'.format(ticker, mse.result().numpy()))

        # setup predictions, test and rescale
        predictions = predictions[1:]
        y_test = y_test[-num_datapoints:]
        predictions /= raw_scale_val
        y_test /= raw_scale_val
        predictions = np.squeeze(predictions)

        # new df for all 3 
        df_pred = pd.DataFrame({'{} pred'.format(ticker): predictions})
        df_y = pd.DataFrame({'{} y'.format(ticker): y_test})
        df_mse = pd.DataFrame({'{} mse'.format(ticker): [mse.result().numpy()]})

        # read current csv
        existing_df = pd.read_csv('pseudo_test_list.csv')

        # append
        result_df = pd.concat([existing_df, df_pred, df_y, df_mse], axis = 1)

        # send to csv
        result_df.to_csv('pseudo_test_list.csv', index = False)


def append_to_csv(file_path, value):
    '''Appends a value to a csv, used this for live data logging'''

    try:
        # read the csv to get a df, make a new row, and then write it to the file
        df = pd.read_csv(file_path)

        new_row = pd.DataFrame([value], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(file_path, index=False)

    # error handling
    except FileNotFoundError:
        print('File {} not found.'.format(file_path))
    except Exception as e:
        print('An error occurred: {}'.format(str(e)))


################
# Live Testing
# live trade function which takes in live data and makes predictions 1 min ahead, and helper to interface with AlphaVantage
################


def get_live_data(ticker, interval = '1min'):
    '''Fetches live stock data and returns as a pd.Dataframe'''

    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&entitlement=realtime&symbol={}&interval={}&apikey={}'.format(ticker, interval, APIKEY)
    response = requests.get(url)
    data = response.json()

    if 'Time Series (1min)' in data:
        time_series = data['Time Series (1min)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
        return df
    else:
        print('No data available for {}'.format(ticker))
        return None


def live_trade():
    '''Prompts for, and then runs a session on the given stock and model'''

    #### SETTING UP THE TICKER AND MODEL ####
    # getting the pieces in place to start
    # get tickers list
    tickers_list = read_chunks('all_stocks_list')[0]
    
    # load model
    model = LSTM('2821_COMPLETE')

    # prompt for ticker
    while True:
        ticker = input('\nWhich ticker would you like to choose? ')
        if ticker in tickers_list:
            print('{} selected'.format(ticker))
            break
        else:
            print('\n{} has not been trained on, select another please'.format(ticker))

    #### PRE SESH TRAIN ####
    # in order to get the model pre fired, we run a quick training session for it
    # process data
    print("\nStarting training update\n")
    X_train, X_test, y_train, y_test, raw_scale_val, batch, mimx_scaler = process_data(ticker[0], ticker, True)

    # train
    h = model.train_model(X_train, y_train, ticker, batch, test = True)
    print('\nTraining done, starting live session\n')

    #### DOING IT LIVE ####
    # every min, 2 seconds in will grab the data and go
    print("\nTraining done, going live...\n")
    predictions = np.empty((1, 1))
    actuals = np.empty((1))
    num_predictions = 0
    pred = 0
    live_data = None
    try:
        while True:
            current_time = time.localtime()
            if current_time.tm_sec == 2:

                # grab live data
                data = get_live_data(ticker).head(1)

                # do the concurrent training, now we have the live data from the last time, and the result
                # the reason why we do it this way is because the way that the original data is set up is that the features predict the t + 1 close result, so we need the t + 1 result for the concurrent fitting
                if num_predictions != 0:
                    true = np.array([float(data['close'][-1])])
                    append1 = append_to_csv('actuals.csv', true[0])
                if (num_predictions - 1) % 5 == 0:
                    true = true * raw_scale_val
                    h = model.model.fit(live_data, true, batch_size = 1, epochs = 10, verbose = 0)

                # now concurrent train is done, we 
                # grab the numbers we want and put into a list for live data prep
                open = data['open'][-1]
                high = data['high'][-1]
                low = data['low'][-1]
                volume = data['volume'][-1]
                live_data_raw = [open, high, low, volume]

                # add the actual close to the actuals list
                if len(predictions) != 0:
                    actuals = np.append(actuals, float(data['close'][-1]))
                    
                # now push the data into a usable format for the model
                live_data = prediction_data_prep(live_data_raw, mimx_scaler, raw_scale_val)

                # now get the prediction
                pred = model.test_model(live_data, batch)
                append2 = append_to_csv('preds.csv', pred[0][0] / raw_scale_val)
                print('Prediction at {}: {}'.format(datetime.now(), pred / raw_scale_val))
                num_predictions += 1

                # add the prediction to the ongoing list
                predictions = np.append(predictions, pred, axis = 0)

                # do something with it
                #####
                # PUT FUNC HERE WHICH COULD OTHERWISE SEND THE DATA TO ANOTHER AI / HEURISTIC AGENT WHICH USES THE PREDICTION TO CARRY OUT TRADING
                #####
                
            # wait for another second to check 
            time.sleep(1)

    # if we stop the sesh, handle
    except KeyboardInterrupt:
        print('\nLive session complete')

    return predictions, actuals
