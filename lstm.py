# data handling useful stuff
import pandas as pd
import numpy as np

# plotting
import matplotlib. pyplot as plotting

# some preprocessing stuff
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# the important keras bits
from keras.models import Sequential, save_model, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import History
from keras.metrics import MeanSquaredError

# misc.
import time
import sys
from datetime import datetime

# from other files
from chunks import *


################
# Manual Operation
# either to do training on everything, or to do manual work and testing
################


def manual_session(tickers):
        '''Runs a manual train / testing session on a given list of tickers'''

        # prompt and load model
        while True:
                try:
                        model_input = input('\nWhich model (if any) would you like to load? ')
                        if model_input == 'N':
                                model = LSTM()
                                print('\nNew model loaded')
                                break
                        else:
                                model = LSTM(model_input)
                                print("\n{} loaded".format(model_input))
                                break
                except OSError:
                        print('\nThat model does not exist, try again')
        
        # get tickers
        tickers = tickers

        # prompt test or train
        while True:
                session = input('\nTest or Train? ')
                if session == 'Test':
                        run_testing_iter(model, tickers)
                        break
                elif session == 'Train':
                        run_training_iter(model, tickers)
                        break
                else:
                        print('\nMust give either Train or Test')

        # done!
        print('\nSession done!')


def chunk_training_session():
        '''Runs a training iteration on a given data chunk'''
        
        # prompt and load model
        while True:
                try:
                        model_input = input('\nWhich model (if any) would you like to load? ')
                        if model_input == 'N':
                                model = LSTM()
                                print('\nNew model loaded')
                                break
                        else:
                                model = LSTM(model_input)
                                print("\n{} loaded".format(model_input))
                                break
                except OSError:
                        print('\nThat model does not exist, try again')

        # prompt for and get stock chunks
        while True:
                try:
                        file_input = input('\nWhich stock list would you like to load? ')
                        stocks = read_chunks(file_input)
                        break
                except FileNotFoundError:
                        print('\nThat file does not exist, try again')

        # prompt for and get how many chunks you'd like to run
        while True:
                try:
                        num_chunks = int(input('\nHow many chunks would you like to do? '))
                        if num_chunks < 1 or num_chunks > 3:
                                print('\nNot allowed to do {} chunks at a time, pick 1/2/3'.format(num_chunks))
                        else:
                                print('\nWill do {} chunks'.format(num_chunks))
                                break
                except TypeError:
                        print('Must be int between 1 and 3')
                except ValueError:
                        print('Must be int between 1 and 3')

        # print chunk options:
        print('\nWhich chunk would you like to train?\nNB if more than 1 chunk, 1st chunk will be one chosen, and so on')
        for i in range(len(stocks)):
                print('{}: {} to {}'.format(i + 1, stocks[i][0], stocks[i][-1]))

        # get chunk input
        while True:
                try:
                        chunk_input = int(input('\nChunk: '))
                        if chunk_input < 1 or chunk_input > len(stocks):
                                print('Choose chunk shown')
                        else:
                                stocks_list = []
                                for i in range(num_chunks):
                                        stocks_list += stocks[chunk_input + i - 1]
                                        print('{} selected'.format(chunk_input + i))
                                print('Will train {} through {}'.format(stocks_list[0], stocks_list[-1]))
                                break
                except TypeError:
                        print('Must be int between 1 and {}'.format(len(stocks)))
                except ValueError:
                        print('Must be int between 1 and {}'.format(len(stocks)))

        # run the training
        res = run_training_iter(model, stocks_list)

        # done!
        print('\nSession done!')
        c = []
        for i in range(num_chunks):
                c.append(i + chunk_input)
        return c, file_input


def run_training_iter(model, tickers):
        '''Takes in a model, and a list of tickers, and runs a training loop on all of them'''

        # loop over the tickers
        for i in range(len(tickers)):

                # find the folder and filename of that ticker
                ticker = tickers[i]
                folder = ticker[0]
                filename = ticker
                print("\nTraining on {}, no. {} of chunk\n".format(filename, i + 1))
                
                # try to process the data of that ticker
                try:
                        X_train, X_test, y_train, y_test, raw_scale_val, batch, mimx = process_data(folder, filename)
                except FileNotFoundError:
                        print("{} does not exist in {}".format(filename, folder))
                        continue

                # train
                h = model.train_model(X_train, y_train, filename, batch)

                # test
                y = model.test_model(X_test, batch, False)

                # scaling data back down for the eval
                y_test = y_test / raw_scale_val
                y = y / raw_scale_val

                # run some mse analysis
                mse = MeanSquaredError()
                mse.update_state(y_test, y)
                print('MSE for {} is {}'.format(filename, mse.result().numpy()))


def run_testing_iter(model, tickers):
        '''Takes in a model, and a list of tickers, and runs a training loop on all of them'''

        # loop over the tickers
        for i in range(len(tickers)):

                # find the folder and filename of that ticker
                ticker = tickers[i]
                folder = ticker[0]
                filename = ticker
                print("\nTesting on {}\n".format(filename))
                print("First, quick retrain:")
                
                # try to process the data of that ticker
                try:
                        X_train, X_test, y_train, y_test, raw_scale_val, batch, mimx = process_data(folder, filename, True)
                except FileNotFoundError:
                        print("{} does not exist in {}".format(filename, folder))
                        continue

                # train
                h = model.train_model(X_train, y_train, filename, batch, test = True)

                # test
                y = model.test_model(X_test, batch, True, y_test, "{} Test".format(filename))

                # scaling data back down for the eval
                y_test = y_test / raw_scale_val
                y = y / raw_scale_val

                # print out last predicted values
                print((y[-1])[0], y_test[-1])

                # run some mse analysis
                mse = MeanSquaredError()
                mse.update_state(y_test, y)
                print('MSE for {} is {}'.format(filename, mse.result().numpy()))
       

################
# Manual Data
# does the handling and prep work for the data
################


def get_raw_data(folder, file):
        '''Takes the folder and file, and gets it into a pd.Dataframe'''

        # if the file is clearly just a ticker, do this
        if len(file) < 8:
                raw_data = pd.read_csv('Data/{}/{}_full_1min_adjsplitdiv.txt'.format(folder, file), na_values = 'null')
                raw_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                raw_data.set_index('date', inplace = True)
        else:
                raw_data = pd.read_csv('Data/{}/{}'.format(folder, file), na_values = 'null')
                raw_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                raw_data.set_index('date', inplace = True)

        # return 
        return raw_data
        

def process_data(folder, file, test = False):
        '''Takes raw data and processes it into x train/test and y train/test sets ready to be used in the model which are returned to caller, does most of the heavy lifting'''
        
        #### DATA READING ####
        # grab the dataset and read it into a dataframe with pandas
        raw_data = get_raw_data(folder, file)

        #### DATA SIZING AND INITIAL SCALING ####
        # if we are testing, use the most recent quarter of the data just so more recent trends are overemphasized
        if test == True:
                raw_data = raw_data[-(raw_data.shape[0] // 4):]

        # determining the batch size we will use for the training based on data size, this affects steps per epoch etc so want to keep in the range of ~8000, which seems to be optimal for the model
        # we achieve this via taking the raw data size, taking the training split, then dividing by 8000 and bringing to closest 10
        inter = raw_data.shape[0] * 0.9 / 8000
        if inter < 1:
                batch = 1
        elif inter < 10:
                batch = int(inter)
        else:
                batch = int(inter // 10 * 10)
        
        # some raw data scale up / downs here, these are necessary as at lower values, it gives the eventual minimax scaling less ability to be granular, and for higher values, seems to keep the data in a bound where the model works well
        raw_scale_val = 1000.0 / raw_data['close'][-1]
        for col in ['open', 'high', 'low', 'volume', 'close']:
                raw_data[col] = raw_data[col] * raw_scale_val

        #### DATA SETUP ####
        # data is given with the features and their corresponding current adj price that they result in, I want to predict the next time period's price, so we adjust the output var up one column so the features are with what I want it to correlate to
        raw_data['close+1'] = raw_data['close'].shift(-1)
        raw_data = raw_data.dropna()

        #### FEATURES SETUP / SCALING ####
        # we will now set up the features and have them ready to be used for training and testing
        features = ['open', 'high', 'low', 'volume']

        # we now scale the values down so it is quicker to process using the minimaxscaler from sklearn's preprocessing suite
        # Initialize MinMaxScaler
        mimx_scaler = MinMaxScaler()

        # Fit and transform the scaler to the raw data
        scaled_data = mimx_scaler.fit_transform(raw_data[features])

        # then we change these back into a dataframe
        scaled_data = pd.DataFrame(columns = features, data = scaled_data, index = raw_data.index)
        scaled_data.head()

        #### OUTPUT VAR SETUP ####
        # create a simple dataframe that contains the adjusted close, which we will use as our output variable
        output = pd.DataFrame(raw_data['close+1'])

        #### TRAINING AND TESTING SPLIT SETUP ####
        # using the TimeSeriesSplit class from sklearn, we create the train and testing splits for the data, then split the data into the training and testing splits
        timesplit= TimeSeriesSplit(n_splits = 10)
        for train_index, test_index in timesplit.split(scaled_data):
                X_train = scaled_data[:len(train_index)]
                X_test = scaled_data[len(train_index):(len(train_index) + len(test_index))]
                y_train = output[:len(train_index)].values.ravel()
                y_test = output[len(train_index):(len(train_index) + len(test_index))].values.ravel()

        #### FINAL RESHAPING FOR LSTM ####
        # we have to do some reshaping for the x train/test arrays, so first we set up into numpy arrays for the training and test sets, then reshape them into the dimensions of the dataframe we created above
        trainX = np.array(X_train)
        testX = np.array(X_test)
        X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

        #### DONE ####
        # we return the x/y train/test arrays for the model as well as the raw scale value and the batch size
        return X_train, X_test, y_train, y_test, raw_scale_val, batch, mimx_scaler


def add_data(original_data, new_data, mimx_scaler, raw_scale_val):
        '''Adds some new data, given as an np.array to the np.array of original data'''

        # reminder of the features, we need this for the rescaling
        features = ['open', 'high', 'low', 'close', 'volume']

        # set up as df of the new data given
        new_df = pd.DataFrame(
                {
                        'open': [new_data[0]],
                        'high': [new_data[1]],
                        'low': [new_data[2]],
                        'close': [new_data[3]],
                        'volume': [new_data[4]]
                }
        )

        # much as we did before, scale the data by the raw scale value to pump it up and give the minmixscaler some more room
        for col in ['open', 'high', 'low', 'volume', 'close']:
                new_df[col] = new_df[col] * raw_scale_val

        # scale the new df around the features
        new_scaled = mimx_scaler.transform(new_df[features])

        # reshape the new scaled data so it can be added to the original
        new_scaled = new_scaled.reshape(new_scaled.shape[0], 1, new_scaled.shape[1])

        # return the concatenated data
        return np.concatenate((original_data, new_scaled), axis=0)


def prediction_data_prep(new_data, mimx_scaler, raw_scale_val):
        '''Takes some new data for a given time, reshapes and scales it so it can be used for prediction'''

        # reminder of the features, we need this for the rescaling
        features = ['open', 'high', 'low', 'volume']

        # set up as df of the new data given
        new_df = pd.DataFrame(
                {
                        'open': [new_data[0]],
                        'high': [new_data[1]],
                        'low': [new_data[2]],
                        'volume': [new_data[3]]
                }
        )

        # much as we did before, scale the data by the raw scale value to pump it up and give the minmixscaler some more room
        for col in ['open', 'high', 'low', 'volume']:
                new_df[col] = new_df[col].astype(float) * raw_scale_val

        # scale the new df around the features
        new_scaled = mimx_scaler.transform(new_df[features])

        # reshape the new scaled data so it can be added to the original
        new_scaled = new_scaled.reshape(new_scaled.shape[0], 1, new_scaled.shape[1])

        # return the newly prepped data for be used for a prediction
        return new_scaled


################
# Manual Models
# functions to manually create, train, test and plot models
################


def build_model(feature_length) -> Sequential:
        '''Builds an LSTM model using keras' implementation, and returns it to caller'''

        # begin by making the model, here we are using a sequential model as the base
        model = Sequential()

        # then we add layers
        # Layer 1: LSTM with 48 units, and relu activation
        model.add(LSTM(48, input_shape = (1, feature_length), activation = 'relu', return_sequences = False))
        # Layer 2: Dense
        model.add(Dense(1))

        # we now compile the model, currently using MSE for loss and adam optimizer for the optmization
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')

        # return the compiled model
        return model


def train_model(model, X_train, y_train, batch_size, epochs = 10) -> History:
        '''Takes a training set, and fits it to the model using a given number of epochs and batch size, returns the History keras class which is record of training loss values and metrics values at successive epochs'''

        # first reset states
        model.reset_states()
        # rather simple:
        return model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1, shuffle = False, use_multiprocessing = True)


def test_model(model, X_test, batch):
        '''Tests the model with a given testing set'''

        # simple
        return model.predict(X_test, batch)


def plot_results(y_pred, y_test, title):
        '''Using matplotlib, plots the given data'''

        # plot all the data given
        plotting.plot(y_test, label='True Value')
        plotting.plot(y_pred, label='Model Value')

        # some other setup
        plotting.title(title)
        plotting.xlabel('Time Scale')
        plotting.ylabel('Price (USD)')
        plotting.legend()

        # show 
        plotting.show()
 

################
# LSTM Model Class
# handles all of my interfacing and work with the LSTM prediction model
################


class LSTM():
        '''Handles all of my interfacing and work with the LSTM model'''

        def __init__(self, model_file = None):
                '''Initialization of all important parts of the model for its function'''
                
                # set up model here
                self.model = None

                # training iterations will be useful for designation of saving the model after each training is done, this is purely for my records
                self.training_iterations = 0

                # on init, if no model file argument given, will assume that we are building a new model, so builds one, but if model file is given it loads the model and also updates the training iterations int
                if model_file:
                        self.model = load_model('LSTM_Models/{}.keras'.format(model_file), compile = True)
                        loaded_iters = ''
                        i = 0
                        while True:
                                if model_file[i] != '_':
                                        loaded_iters += model_file[i]
                                        i += 1
                                else:
                                        break
                        self.training_iterations = int(loaded_iters)
                else:
                        self.build_model()

        def build_model(self):
                '''Builds an LSTM model using keras' implementation, with dimensions to fit the training data, and returns it to caller'''

                # begin by making the model, here we are using a sequential model as the base
                model = Sequential()

                # then we add layers
                # Layer 1: LSTM with 48 units, and relu activation
                model.add(LSTM(48, input_shape = (1, 4), activation = 'relu', return_sequences = False))
                # Layer 2: Dense
                model.add(Dense(1))

                # we now compile the model, currently using MSE for loss and Adam optimizer for the optmization
                model.compile(loss = 'mean_squared_error', optimizer = 'adam')

                # set the model as this now
                self.model = model

        def train_model(self, X_train, y_train, filename, batch_size, test = False, epochs = 10) -> History:
                '''Takes a training set, and fits it to the model using a given number of epochs and batch size, returns the History keras class which is record of training loss values and metrics values at successive epochs'''

                # rather simple, if model is built/loaded, we fit the data, return the history for any analysis
                # note that if we are doing a quick re train for testing, we don't do a save, if training we do
                if self.model:
                        if test == True:
                                self.model.reset_states()
                                history = self.model.fit(X_train, y_train, epochs = epochs * 2, batch_size = batch_size, verbose = 1, shuffle = False, use_multiprocessing = True)
                                return history
                        else:
                                self.model.reset_states()
                                history = self.model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1, shuffle = False, use_multiprocessing = True)
                                self.training_iterations += 1
                                name = ""
                                for letter in filename:
                                        if letter == '_':
                                                break
                                        else:
                                                name += letter
                                save_model(self.model, 'LSTM_Models/{}_{}.keras'.format(self.training_iterations, name))
                                return history
                else:
                        print('No model loaded or built, do so please') 

        def test_model(self, X_test, batch, plot = False, y_test = None, title = None):
                '''Tests the model with a given testing set'''

                # simple, if model is built/loaded, we test and then plot the results for fun if we ask for it
                if self.model:
                        y_pred = self.model.predict(X_test, batch, verbose = 0)
                        if plot == True:
                                self.plot_results(y_pred, y_test, title)
                        return y_pred
                else:
                        print('No model loaded or built, do so please') 

        def plot_results(self, y_pred, y_test, title):
                '''Using matplotlib, plots the given data'''

                # plot all the data given
                plotting.plot(y_test, label='True Value')
                plotting.plot(y_pred, label='Model Value')

                # some other setup
                plotting.title(title)
                plotting.xlabel('Time Scale')
                plotting.ylabel('Price (USD)')
                plotting.legend()

                # show 
                plotting.show()


if __name__ == "__main__":

        # shit to do chunk training and log it
        start = time.time()
        chunk, file = chunk_training_session()
        end = time.time()

        original = sys.stdout
        with open("training_log.txt", 'a') as log:
                sys.stdout = log
        
                print("Training chunk {} on {} stocks\n".format(chunk, file))
                print("Start time is {}".format(datetime.fromtimestamp(start)))
                print("Finish time is {}".format(datetime.fromtimestamp(end)))
                print("Took {} mins, {} hours".format(float(end - start) / 60, float(end - start) / 60 / 60))
                print("\n")

        sys.stdout = original

        print('\n######\nDONE\n######\n')
        