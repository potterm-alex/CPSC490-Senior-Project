# data handling / scaling
import pandas as pd


################
# Stock List Prep
# prepped my stock list data of stocks with market cap >$2B into something I could use for training
# these made 'chunk' lists which I could then use to rip training iterations, very useful
################


def raw_data_prep(filename):
        '''Takes the file we want to prep for stock list'''

        # read in data
        stocks = pd.read_csv('Ticker_lists/{}.csv'.format(filename), na_values = None)
        
        # drop unneccessary cols
        stocks = stocks.drop(['Name', 
                              'Sector', 
                              'Industry', 
                              'Last Sale',
                              'Net Change',
                              '% Change',
                              'Market Cap',
                              'Country',
                              'IPO Year',
                              'Volume'], 
                              axis=1)
        
        # return
        return stocks


def split_to_chunks(dataframe, chunk_size = 100):
        '''Splits a df down to the chunks we want'''
    
        # convert df col to a list
        data = dataframe.iloc[:, 0].tolist()

        # calc num chunks
        num_chunks = len(data) // chunk_size

        chunks = []

        # split data into chunks
        for i in range(num_chunks):
                start_index = i * chunk_size
                end_index = (i + 1) * chunk_size
                chunk = data[start_index:end_index]
                chunks.append(chunk)

        # add remaining elements to last chunk
        if len(data) % chunk_size != 0:
                last_chunk = data[num_chunks * chunk_size:]
                chunks.append(last_chunk)

        # return
        return chunks


def write_chunks(chunks, filename):
        '''Writes the chunks of the df into a new csv'''

        # make a df of chunks
        df_chunks = pd.DataFrame({'Chunk': chunks})
    
        # write this to a csv
        df_chunks.to_csv('Ticker_lists/{}.csv'.format(filename), index = False)


def read_chunks(filename):
        '''Read the chunks from the csv'''

        import ast
    
        # read the chunks from the csv
        chunks = pd.read_csv('Ticker_lists/{}.csv'.format(filename))
    
        # convert these to lists so we can use them
        chunks['Chunk'] = chunks['Chunk'].apply(ast.literal_eval)

        # empty list
        chunks_list = []

        # add these to the list
        for list in chunks['Chunk']:
                chunks_list.append(list)
    
        # return
        return chunks_list
