import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

def read_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: filepath for raw messages dataset
        categories_filepath: filepath for raw categories dataset
    ouput:
        merge_df: the two datasets joined
    '''
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    merge_df = messages.merge(categories,how='left',on='id')
    return merge_df

def clean_data(messy_data):
    '''
    input: 
        messy_data: merged dataset without any cleaning
    output:
        messy_data: the dataset after cleaning
    '''
    categories = messy_data.categories.str.split(';', expand = True)
    row = categories.loc[0]
    category_names = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_names
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    messy_data.drop('categories', axis = 1, inplace = True)
    messy_data = pd.concat([messy_data, categories], axis = 1)
    messy_data.drop_duplicates(subset = 'id', inplace = True)
    return messy_data

def store_data(df, filepath):
    '''
    save the dataset to a database
    '''
    engine=create_engine('sqlite:///' + filepath)
    df.to_sql('df', engine, index=False)
    print('store data')


def main():
    if len(sys.argv)==4:
        messages, categories, database = sys.argv[1:]

        print(f'reading and combining the datasets {messages} and {categories}')
        df = read_data(messages,categories)
        print('cleaning data')
        df = clean_data(df)
        print(f'saving data to {database}')
        store_data(df, database)
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
    

if __name__=='__main__':
    main()