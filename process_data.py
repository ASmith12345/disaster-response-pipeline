import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def read_data(messages_filepath, categories_filepath):
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    merge_df = messages.merge(categories,how='left',on='id')
    return merge_df

def clean_data(messy_data):
    categories = messy_data['categories'].str.split(';', expand=True)

    category_names = categories.iloc[0].str[:-2]
    categories.columns = category_names

    for col in categories:
        categories[col]=categories[col].str[-1]

    messy_data.drop('categories', axis=1, inplace=True)
    clean_data = pd.concat([messy_data,categories], axis=1)
    clean_data.drop_duplicates(subset='id', inplace=True)
    clean_data.drop(['child_alone'],axis=1)
    clean_data['related'] = np.where(clean_data['related']==2, 0, clean_data['related'])
    return clean_data

def store_data(df, filepath):
    engine=create_engine('sqlite:///' + filepath)
    df.to_sql('df', engine, index=False)
    print('store data')


def main():

    # replace with sys stuff, this needs to be run in command line :o
    messages='./data/disaster_messages.csv'
    categories='./data/disaster_categories.csv'
    database='./data/disaster_clean_data.db'
    # keep me
    print(f'reading and combining the datasets {messages} and {categories}')
    merged_df = read_data(messages,categories)
    print('cleaning data')
    clean_df = clean_data(merged_df)
    print(f'saving data to {database}')
    store_data(clean_df, database)
    

if __name__=='__main__':
    main()