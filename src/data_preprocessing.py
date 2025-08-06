import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df
