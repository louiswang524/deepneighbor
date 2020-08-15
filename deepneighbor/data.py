import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("datasets/data.csv",nrows=10000)
    data = data[['item_id','user_id']]
    data.columns = ['item','user']
    data['item'] = data['item'].astype(str)
    data['user'] = data['user'].astype(str)
    data.to_csv("datasets/processed_data.csv",index=None)
    #print(data.head())
    #print("read 100 rows!")
