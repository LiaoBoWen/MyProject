import pickle

with open('temp_data.pkl','rb') as f:
    all = pickle.load(f)
    print(all)
    print(len(all))