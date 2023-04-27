import os

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
os.makedirs(os.path.join('data','summaries'), exist_ok=True)
for base_folder in os.listdir('data'):
    cur_path=os.path.join('data',base_folder)
    data=None
    for i in os.listdir(cur_path):
        with open(os.path.join(cur_path, i), 'rb') as fp:
            cur_data = pickle.load(fp)
        if data is None:
            data = cur_data
        else:
            for k,v in cur_data.items():
                data[k].append(v)