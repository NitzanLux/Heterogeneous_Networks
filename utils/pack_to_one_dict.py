import os

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
# os.makedirs(os.path.join('data','summaries'), exist_ok=True)
summaries_flag = True
for base_folder in os.listdir('data') + ['summaries']:
    if base_folder == 'summaries' and summaries_flag:
        summaries_flag=False
        continue
    cur_path = os.path.join('data', base_folder)
    data = None
    for i in os.listdir(cur_path):
        if i == 'summaries':
            continue
        with open(os.path.join(cur_path, i), 'rb') as fp:
            cur_data = pickle.load(fp)
        if data is None:
            data = cur_data
        else:
            length = None
            for k in cur_data.keys():
                if k == 'condition': continue
                if length is None:
                    length = len(cur_data[k])
                if length == 0:
                    break
                if length != len(cur_data[k]):
                    print("\n".join(["%s - %d" % (i, len(cur_data[k])) for i in cur_data.keys()]))
                    break
            else:
                for k, v in cur_data.items():
                    data[k].extend(v)

    dest_path = os.path.join('data', 'summaries')
    os.makedirs(dest_path, exist_ok=True)
    with open(os.path.join(dest_path, f'{base_folder}.p'), 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
