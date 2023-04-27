
import subprocess
import re
import os
import time
def get_works_on_cluster(match_filter:str):
    result = subprocess.run(['squeue', '--me', '-o', '"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = str(result)
    result = result.split('\n')
    for i, s in enumerate(result):
        result[i] = re.split('[\s]+', s)
    index = result[0].index("NAME")
    print(f"{match_filter}")
    m = re.compile(f"{match_filter}")
    filterd_names = []
    for i, arr in enumerate(result):
        if i == 0 or len(arr) < index + 1:
            continue
        if m.match(arr[index]):
            print(arr[index])
            filterd_names.append(arr[index])
    return filterd_names

def get_exceptions(last_n_hours):
    m = re.compile('Traceback[\s\S]*')
    b_path='cluster_logs'
    if not os.path.exists(b_path):
        b_path = os.path.join('..',b_path)
    data_list = os.listdir(b_path)
    data_list_new=[]
    for i in data_list:
        if not i.endswith('.log'):
            continue
        file_path = os.path.join(b_path,i)
        t = os.path.getmtime(file_path)

        if (time.time()-t)/3600<=last_n_hours:
            data_list_new.append((i,t))
    sorted(data_list_new,key= lambda x:-x[1])
    out_str = ''
    for i,t in data_list_new:
        print(i,t)
        with open(os.path.join(b_path,i),'r') as f:
            cur_str = f.readlines()
        cur_str = '\n'.join(cur_str)
        traceback_arr = m.findall(cur_str)
        if len(traceback_arr)>0:
            print(type(i))
            out_str+='\n*************************************************************************************\n'+i+ time.ctime(t)+'\n*************************************************************************************\n'
            out_str+='\n'.join(traceback_arr)
    with open('cluster_report.log','w') as f:
        f.writelines(out_str)
    print(out_str)
