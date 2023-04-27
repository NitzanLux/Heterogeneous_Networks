import time
import os
import logging
import argparse
import json
from utils.slurm_job import *
import subprocess
import re
from utils.cluster_aid_methods import get_works_on_cluster
parser = argparse.ArgumentParser(description='json file...')

parser.add_argument('-re',dest="job_name_format", type=str,
                    help='configurations json file of paths')

args = parser.parse_args()
deleted_names = get_works_on_cluster(f"{args.job_name_format}")
delete_all = input('delete all those files? (y/n)?')
# delete_all = subprocess.run(["read answer"], stdout=subprocess.PIPE)
if delete_all=='y':
    for i in deleted_names:
        command = ['scancel',f'--jobname={i}']
        # print(command)
        subprocess.run(command, stdout=subprocess.PIPE)
