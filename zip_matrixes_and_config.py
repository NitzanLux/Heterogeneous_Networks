import os
import zipfile
import re
import argparse

def zip_matching_folders(data_folder, r_str, files_to_zip):
    for root, dirs, _ in os.walk(data_folder):
        for dir in dirs:
            if re.match(r_str, dir):

                sub_folder = os.path.join(root, str(dir))
                zip_path = f"{sub_folder}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for _, sub_dirs, _ in os.walk(sub_folder):
                        for sub_dir in sub_dirs:
                                sub_sub_folder = os.path.join(sub_folder, sub_dir)
                                for _, _, files in os.walk(sub_sub_folder):
                                    for file in files:
                                        if file in files_to_zip:
                                            abs_path = os.path.join(sub_sub_folder, file)
                                            zipf.write(abs_path, os.path.relpath(abs_path, sub_sub_folder))

parser = argparse.ArgumentParser(description='Process some strings.')
parser.add_argument('r_str', type=str, help='The regular expression to match folder names')

args = parser.parse_args()

data_folder = "data"
files_to_zip = ["config_dict.pickle", "performance_mat.p"]

zip_matching_folders(data_folder, args.r_str, files_to_zip)