import os
import shutil

#deletes all files in given directory
def del_all_files(dirpath: str):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)