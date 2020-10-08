import os
import shutil

#deletes all .png files in given directory
def del_all_files(dirpath: str):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            if filename.endswith(".png"):
                os.remove(filepath)
        except OSError:
            os.remove(filepath)