import os
import shutil


# deletes all .png or .jpg files in given directory
def del_all_files(dirpath: str):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                os.remove(filepath)
        except OSError:
            os.remove(filepath)


# deletes every file and directory in given directory, INCLUDING THE DIRECTORY
def del_everything(dirpath: str):
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    print(dirpath, '- directory cleaned.')


if __name__ == '__main__':
    pass
    # del_everything(r'C:\Users\jpawl\Desktop\spacesniffer_1_3_0_2')
