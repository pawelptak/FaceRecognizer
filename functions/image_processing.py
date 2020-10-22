import os
import cv2

def normalize_images(dir_path): #normalizes all images in given directory
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith('jpg') or file_name.endswith('png'):
                print(file_name)
                img = cv2.imread(file_path)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite(file_path, img)
    print('Images normalized')


def resize_images(dir_path, x, y): #resizes all images in given directory to given dimensions
    size = (x,y)
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_name.endswith('jpg') or file_name.endswith('png'):
                print(file_name)
                img = cv2.imread(file_path)
                img = cv2.resize(img, size)
                cv2.imwrite(file_path, img)
    print('Images resized')

if __name__ == '__main__':
    resize_images('../validation_set/pawel', 200, 200)
    resize_images('../validation_set/dorota', 200, 200)
    resize_images('../validation_set/justyna', 200, 200)
