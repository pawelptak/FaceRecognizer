import cv2


# crops image with given dimensions
def crop_image(src, x, y, w, h):
    img = cv2.imread(src)
    crop_img = img[y:y + h, x:x + w]
    return crop_img
