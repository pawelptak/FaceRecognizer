import cv2


# crops image with given dimensions
def crop_image(src, x, y, w, h):
    img = cv2.imread(src)
    crop_img = img[y:y + h, x:x + w]
    #cv2.imshow("cropped", crop_img)
    #cv2.waitKey(0)
    return crop_img
