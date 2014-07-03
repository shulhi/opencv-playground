"""
Load image and crop faces from image to use for SVM face classification
"""

import argparse
import cv2
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
import numpy as np

def detect(img):
    cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    rects, img = detect(image)

    cropped = []

    for idx, (x1, y1, x2, y2) in enumerate(rects):
        crop_img = image[y1:y1 + (y2 - y1), x1:x1 + (x2 - x1)]
        crop_img = cv2.resize(crop_img, (100,100), interpolation = cv2.INTER_AREA)
        cv2.imshow("image" + str(idx), crop_img)
        new_img = crop_img.reshape(crop_img.shape[0] * crop_img.shape[1], 3)
        cropped.append(new_img.flatten())

    # reduce feature size
    cropped_pca = []
    pca = RandomizedPCA(n_components=100)
    cropped_pca = pca.fit_transform(cropped)

    # training
    clf   = SVC(probability=True)
    train = cropped_pca[:7]
    test  = cropped_pca[7:13]
    # clf.fit([[0,0],[1,1]], [1, 2])
    clf.fit(train, [1,2,2,1,2,1,1])

    for item in test:
        print clf.predict_proba(item)
        print clf.predict(item)

    cv2.waitKey(0)