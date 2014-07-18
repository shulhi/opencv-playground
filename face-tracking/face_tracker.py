import cv2
import cv2.cv as cv

def detect(img):
    cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    # some other examples
    # cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml")
    # cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_upperbody.xml")
    # cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml")
    rects = cascade.detectMultiScale(img, 1.05, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (100,100))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def main():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    # small resolution, so the frame rate is higher
    vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        # detection
        rects, img = detect(frame)
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)

        cv2.imshow("preview", img)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")

if __name__ == '__main__':
    main()
