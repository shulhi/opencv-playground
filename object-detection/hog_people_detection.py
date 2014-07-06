import argparse
import cv2
import cv2.cv as cv

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

    return img

def detect(im):
    hog = cv2.HOGDescriptor()
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
    hog.setSVMDetector(detector)
    found, w = hog.detectMultiScale(im, winStride=(8,8), padding=(32,32), scale=1.05)

    return found

def main():
    """
    Somehow this is running extremely slow
    """
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    # args = vars(ap.parse_args())

    # im = cv2.imread(args['image'])

    cv2.namedWindow("HOG")
    vc = cv2.VideoCapture(0)

    # downsize the resolution
    vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("HOG", frame)
        rval, frame = vc.read()

        # do stuff here
        found = detect(frame)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)

        img = draw_detections(frame, found_filtered, 3)

        cv2.imshow("HOG", img)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("HOG")


if __name__ == '__main__':
    main()