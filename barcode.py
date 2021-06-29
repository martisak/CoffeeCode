# -*- coding: utf-8 -*-

import numpy as np
import argparse
import cv2
from pyzbar.pyzbar import decode
from glob import glob
import os
import imutils

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = ()
radPt = ()


def click(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, rad

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (x, y)
        cv2.rectangle(crop_output,
                      (refPt[0] - 20, refPt[1] - 20),
                      (refPt[0] + 20, refPt[1] + 20), (0, 255, 0), 2)

    if event == cv2.EVENT_RBUTTONDOWN:
        radPt = (x, y)

        if refPt:
            cv2.line(crop_output, refPt, radPt, (0, 255, 0), 2)

            cv2.rectangle(crop_output,
                          (radPt[0] - 20, radPt[1] - 20),
                          (radPt[0] + 20, radPt[1] + 20), (255, 0, 0), 2)


def detect_circle(im):

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    #blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    #thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)[1]
    circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1.2, 100)

    return circles


def draw_circle(im, center, radius):
    cv2.circle(im, center, radius, (0, 255, 0), 4)
    cv2.rectangle(im, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return im

def unwrap(im):

    height, width = im.shape[0:2]
    r = height // 2
    dst = cv2.warpPolar(im, (width, height),
                        (int(r), int(r)), int(r), 0)
    dst = dst[:, width - 80:width]
    dst = cv2.rotate(dst, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return dst


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--image_path", required=True,
                help="Path to the image of the pod")
args = vars(ap.parse_args())

outputs = []
names = []
images = glob(os.path.join(args["image_path"], "*"))
print(images)

for image_filename in images:

    image = cv2.imread(image_filename)
    output = image.copy()

    circles = detect_circle(image)
    # ensure at least some circles were found

    (h, w, c) = output.shape
    cv2.line(output, (0, h // 2), (w, h // 2), (255, 0, 0), 2)
    theta = 0

    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")
        (x, y, r) = circles[0]

        draw_circle(output, (x, y), r)

        crop_img = image[y - r:y + r, x - r:x + r]
        crop_output = output[y - r:y + r, x - r:x + r]

        #(h, w) = crop_output.shape[:2]
        #(cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        #M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)

        #rotated = imutils.rotate_bound(crop_output, theta)
        #crop_output = rotated.copy()
        dst = unwrap(crop_output)

        cv2.namedWindow("image")

        while True:

            cv2.imshow("image", np.vstack([crop_output, dst]))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                outputs.append(dst)
                names.append(image_filename)
                break

            if key == ord("d"):
                x += 2
            elif key == ord("a"):
                x -= 2
            elif key == ord("w"):
                y -= 2
            elif key == ord("s"):
                y += 2
            elif key == ord("p"):
                r += 2
            elif key == ord("l"):
                r -= 2
            elif key == ord("i"):
                theta += 2
            elif key == ord("j"):
                theta -= 2

            output = image.copy()

            #rotated = imutils.rotate_bound(output, theta)
            #output = rotated

            (h, w, c) = output.shape
            cv2.line(output, (0, y), (w, y), (255, 0, 0), 1)
            cv2.line(output, (x, 0), (x, h), (255, 0, 0), 1)
            cv2.line(output, (x - 100, y - 100),
                     (x + 100, y + 100), (255, 0, 0), 1)
            cv2.line(output, (x - 100, y + 100),
                     (x + 100, y - 100), (255, 0, 0), 1)

            draw_circle(output, (x, y), r)

            crop_img = image[y - r:y + r, x - r:x + r]
            crop_output = output[y - r:y + r, x - r:x + r]

            dst = unwrap(crop_img)

max_w = 0

for o in outputs:
    w = o.shape[1]
    if w > max_w:
        max_w = w

print(max_w)

new_outputs = []

for i, (o, image_filename) in enumerate(zip(outputs, names)):

    (h, w, c) = o.shape

    dst2 = o.copy()

    for x in range(w):
        dst2[:, x, :] = np.mean(dst2[:, x, :])

    dst_g = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
    thresh1, dst2 = cv2.threshold(dst_g, 127, 255, cv2.THRESH_BINARY)
    dst2 = cv2.cvtColor(dst2, cv2.COLOR_GRAY2RGB)

    bordersize = 2
    border = cv2.copyMakeBorder(
        dst2,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 0]
    )

    print(decode(border))

    blurred = cv2.bilateralFilter(border, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(dst2, [screenCnt], -1, (0, 255, 0), 3)

    tmp = cv2.resize(border, (max_w, 100))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tmp, image_filename, (10, 40), font,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    new_outputs.append(tmp)

cv2.imshow("output", np.vstack(new_outputs))
cv2.waitKey(0)
