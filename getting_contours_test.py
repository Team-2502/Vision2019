import cv2
import numpy as np
import os
import sys

AUTOGEN_IMAGE_FOLDER = "./autogen_images"


def get_edges(angle):
    string_to_format = os.path.join(AUTOGEN_IMAGE_FOLDER, "2019_vision_angle_{0:0.2f}.png")

    # Read the image using CV / Convert to greyscale
    image = cv2.imread(string_to_format.format(angle))  # Image based on angle
    # image = cv2.imread("2019_vision_sample_noisy.png") # Noise image for testing

    bitmask = cv2.inRange(image, (30, 30, 30), (255, 255, 255))

    # Remove surrounding noise via opening
    # bitmask = cv2.morphologyEx(bitmask, cv2.MORPH_OPEN, np.ones((5, 5)))

    # Dilate to clean up the white blocks
    # bitmask = cv2.dilate(bitmask, np.ones((6, 6)))

    # Convert to a bitmask

    # bitmask = cv2.dilate(bitmask, np.ones((6, 6)), iterations=2)  # We dilate this twice for some reason...

    # Get messy contours
    contours, _ = cv2.findContours(bitmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Loop through contours and draw them
    for cnt in contours:
        # Turn the messy path into a clean one by approximating it
        epsilon = 0.02 * cv2.arcLength(cnt, True) # Tolerance (decrease for a tighter fit if needed)
        approx = (cv2.approxPolyDP(cnt, epsilon, True))
        # Draw the approx
        rect = (cv2.minAreaRect(cnt))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(box)

        left = min(box, key=lambda row: row[0])
        right = max(box, key=lambda row: row[0])

        # centroid
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


        if left[1] < right[1]:
            print("right")
        else:
            print("left")
        cv2.drawContours(image, [box], -1, (127, 10, 0), 3)
        cv2.circle(image, (int(cX), int(cY)), 3, (0, 255, 0), thickness=4)
        break

    cv2.imshow('done', image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


i = 0
while True:
    i += 10
    if i <= 180:
        get_edges(i)
    else:
        cv2.destroyAllWindows()
        sys.exit()


