# USAGE
# python getting_and_setting.py --image ../images/trex.png
# image[y, x] . Why does the y-coordinate come first? Well, keep in mind that an image is defined as a matrix.
# first supply y, the row number, followed by x, the column number
# Import the necessary packages
from __future__ import print_function
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image")
args = vars(ap.parse_args())

# Load the image and show it (handle loading the actual image off disk and displaying it to us.)
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Images are just NumPy arrays. The top-left pixel can be
# found at (0, 0)
(b, g, r) = image[219, 90] # we grab the pixel located at (0, 0) – the top- left corner of the image
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))
# quiz x=90, y=219 remember y,x
# Now, let's change the value of the pixel at (0, 0) and
# make it red (demonstrate that we have indeed successfully changed the color of the pixel)
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Since we are using NumPy arrays, we can apply slicing and
# grab large chunks of the image. Let's grab the top-left (100 × 100 pixel region of the image)
# corner (numpy start y : finnih y, start x : finnish x)
corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)

# Let's make the top-left corner of the image green
image[0:100, 0:100] = (0, 255, 0)

# Show our updated image
cv2.imshow("Updated", image)
cv2.waitKey(0)

