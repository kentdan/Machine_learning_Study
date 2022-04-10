# USAGE
# python challenge_checker.py
# Import the necessary packages
import numpy as np
import cv2
from itertools import product  # basically for for

# read about itertools and functools
# faster checker
canvas = np.zeros((300, 300, 3), dtype="uint8")  # 300 x 300 y 3 channel rgb
temp = np.arange(0, 90000).reshape(-1, 300)
 #tiap row add 1 more
where = (temp + temp // 300) & 1
print(where)
red = (0, 0, 255)
size = 10
x = np.arange(0, 310, 2 * size)  # dumb but work
for x1, y1 in product(x, x - size):
    cv2.rectangle(canvas, (x1, y1), (x1 + size, y1 + size), red, -1)
    cv2.rectangle(canvas, (x1 + size, y1 + size), (x1 + 2 * size, y1 + 2 * size), red, -1)
# then loop
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
cv2.circle(canvas, (centerX, centerY), 50, (0, 255, 0), -1)
# circle where,center coordinate,radius,color,thickness
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
