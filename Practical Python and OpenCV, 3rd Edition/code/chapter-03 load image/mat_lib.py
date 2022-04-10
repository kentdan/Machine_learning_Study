# Import the necessary packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
image = mpimg.imread("newimage.jpg")
plt.axis("off")
plt.imshow(image)
plt.show()
#matplot to matplot
image1 = cv2.imread("newimage.jpg")
plt.axis("off")
plt.imshow(image1)
plt.show()
#cv2 to matplot there is something wrong bcs cv2 is BGR but matplotlib is RGB
plt.axis("off")
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.show()