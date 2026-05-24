import cv2
import numpy as np

# Load image in grayscale
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Define kernel
kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]], dtype=np.uint8)

# 1. Dilation
dilated = cv2.dilate(image, kernel, iterations=1)

# 2. Morphological Closing
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# 3. Black Hat Transformation
blackhat = cv2.morphologyEx(closed, cv2.MORPH_BLACKHAT, kernel)

# 4. Median Blur (5x5)
blurred = cv2.medianBlur(blackhat, 5)

# 5. Horizontal Flip
flipped = cv2.flip(blurred, 1)

# Save output matrix to submission.txt
np.savetxt(
    'submission.txt',
    flipped,
    fmt='%d',
    delimiter=','
)
