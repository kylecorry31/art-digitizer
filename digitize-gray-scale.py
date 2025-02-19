import cv2
import numpy as np
import sys

def remove_paper_background(image, block_size=251, constant=10, hole_close_iterations=1):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Extract lines using adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant
    )

    # Fill small holes inside lines
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=hole_close_iterations)

    # Smooth edges of lines only (alpha mask)
    mask = cv2.GaussianBlur(closed, (3, 3), 0)

    # Convert non-black pixels to black pixels with an alpha value
    distance_from_black = np.sum(image, axis=2) / 3
    alpha = 255 - distance_from_black
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[:, :, 3] = alpha
    result[distance_from_black >= 254] = [0, 0, 0, 0]
    result[:, :, 3] = cv2.bitwise_and(result[:, :, 3], mask)

    return result

if len(sys.argv) != 3:
    print("Usage: script.py <input_image> <output_image>")
    sys.exit(1)

input_image = sys.argv[1]
output_image = sys.argv[2]
image = cv2.imread(input_image)

# Resize the image to a maximum of 2000x2000
if image.shape[0] > 2000 or image.shape[1] > 2000:
    scale = min(2000 / image.shape[1], 2000 / image.shape[0])
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (new_width, new_height))

# Parameters
min_image_dimension = max(image.shape[0], image.shape[1])
block_size = (min_image_dimension // 2) * 2 + 1
constant = 20
hole_close_iterations = 1

processed = remove_paper_background(image, block_size, constant, hole_close_iterations)

# If it is a JPG, add a white background
if output_image.endswith('.jpg'):
    alpha = processed[:, :, 3] / 255.0
    white_background = np.ones_like(processed[:, :, :3], dtype=np.uint8) * 255
    for c in range(3):
        white_background[:, :, c] = white_background[:, :, c] * (1 - alpha) + processed[:, :, c] * alpha
    processed = white_background

cv2.imwrite(output_image, processed)
