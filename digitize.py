import cv2
import numpy as np
import argparse

def remove_paper_background(image, hole_close_iterations=1, threshold_algorithm='otsu', thin_lines=False):
    # Convert to grayscale
    if threshold_algorithm == 'otsu':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
    elif threshold_algorithm == 'hsv':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1]
        s = np.where(s < 127, 0, 1)
        v = (hsv[:,:,2] + 127) % 255
        v = np.where(v > 127, 1, 0)
        thresh = np.where(s + v > 0, 255, 0).astype(np.uint8)
    elif threshold_algorithm == 'contour':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find initial contours
        ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create mask and fill contours
        thresh = np.zeros_like(gray)
        cv2.drawContours(thresh, contours, -1, (255,255,255), -1)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(
            blurred,
            127,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

    # Fill small holes inside lines
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=hole_close_iterations)

    if thin_lines:
        # Apply thinning using Zhang-Suen algorithm
        closed = cv2.ximgproc.thinning(closed)

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

parser = argparse.ArgumentParser(description='Process an image to remove paper background.')
parser.add_argument('input_image', help='Input image file')
parser.add_argument('output_image', help='Output image file')
parser.add_argument('--color', action='store_true', help='Use original colors')
parser.add_argument('--binary', action='store_true', help='Apply binary threshold')
parser.add_argument('--background', action='store_true', help='Use a white background')
parser.add_argument('--threshold', type=str, default='otsu', choices=['otsu', 'hsv', 'binary', 'contour'], help='Thresholding method')
parser.add_argument('--padding', type=int, default=100, help='Padding in pixels around the drawing')
parser.add_argument('--rotate', type=float, default=0.0, help='Rotation angle in degrees')
parser.add_argument('--thin', action='store_true', help='Apply line thinning')

args = parser.parse_args()

image = cv2.imread(args.input_image)

# Resize the image to a maximum of 2000x2000
if image.shape[0] > 2000 or image.shape[1] > 2000:
    scale = min(2000 / image.shape[1], 2000 / image.shape[0])
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (new_width, new_height))

# Parameters
min_image_dimension = max(image.shape[0], image.shape[1])
hole_close_iterations = 1

processed = remove_paper_background(image, hole_close_iterations, args.threshold, args.thin)

if args.color:
    processed[:, :, :3] = image[:, :, :3]

if args.binary:
    processed[:, :, 3] = np.where(processed[:, :, 3] > 0, 255, 0)

# Rectangular crop of the image to contain the non-transparent pixels
non_zero_indices = np.argwhere(processed[:, :, 3] > 0)
if non_zero_indices.size > 0:
    y_min, x_min = non_zero_indices.min(axis=0)
    y_max, x_max = non_zero_indices.max(axis=0)
    processed = processed[y_min:y_max+1, x_min:x_max+1]

# Pad the image with transparent black pixels
processed = cv2.copyMakeBorder(processed, args.padding, args.padding, args.padding, args.padding, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

if args.rotate:
    center = (processed.shape[1] // 2, processed.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, args.rotate, 1.0)
    processed = cv2.warpAffine(processed, rotation_matrix, (processed.shape[1], processed.shape[0]))

# If it is a JPG, add a white background
if args.output_image.endswith('.jpg') or args.background:
    alpha = processed[:, :, 3] / 255.0
    white_background = np.ones_like(processed[:, :, :3], dtype=np.uint8) * 255
    for c in range(3):
        white_background[:, :, c] = white_background[:, :, c] * (1 - alpha) + processed[:, :, c] * alpha
    processed = white_background

cv2.imwrite(args.output_image, processed)
