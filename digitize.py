import cv2
import numpy as np
import argparse

def quantize_image(image, num_colors, background_color):
    # Create mask of non-transparent pixels
    non_transparent_mask = image[:, :, 3] > 0

    # Only process non-transparent pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    background_color = np.array([[[background_color[2], background_color[1], background_color[0]]]], dtype=np.uint8)
    background_color = cv2.cvtColor(background_color, cv2.COLOR_BGR2HSV)[0][0]

    # Reshape the image to be a list of pixels
    pixels = hsv.reshape((-1, 3))
    pixels = pixels[non_transparent_mask.flatten()]

    # Convert to float32 for k-means
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    compactness, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8
    centers = np.uint8(centers)

    # Calculate distance of each center from background color
    # Convert centers and background color to RGB for proper distance comparison
    centers_rgb = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR)
    background_rgb = cv2.cvtColor(np.array([[background_color]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
    color_distances = np.sqrt(np.sum((centers_rgb.reshape(-1,3) - background_rgb.reshape(-1,3))**2, axis=1))

    # Create mask for centers that are far enough from background
    # Using 50 as threshold distance (adjust as needed)
    valid_centers = color_distances > 0

    # Create mapping that maps invalid centers to background color
    center_mapping = np.arange(len(centers))
    center_mapping[~valid_centers] = -1

    # Map pixels to centers, using background color for invalid centers
    quantized = np.zeros_like(pixels, dtype=np.uint8)
    for i in range(len(centers)):
        if valid_centers[i]:
            mask = labels.flatten() == i
            quantized[mask] = centers[i]
        else:
            # Find the closest valid center or fill with the background color
            other_centers = centers[valid_centers]
            closest_valid_center = np.argmin(np.linalg.norm(centers[i] - other_centers, axis=1))
            mask = labels.flatten() == i
            quantized[mask] = other_centers[closest_valid_center]

    # Create output image with same dimensions as input
    quantized_image = np.zeros_like(hsv)
    quantized_image[non_transparent_mask] = quantized

    # Convert back to BGR for saving/displaying with cv2
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_HSV2BGR)
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2BGRA)

    # Copy alpha channel from original image
    quantized_image[:, :, 3] = image[:, :, 3]

    return quantized_image

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
        try:
            threshold_value = int(threshold_algorithm)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            thresh = cv2.threshold(
                blurred,
                threshold_value,
                255,
                cv2.THRESH_BINARY_INV
            )[1]
        except ValueError:
            # If threshold is not a number, use default threshold
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

    average_background_color = np.mean(image[distance_from_black < 254], axis=0)

    return (result, average_background_color)

def blend_lines(image, percent):
    # Create a binary mask of non-black pixels with a higher threshold
    # Convert to grayscale and apply Otsu's threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create masks for black and non-black pixels based on threshold
    non_black = (gray > thresh) & (image[:,:,3] > 0)
    black_pixels = (gray <= thresh) & (image[:,:,3] > 0)

    if np.any(black_pixels):
        # For each black pixel, find nearby non-black pixels
        y_coords, x_coords = np.where(black_pixels)
        for i in range(len(y_coords)):
            y, x = y_coords[i], x_coords[i]

            # Look in a square region, starting at 10 pixels and increasing until we have enough neighbors
            max_size = int(0.05 * min(image.shape[0], image.shape[1]))
            size = 10
            desired_neighbor_count = 30
            enough_neighbors = False

            while not enough_neighbors and size <= max_size:
                y_start = max(0, y-(size // 2))
                y_end = min(image.shape[0], y+(size // 2 + 1))
                x_start = max(0, x-(size // 2))
                x_end = min(image.shape[1], x+(size // 2 + 1))

                neighborhood = non_black[y_start:y_end, x_start:x_end]
                if np.sum(neighborhood) >= desired_neighbor_count:
                    enough_neighbors = True
                else:
                    size += 10

            neighborhood = non_black[y_start:y_end, x_start:x_end]
            if np.any(neighborhood):
                # Get colors of nearby non-black pixels
                nearby_colors = image[y_start:y_end, x_start:x_end][neighborhood > 0]
                avg_color = np.mean(nearby_colors, axis=0)
                # Make it darker
                image[y,x] = (avg_color * (percent / 100)).astype(np.uint8)
                image[y,x,3] = max(127, avg_color[3])
            else:
                image[y,x,3] = np.array([0, 0, 0])

    return image

parser = argparse.ArgumentParser(description='Process an image to remove paper background.')
parser.add_argument('input_image', help='Input image file')
parser.add_argument('output_image', help='Output image file')
parser.add_argument('--color', action='store_true', help='Use original colors')
parser.add_argument('--binary', action='store_true', help='Apply binary threshold')
parser.add_argument('--background', action='store_true', help='Use a white background')
parser.add_argument('--threshold', type=str, default='otsu', help='Thresholding method')
parser.add_argument('--padding', type=int, default=100, help='Padding in pixels around the drawing')
parser.add_argument('--rotate', type=float, default=0.0, help='Rotation angle in degrees')
parser.add_argument('--thin', action='store_true', help='Apply line thinning')
parser.add_argument('--quantize', type=int, default=None, help='Number of colors to quantize to')
parser.add_argument('--blend-lines', type=int, default=None, help='Replace black lines with darker shades of nearby colors. Pass in the percent (integer).')

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

processed, background_color = remove_paper_background(image, hole_close_iterations, args.threshold, args.thin)

if args.color:
    processed[:, :, :3] = image[:, :, :3]

if args.binary:
    processed[:, :, 3] = np.where(processed[:, :, 3] > 0, 255, 0)

if args.quantize:
    processed = quantize_image(processed, args.quantize, background_color)

if args.blend_lines:
    processed = blend_lines(processed, args.blend_lines)

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
