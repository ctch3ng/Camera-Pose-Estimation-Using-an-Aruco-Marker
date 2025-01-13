import cv2 as cv
import os
import numpy as np

# Checkerboard dimensions for display (number of squares)
DISPLAY_BOARD_DIM = (7, 10)

# Checkerboard dimensions for detection (number of internal corners)
CHESS_BOARD_DIM = (9, 6)

# Checkerboard square size in pixels (customise as needed)
SQUARE_SIZE = 79

# Checkerboard square size on screen (in mm) - a value obtained on a 24 inch monitor
SQUARE_SIZE_OS = 24

# Directory to save captured images
image_dir_path = "webcam_images"
# Directory to save calibration data
calib_data_path = "calib_data"

# Ensure the output directories exist
if not os.path.exists(image_dir_path):
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" directory created.')
else:
    print(f'"{image_dir_path}" directory already exists.')

if not os.path.exists(calib_data_path):
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" directory created.')
else:
    print(f'"{calib_data_path}" directory already exists.')

# Termination criteria for refining corner detection
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the checkerboard
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
obj_3D[:, :2] = np.mgrid[0:CHESS_BOARD_DIM[0], 0:CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
obj_3D *= SQUARE_SIZE_OS

# Lists to store object points and image points
obj_points_3D = []  # 3D points in real-world space
img_points_2D = []  # 2D points in image plane

# Function to generate a checkerboard image
def create_checkerboard(board_dimension, square_size, border_thickness=20):
    """
    Generate a checkerboard image with a thick white border to display on the screen.

    Args:
        board_dimension: Dimensions of the checkerboard (number of squares).
        square_size: Size of each square in pixels.
        border_thickness: Thickness of the white border in pixels.

    Returns:
        Checkerboard image with a white border.
    """
    rows, cols = board_dimension
    board_image = np.zeros(((rows * square_size) + 2 * border_thickness,
                            (cols * square_size) + 2 * border_thickness), dtype=np.uint8)

    # Fill the border with white
    board_image[:, :] = 255

    # Draw the checkerboard inside the border
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                y_start = row * square_size + border_thickness
                x_start = col * square_size + border_thickness
                board_image[y_start:y_start + square_size, x_start:x_start + square_size] = 0

    return board_image

# Function to detect and refine checkerboard corners
def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret

# Generate and display the checkerboard
checkerboard_image = create_checkerboard(DISPLAY_BOARD_DIM, SQUARE_SIZE, border_thickness=20)
cv.imshow("Checkerboard", checkerboard_image)

# Initialise webcam capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam.")

print("Press 's' to save an image when the checkerboard is detected.")
print("Press 'q' to quit.")

n = 0  # image_counter

while True:
    _, frame = cap.read()
    copyFrame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)
    # print(ret)
    cv.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.imshow("frame", frame)

    key = cv.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("s") and board_detected == True:
        # storing the checker board image
        cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)

        print(f"saved image number {n}")
        n += 1  # incrementing the image counter
    if n >= 45:
        break

# Release resources and close windows
cap.release()
cv.destroyAllWindows()
print("Total saved Images:", n)

files = os.listdir(image_dir_path)

for file in files:
    print(f"Processing file: {file}")
    imagePath = os.path.join(image_dir_path, file)
    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(grayScale, CHESS_BOARD_DIM, None)
    if ret:
        obj_points_3D.append(obj_3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)
        cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None)
print("Calibration successful.")

# Save calibration data to individual .npy files
np.save(f"{calib_data_path}/mtx.npy", mtx)    # Camera matrix
np.save(f"{calib_data_path}/dist.npy", dist)  # Distortion coefficients
print("Calibration data saved successfully as .npy files.")

mtx_loaded = np.load(f"{calib_data_path}/mtx.npy")
dist_loaded = np.load(f"{calib_data_path}/dist.npy")
print("Camera Matrix:\n", mtx_loaded)
print("Distortion Coefficients:\n", dist_loaded)
