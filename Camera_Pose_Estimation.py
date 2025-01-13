import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load pre-calibrated camera matrix and distortion coefficients
mtx = np.load("calib_data/mtx.npy")
dist = np.load("calib_data/dist.npy")

# Define the ArUco dictionary and detection parameters
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
ARUCO_PARAMETERS = aruco.DetectorParameters()
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

# Marker size in metres (adjust based on your setup)
marker_length = 0.16

# Define a rotation matrix for coordinate system alignment
rot_x_90 = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

def estimate_pose(corners, marker_size, camera_matrix, distortion):
    """
    Estimate the pose of ArUco markers using corners detected in the image.
    Returns rotation vectors and translation vectors.
    """
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    rvecs, tvecs = [], []

    for corner in corners:
        _, rvec, tvec = cv2.solvePnP(marker_points, corner, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(rvec)
        tvecs.append(tvec)

    return rvecs, tvecs

def draw_gripper(ax, position, rotation, scale=0.3):
    """
    Visualise a simple gripper at the given position and orientation in 3D space.
    """
    joint_positions = np.array([
        [-1, 0, 0], [1, 0, 0],
        [-1, 1, 0], [1, 1, 0],
        [0, 0, -1]
    ]) * scale

    joint_positions = np.dot(joint_positions, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    joint_positions_rotated = np.dot(rotation, joint_positions.T).T
    joint_positions_world = position + joint_positions_rotated

    ax.plot(joint_positions_world[:2, 0], joint_positions_world[:2, 1], joint_positions_world[:2, 2], 'k-', lw=2)
    ax.plot([joint_positions_world[0, 0], joint_positions_world[2, 0]],
            [joint_positions_world[0, 1], joint_positions_world[2, 1]],
            [joint_positions_world[0, 2], joint_positions_world[2, 2]], 'k-')
    ax.plot([joint_positions_world[1, 0], joint_positions_world[3, 0]],
            [joint_positions_world[1, 1], joint_positions_world[3, 1]],
            [joint_positions_world[1, 2], joint_positions_world[3, 2]], 'k-')
    ax.plot([position[0], joint_positions_world[4, 0]],
            [position[1], joint_positions_world[4, 1]],
            [position[2], joint_positions_world[4, 2]], 'k-')

def setup_plot():
    """
    Set up the Matplotlib figure and subplots for video feed and 3D visualisation.
    """
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Real-Time ArUco Marker Detection and Pose Estimation (press 'q' to quit)", fontsize=16)

    ax_video = fig.add_subplot(121)
    ax_video.set_title("Webcam Feed")
    
    ax_3d = fig.add_subplot(122, projection='3d')
    ax_3d.set_title("3D Camera Pose")
    ax_3d.set_xlim([-2, 2])
    ax_3d.set_ylim([-2, 2])
    ax_3d.set_zlim([-2, 2])
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")

    plt.show(block=False)
    return fig, ax_video, ax_3d

# Initialise video capture and plotting
cam = cv2.VideoCapture(0)
fig, ax_video, ax_3d = setup_plot()

while cam.isOpened():
    ret, frame = cam.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if len(corners) > 0:
            rvecs, tvecs = estimate_pose(corners, marker_length, mtx, dist)
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

            for rvec, tvec in zip(rvecs, tvecs):
                frame = cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length / 2)
                R, _ = cv2.Rodrigues(rvec)
                xyz = np.dot(np.dot(rot_x_90, R.T), -tvec).squeeze()

                ax_3d.clear()
                ax_3d.set_xlim([-2, 2])
                ax_3d.set_ylim([-2, 2])
                ax_3d.set_zlim([-2, 2])
                ax_3d.set_xlabel("x")
                ax_3d.set_ylabel("y")
                ax_3d.set_zlabel("z")
                ax_3d.scatter(0, 0, 0, color="k")
                ax_3d.quiver(0, 0, 0, 1, 0, 0, length=1, color="r")
                ax_3d.quiver(0, 0, 0, 0, 0, 1, length=1, color="g")
                ax_3d.quiver(0, 0, 0, 0, -1, 0, length=1, color="b")
                draw_gripper(ax_3d, xyz, np.dot(rot_x_90, R.T))

        ax_video.clear()
        ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_video.axis('off')

        fig.canvas.draw()
        fig.canvas.flush_events()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
plt.close(fig)