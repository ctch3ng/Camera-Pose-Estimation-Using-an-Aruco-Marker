# Camera-Pose-Estimation-Using-an-Aruco-Marker

## Prerequisites
This tutorial assumes that you already have Anaconda installed on your system. If not, please download and install it from [Anaconda’s official website](https://www.anaconda.com/download).

## Step 1: Create a Virtual Environment
- To ensure that the program runs in a clean and isolated environment, we’ll create a virtual environment using Anaconda.

- Open a terminal (or Anaconda Prompt).

- Create a virtual environment named `py38_cv` with `Python 3.8`:
```bash
conda create -n py38_cv python=3.8 -y
```
- Activate the virtual environment:
```bash
conda activate py38_cv
```

## Step 2: Install Required Python Packages
- Once the virtual environment is activated, install the necessary Python packages using `pip`:
```bash
pip install opencv-python numpy matplotlib
```
- Install `git` using `conda`:
```bash
conda install git
```
## Step 3: Clone this Repository
```bash
git clone https://github.com/ctch3ng/Camera-Pose-Estimation-Using-an-Aruco-Marker.git
```
- Change directory
```
cd Camera-Pose-Estimation-Using-an-Aruco-Marker
```
## Step 4: Camera Calibration
- Before running the main program, calibrate your camera using the provided `Image_Capture_Camera_Calibration.py` script.
```
python Image_Capture_Camera_Calibration.py
```
- A checkerboard will appear on the screen. Measure the size of a block (mm) on the screen and adjust the `SQUARE_SIZE_OS` constant in the script accordingly. (i.e. measure it, then hit Ctrl-C to terminate the program. Afterwards, update the parameter and then run the script again.)
- The script will prompt you to capture 45 images by pressing `s` when the checkerboard is detected.
- Once 45 images are captured, the program will compute the calibration matrix and save it in the `calib_data` folder.

## Step 5: Run the Camera Pose Estimation Program
- Run the `Camera_Pose_Estimation.py` script:
```
python Camera_Pose_Estimation.py
```
- The program will look for an Aruco marker with the attribute `DICT_5X5_1000`. A sample marker (`DICT_5X5_1000_ID2.png`) is provided in the repository.

- Display the marker on a screen (with a white background) and point your webcam towards it.

- You should see a 3D stick figure representing the camera’s position relative to the marker.

## Acknowledgements

I would like to thank Chinmay Prashant Kashid and Peter George for their contributions to the earlier versions of this code. Additionally, credit goes to the repository [naruya/aruco](https://github.com/naruya/aruco) for inspiration and reference.
