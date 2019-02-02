# Vision2019

Python version: 3.5.2
OpenCV version: 4.0.1

This vision code is designed to locate the vision target and find its pose relative to the robot (i.e angle + direction)

See main.py for usage

# Learn OpenCV
- [Python](https://docs.opencv.org/4.0.1/d6/d00/tutorial_py_root.html)

## Setup

1. Install Python 3.6 and OpenCV on your system
    * Make sure that your OpenCV installation can be used from the global Python scope. 
1. `python3.6 -m venv venv/`
    * This creates a virtual environment (essentially a sandbox) for this particular project. It helps prevent contamination of your global python space
1. `pip install -r requirements.txt`
    * This installs all the dependencies that are not OpenCV
1. Create a symlink between your global install of OpenCV and `venv/lib/python3.6/site-packages`
    * Chances are that the global installation of OpenCV is at `/usr/local/lib/python3.6/site-packages/cv2`
    * To create the symlink, run `ln -s /usr/local/lib/python3.6/site-packages/cv2 venv/lib/python3.6/site-packages` from the `Vision2019` directory
    * Depending on how you installed OpenCV your global installation may be located somewhere else
1. Try running `import cv2` in Python
    * If it works, you're all set!

### Camera Calibration


## Running it

`python3 main.py`

### Flags

* `--no_sockets`
    * Prevents `main.py` from trying to send data to a socket client
    * Usage: `python3 main.py --no_sockets`
    
### Environment Variables

* `V19_CAMERA_ID`
    * Sets the camera ID i.e the number after `/dev/video`
    * Usage: `V19_CAMERA_ID=0 python3 main.py`
    * Default: `4`
* `V19_PORT`
    * Sets the port that the program will try to send data to
    * Usage: `V19_PORT=2502 python3 main.py`
    * Default: `5800`
* `V19_CALIBRATION_FILE_LOCATION`
    * Sets the location of the `.pickle` file that the program will look for calibration info in
    * Usage: `V19_CALIBRATION_FILE_LOCATION=calibration_info.pickle python3 main.py`
    * Default: `prod_camera_calib.pickle`
    
    
# Files

There are a bunch of files. Not all of them are used while `main.py` is running.

* `main.py`
    * Main entrypoint of program
    * Reads image directly from camera, calculates relative position (ft) + angle (radians) of vision target, 
    will try to send data over TCP socket in `|x,y,angle|` form
* `constants.py`
    * Contains most of the constants relevant to vision processing (e.g camera ID)
* `pipeline.py`
    * Defines the `VisionPipeline` class, which contains all the relevant logic for processing an image
* `frc.py`
    * Playground for testing things. Contains some useful code for development.
* `getting_contours_test.py`, `test.py`
    * Used for testing.
* `calibrate.py`
    * Used for calibration. Very important for accurate localization
    * Usage
        1. Print out`checkerboard.jpg` which is located in the root folder of this project
        1. Tape the checkerboard pattern to a stiff flat surface, like a cereal box or wood.
            * If the pattern is bent the calibration results will be wrong. 
        1. Run `python3 calibrate.py`. The same environment variables for `main.py` apply. 
            * To specify the file name to save the calibration info to, use the `V19_CALIBRATION_FILE_LOCATION`
            environment variable as shown above.
        1. Move the checkerboard around the camera's field of view.
        Make sure to rotate it and angle it in different ways for best results.
        1. Press the space bar when you want to take a picture. 
* `socket_client.py`
    * Used as a test socket client to listen to messages from `main.py`.
* `color_range_finder.py`
    * Used to find adequate HSV ranges for `cv2.inRange` 
