# Vision2019

Python version: 3.6
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
