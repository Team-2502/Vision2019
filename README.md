# Vision2019

Python version: 3.6
OpenCV version: 4.0.1

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

Useful links:
- [OpenCV ReadTheDocs](https://opencv-java-tutorials.readthedocs.io/en/latest/03-first-javafx-application-with-opencv.html)
