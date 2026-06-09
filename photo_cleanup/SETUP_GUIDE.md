Check OpenCV is installed and working (version 4.13.0). Here's the brief command-line guide.
Setup (once)

Install Python 3 from https://www.python.org/downloads/ — on Windows, check "Add python.exe to PATH" during install.
Install the one library:

Windows: pip install opencv-python
Mac: pip3 install opencv-python


Verify it installed:

Windows: python -c "import cv2; print(cv2.__version__)"
Mac: python3 -c "import cv2; print(cv2.__version__)"

If it prints a version number (e.g. 4.13.0), you're good. If it errors, the install didn't take — re-run step 2.

Run it
Point the command at the script, then your photo folder in quotes:

Windows: python "C:\path\to\sort_photos.py" "G:\My Drive\Equipment Photos\_Unsorted"
Mac: python3 ~/Desktop/sort_photos.py "/Users/you/Google Drive/My Drive/Equipment Photos/_Unsorted"


For group_duplicates:
pip install imagehash (Mac: pip3). 
Run: python group_duplicates.py "/path/to/photos" searches subfolders too.
--threshold controls strictness. Default is 3 (same shot, tiny differences). If it's missing obvious duplicates, bump it up (--threshold 6); if it's wrongly lumping different photos together, drop it (--threshold 1).