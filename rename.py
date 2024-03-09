# function to rename all jpgs in a directory

import os

directory = "./gestures/click"

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        os.rename(os.path.join(directory, filename), os.path.join(directory, filename[:-4] + "-click.jpg"))