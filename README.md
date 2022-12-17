# human-movement-detection

# Running the skeletonization on a video
Download the AOF skeletons code and the main.py file. This should be run from the command line.  
Make sure that the python MATLAB engine is enabled wherever you run the file from.  
Run the command `python [path to file]/main.py --video [path to video]/video`.  
This will create 3 video feeds on top of each other: Frame Delta, Thresh (binary image), and the Skeletonized image.

# Running the different optical flow algorithms on test videos
Download the repository containing the different flow algorithms present in the `scripts` folder.
All files can be executed from the command line. Simply run `python <file path> --video <path to video>` to run any of the algorithms. There are two example on which the algorithms can be tested. Both files, and partial results, can be found in the `examples` folder.