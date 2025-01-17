# check_sleap_tracks
Checks final sleap tracks for any errors for fly courtship behavior.





**Script to check output of the sleap tracks, and make sure there are not large errors:**
	- large jumps (and switching identity between animals)
	- large changes in head dir (error in identifying head-body orientation)
	- large backwards walking (another sign sleap has wrong orientation)
	- large wing extensions (error in animal identity; also barrel rolls from flies)
	- any NaN frames (these can usually be interpolated automatically with post-processing)

This script will only look at the .h5 file emitted from sleap (not the .slp file).

to run on command line:
>> python script_check_sleap_tracks.py ./filepath_to_data/ filename.h5

This script will save a .txt file (in the same directory) with the frame indices of uncertain frames, grouped.
It will display the top 20 frames for each group (e.g., largest jumps, etc.), as well as
	all frames with NaNs.
It will likely suffice to check the top 5 frames for each category and see what mistakes sleap is making.
 ---> There may be ways to automatically correct these errors with post-processing, especially
	if proofreading takes too long.
NOTE: The indices are not unique across groups, so there may be duplicates (e.g., a large jump and large change in head dir).

Also includes ability to plot the tracks at different frames. This will save .pdfs in same directory
	where each pdf is one frame:
	plot_tracks_for_frame(X, np.arange(iframe-10,iframe+10)) # saves 20 frames
This is helpful to see the tracks over time from sleap.

written by Ben Cowley, cowley@cshl.edu
	Jan 2025.
