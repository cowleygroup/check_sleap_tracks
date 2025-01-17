

# Script to check output of the sleap tracks, and make sure there are not large errors:
#	- large jumps (and switching identity between animals)
#	- large changes in head dir (error in identifying head-body orientation)
#	- large backwards walking (another sign sleap has wrong orientation)
#	- large wing extensions (error in animal identity; also barrel rolls from flies)
#	- any NaN frames (these can usually be interpolated automatically with post-processing)

# This script will only look at the .h5 file emitted from sleap (not the .slp file).

# to run on command line:
# >> python script_check_sleap_tracks.py ./filepath_to_data/ filename.h5

# This script will save a .txt file (in the same directory) with the frame indices of uncertain frames, grouped.
# It will display the top 20 frames for each group (e.g., largest jumps, etc.), as well as
#	all frames with NaNs.
# It will likely suffice to check the top 5 frames for each category and see what mistakes sleap is making.
#  ---> There may be ways to automatically correct these errors with post-processing, especially
#	if proofreading takes too long.
# NOTE: The indices are not unique across groups, so there may be duplicates (e.g., a large jump and large change in head dir).
#
# Also includes ability to plot the tracks at different frames. This will save .pdfs in same directory
#	where each pdf is one frame:
#	plot_tracks_for_frame(X, np.arange(iframe-10,iframe+10)) # saves 20 frames
# This is helpful to see the tracks over time from sleap.
#
# written by Ben Cowley, cowley@cshl.edu
#	Jan 2025.

import numpy as np
import h5py
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


### HELPER FUNCTIONS

def compute_head_dir(X):
	# X: (2,3)
	ihead=0; ibody=1; itail=2;

	head_dir = X[:,ihead] - X[:,ibody]
	head_dir = head_dir / np.sqrt(np.sum(head_dir**2))
	return head_dir


def compute_angle_head_dir(X1, X2):
	# X1: (2,3) [x/y; head/body/tail]
	ihead=0; ibody=1; itail=2;

	head_dir1 = X1[:,ihead] - X1[:,ibody]
	head_dir1 = head_dir1 / np.sqrt(np.sum(head_dir1**2))

	head_dir2 = X2[:,ihead] - X2[:,ibody]
	head_dir2 = head_dir2 / np.sqrt(np.sum(head_dir2**2))

	ang = np.arccos(np.sum(head_dir1 * head_dir2)) / np.pi * 180.

	return ang


def plot_tracks_for_frame(X, frames):
	for iframe in frames:
		f = plt.figure()

		# arena
		center_x = 550
		center_y = 420
		radius=400
		circle = Circle((center_x, center_y), radius, color='black', fill=False)
		plt.gca().add_patch(circle)

		# female
		for ijoint in range(4):
			plt.plot(X[0,0,ijoint,iframe], X[0,1,ijoint,iframe], '.r')
		plt.plot([X[0,0,0,iframe], X[0,0,1,iframe]], [X[0,1,0,iframe], X[0,1,1,iframe]], '-r') # body-head
		plt.plot([X[0,0,1,iframe], X[0,0,2,iframe]], [X[0,1,1,iframe], X[0,1,2,iframe]], '-r') # body-left wing
		plt.plot([X[0,0,1,iframe], X[0,0,3,iframe]], [X[0,1,1,iframe], X[0,1,3,iframe]], '-r') # body-right wing

		# male
		for ijoint in range(4):
			plt.plot(X[1,0,ijoint,iframe], X[1,1,ijoint,iframe], '.b')
		plt.plot([X[1,0,0,iframe], X[1,0,1,iframe]], [X[1,1,0,iframe], X[1,1,1,iframe]], '-b') # body-head
		plt.plot([X[1,0,1,iframe], X[1,0,2,iframe]], [X[1,1,1,iframe], X[1,1,2,iframe]], '-b') # body-left wing
		plt.plot([X[1,0,1,iframe], X[1,0,3,iframe]], [X[1,1,1,iframe], X[1,1,3,iframe]], '-b') # body-right wing

		plt.xlim([0,960])
		plt.ylim([0,960])

		plt.axis('equal')

		plt.title('frame {:d}'.format(iframe))
		f.savefig('./frame{:d}.pdf'.format(iframe))

		plt.close('all')



### MAIN SCRIPT



h5_filepath = sys.argv[1]  # file path to file...e.g., '/home/cowley/research/project_knockout/tracking/sample_data/'
h5_filename = sys.argv[2]  # file name for h5 tracks... e.g., '20240716_140713_left.analysis.h5'

# will output a text file '{h5_filename}_weird_frames.txt' with list of weird frames to proofread


## simple checks
if True:
	if h5_filepath[-1] != '/': # append forward slash if needed
		h5_filepath = h5_filepath + '/'

	if h5_filename[-3:] != '.h5':
		print('h5_filename (2nd argument) needs to be h5 file!')
		exit(1)


## load data
if True:
	sexes = ['female', 'male']  # assumes female in first index, male in second
	ifemale = 0; imale = 1

	data = h5py.File(h5_filepath + h5_filename, 'r')

	scores = np.array(data['tracking_scores'])
		# (2,N) for female/male and N frames
		# score of 0 --> ground truth label

	X = np.array(data['tracks'])
	X[:,1,:,:] = 960-X[:,1,:,:]  # flips y to be correctly oriented with videos
	#		(2,2,3,N) where:
	#		tracks[i,:,:,:] --> female if i=0, male if i=1
	#		tracks[:,j,:,:] --> x if i=0, y if i=1
	#		tracks[:,:,k,:] --> head if k=0, thorax if k=1, abdomen if k=2
	#		tracks[:,:,:,n] --> nth frame

	data.close()

	num_frames = X.shape[-1]

	ix=0; iy=1;
	ihead=0; ibody=1; ileft_wing=2; iright_wing=3

	indices_nan_tracks = np.isnan(np.sum(X[:,:,:,:], axis=(0,1,2)))

	indices_human_labels = np.isnan(np.sum(scores[:,:], axis=0)) * ~indices_nan_tracks


## check for nans/human labels
if True:
	print('X tracks shape:')
	print(X.shape)

	print('number of total frames = {:d}'.format(num_frames))

	num_nan_frames = np.sum(indices_nan_tracks)
	print('number of frames with nans: {:d}'.format(np.sum(indices_nan_tracks)))

	if np.sum(indices_nan_tracks[:num_nan_frames]) < num_nan_frames or np.sum(indices_nan_tracks[-num_nan_frames:]) < num_nan_frames:
		print('!!! WARNING: nan frames occur within video, not simply at beginning or end. !!!')

	print('number of human labels: {:d}'.format(np.sum(indices_human_labels)))

	inds_nan_frames = np.flatnonzero(indices_nan_tracks)


## check for jumps
if True:
	frames_jumps = []

	dists_female = np.sqrt(np.sum((X[ifemale,:,ibody,1:] - X[ifemale,:,ibody,:-1])**2,axis=0))
	dists_male = np.sqrt(np.sum((X[imale,:,ibody,1:] - X[imale,:,ibody,:-1])**2,axis=0))

	dists = np.concatenate([dists_female[np.newaxis,:], dists_male[np.newaxis,:]], axis=0)
	dists = np.max(dists,axis=0)

	dists[np.isnan(dists)] = 0.

	inds_jumps = np.argsort(dists)[::-1]



## check for large changes in head dir
if True:
	angs_head_dir_female = np.zeros((num_frames-1,))
	for iframe in range(num_frames-1):
		angs_head_dir_female[iframe] = compute_angle_head_dir(X[ifemale,:,:,iframe], X[ifemale,:,:,iframe+1])

	angs_head_dir_male = np.zeros((num_frames-1,))
	for iframe in range(num_frames-1):
		angs_head_dir_male[iframe] = compute_angle_head_dir(X[imale,:,:,iframe], X[imale,:,:,iframe+1])

	angs_head_dir = np.concatenate([angs_head_dir_female[np.newaxis,:], angs_head_dir_male[np.newaxis,:]], axis=0)
	angs_head_dir[np.isnan(angs_head_dir)] = 0.
	angs_head_dir = np.max(angs_head_dir, axis=0)

	inds_change_head_dir = np.argsort(angs_head_dir)[::-1]



## check for large backwalking (signs of misoriented head/body)
if True:
	projs_head_dir_vs_vel = np.zeros((2,num_frames-1))

	## female
	for iframe in range(num_frames-1):
		head_dir = compute_head_dir(X[ifemale,:,:,iframe])
		vel_dir = X[ifemale,:,ibody,iframe+1] - X[ifemale,:,ibody,iframe]

		projs_head_dir_vs_vel[ifemale,iframe] = np.sum(head_dir * vel_dir)

	## male
	for iframe in range(num_frames-1):
		head_dir = compute_head_dir(X[imale,:,:,iframe])
		vel_dir = X[imale,:,ibody,iframe+1] - X[imale,:,ibody,iframe]

		projs_head_dir_vs_vel[imale,iframe] = np.sum(head_dir * vel_dir)

	forward_vels = np.max(projs_head_dir_vs_vel, axis=0)

	inds_backwards_walking = np.argsort(forward_vels)


## check for wing extension
if True:
	dists_between_wings = np.zeros((2,num_frames))

	dists_between_wings[0] = np.sqrt(np.sum((X[ifemale,:,2,:] - X[ifemale,:,3,:])**2,axis=0))
	dists_between_wings[1] = np.sqrt(np.sum((X[imale,:,2,:] - X[imale,:,3,:])**2,axis=0))

	if np.nanmax(dists_between_wings[0]) > np.nanmax(dists_between_wings[1]):
		print('!!! WARNING: Sex of fly may be mixed up...first fly appears to have wing extensions for song? !!!')

	dists_between_wings = np.max(dists_between_wings, axis=0)
	dists_between_wings[np.isnan(dists_between_wings)] = 0.

	inds_large_wing_extensions = np.argsort(dists_between_wings)[::-1]


## save inds in text file
if True:
	with open('./' + h5_filename[:-3] + '_uncertain_frames.txt', 'w') as file:

		file.write('Uncertain frames for {:s}\n'.format(h5_filename))
		file.write('\n')

		file.write('frames with jumps:\n')
		for iframe in inds_jumps[:20]:
			time = iframe * (1/25)
			hrs = np.floor(time/60/60).astype('int')
			mins = np.mod(np.floor(time/60), 60)
			seconds = np.mod(np.floor(time),60)
			file.write('{:d}, dist={:.02f}, {:d}:{:02.0f}:{:02.0f}\n'.format(iframe, dists[iframe], hrs, mins, seconds))
		file.write('\n')

		file.write('frames with large changes in head direction:\n')
		for iframe in inds_change_head_dir[:20]:
			time = iframe * (1/25)
			hrs = np.floor(time/60/60).astype('int')
			mins = np.mod(np.floor(time/60), 60)
			seconds = np.mod(np.floor(time),60)
			file.write('{:d}, ang={:.01f}, {:d}:{:02.0f}:{:02.0f}\n'.format(iframe, angs_head_dir[iframe], hrs, mins, seconds))
		file.write('\n')

		file.write('frames with backwards walking:\n')
		for iframe in inds_backwards_walking[:20]:
			time = iframe * (1/25)
			hrs = np.floor(time/60/60).astype('int')
			mins = np.mod(np.floor(time/60), 60)
			seconds = np.mod(np.floor(time),60)
			file.write('{:d}, for. vel.={:.02f}, {:d}:{:02.0f}:{:02.0f}\n'.format(iframe, forward_vels[iframe], hrs, mins, seconds))
		file.write('\n')

		file.write('frames with large wing extensions:\n')
		for iframe in inds_large_wing_extensions[:20]:
			time = iframe * (1/25)
			hrs = np.floor(time/60/60).astype('int')
			mins = np.mod(np.floor(time/60), 60)
			seconds = np.mod(np.floor(time),60)
			file.write('{:d}, dists_wing={:.02f}, {:d}:{:02.0f}:{:02.0f}\n'.format(iframe, dists_between_wings[iframe], hrs, mins, seconds))
		file.write('\n')

		file.write('frames with NaNs:\n')
		for iframe in inds_nan_frames:
			time = iframe * (1/25)
			hrs = np.floor(time/60/60).astype('int')
			mins = np.mod(np.floor(time/60), 60)
			seconds = np.mod(np.floor(time),60)
			file.write('{:d}, {:d}:{:02.0f}:{:02.0f}\n'.format(iframe, hrs, mins, seconds))
		file.write('\n')
