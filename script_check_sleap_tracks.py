

# Script to check output of the sleap tracks, and make sure there are not large errors:
# 	- large jumps (and switching identity between animals)
# 	- large changes in head dir (error in identifying head-body orientation)
# 	- large backwards walking (another sign sleap has wrong orientation)
# 	- large wing extensions (error in animal identity; also barrel rolls from flies)
# 	- any NaN frames (these can usually be interpolated automatically with post-processing)

# This script will only look at the .h5 file emitted from sleap (not the .slp file).

# to run on command line:
# >> python script_check_sleap_tracks.py ./filepath_to_data/ filename.h5

# This script will save a .txt file (in the same directory) with the frame indices of uncertain frames, grouped.
# It will display the top 20 frames for each group (e.g., largest jumps, etc.), as well as
# 	all frames with NaNs.
# It will likely suffice to check the top 5 frames for each category and see what mistakes sleap is making.
#  ---> There may be ways to automatically correct these errors with post-processing, especially
# 	if proofreading takes too long.
# NOTE: The indices are not unique across groups, so there may be duplicates (e.g., a large jump and large change in head dir).

# This script will also produce 6 useful plots describing position and courtship behavior.
# This includes:
#	- male-to-female distance over time
#	- male/female velocity magnitude over time
#	- male/female position density (mostly on the edge of the chamber)
#	- male/female egocentric density (ego fly is in middle facing upwards;
#			we expect female to mostly be directly in front of male)

# Also includes ability to plot the tracks at different frames. This will save .pdfs in same directory
# 	where each pdf is one frame:
# 	plot_tracks_for_frame(X, np.arange(iframe-10,iframe+10)) # saves 20 frames
# This is helpful to see the tracks over time from sleap.
# (This is optional and needs to be activated to work.)

# written by Ben Cowley, cowley@cshl.edu
# 	Jan 2025.

import numpy as np
import h5py
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse


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



def get_relative_positions(positions_centric_fly, positions_noncentric_fly):
	# helper function to plot egocentric position densities
	# Difference in positions and rotated in respect to the centric fly (i.e., the fly in the center)
	#
	# Inputs:
	#	positions_centric_fly: (2,3,T) for (x/y, head/body/tail, num_timepoints) joint positions of centric fly (i.e., the fly who will be in the center and whose
	#		head direction will be facing upwards). Same format as for male_positions and female_positions. 
	#		See documentation for plot_distance_over_time.
	#	positions_noncentric_fly: (2,3,T) for (x/y, head/body/tail, num_timepoints), joint positions noncentric fly (fly whose position is of interest relative to
	#		the perspective of the centric fly). Same format as positions_centric_fly.
	#
	# Outputs:
	#	relative_positions: (2,3,T) for (x/y, head/body/tail, num_timepoints), relative positions of noncentric_fly with respect to centric_fly. 
	#		Same format as positions_centric_fly and positions_noncentric_fly.
	
	num_flies = len(positions_centric_fly)

	upward_dir = np.array([0,1])
	rightward_dir = np.array([1,0])

	relative_positions = positions_noncentric_fly - positions_centric_fly[:,1,:][:,np.newaxis,:]

	head_dirs = positions_centric_fly[:,0,:] - positions_centric_fly[:,1,:]
	head_dirs = head_dirs / (np.sqrt(np.sum(head_dirs**2,axis=0))+1e-5)  # normalize to have length 1

	angles = np.arccos(np.dot(upward_dir, head_dirs)) * np.sign(np.dot(rightward_dir, head_dirs))

	# rotate relative positions 
	num_timepoints = relative_positions.shape[1]
	for itime in range(num_timepoints):
		R = np.array([[np.cos(angles[itime]), -np.sin(angles[itime])], [np.sin(angles[itime]), np.cos(angles[itime])]])
		relative_positions[:,0,itime] = np.dot(R, relative_positions[:,0,itime])
		relative_positions[:,1,itime] = np.dot(R, relative_positions[:,1,itime])

	return relative_positions






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

	X_tracks = np.array(data['tracks'])
	X_tracks[:,1,:,:] = 960-X_tracks[:,1,:,:]  # flips y to be correctly oriented with videos
	#		(2,2,3,N) where:
	#		tracks[i,:,:,:] --> female if i=0, male if i=1
	#		tracks[:,j,:,:] --> x if i=0, y if i=1
	#		tracks[:,:,k,:] --> head if k=0, thorax if k=1, abdomen if k=2
	#		tracks[:,:,:,n] --> nth frame

	data.close()

	num_frames = X_tracks.shape[-1]

	ix=0; iy=1;
	ihead=0; ibody=1; ileft_wing=2; iright_wing=3

	indices_nan_tracks = np.isnan(np.sum(X_tracks[:,:,:,:], axis=(0,1,2)))

	indices_human_labels = np.isnan(np.sum(scores[:,:], axis=0)) * ~indices_nan_tracks


## check for nans/human labels
if True:
	print('X tracks shape:')
	print(X_tracks.shape)

	print('number of total frames = {:d}'.format(num_frames))

	num_nan_frames = np.sum(indices_nan_tracks)
	print('number of frames with nans: {:d}'.format(np.sum(indices_nan_tracks)))

	if np.sum(indices_nan_tracks[:num_nan_frames]) < num_nan_frames or np.sum(indices_nan_tracks[-num_nan_frames:]) < num_nan_frames:
		print('!!! WARNING: nan frames occur within video, not simply at beginning or end. !!!')

	print('number of human labels: {:d}'.format(np.sum(indices_human_labels)))

	inds_nan_frames = np.flatnonzero(indices_nan_tracks) + 1  # add one b/c frames in video are indexed starting at 1


## check for jumps
if True:
	frames_jumps = []

	dists_female = np.sqrt(np.sum((X_tracks[ifemale,:,ibody,1:] - X_tracks[ifemale,:,ibody,:-1])**2,axis=0))
	dists_male = np.sqrt(np.sum((X_tracks[imale,:,ibody,1:] - X_tracks[imale,:,ibody,:-1])**2,axis=0))

	dists = np.concatenate([dists_female[np.newaxis,:], dists_male[np.newaxis,:]], axis=0)
	dists = np.max(dists,axis=0)

	dists[np.isnan(dists)] = 0.

	inds_jumps = np.argsort(dists)[::-1]



## check for large changes in head dir
if True:
	angs_head_dir_female = np.zeros((num_frames-1,))
	for iframe in range(num_frames-1):
		angs_head_dir_female[iframe] = compute_angle_head_dir(X_tracks[ifemale,:,:,iframe], X_tracks[ifemale,:,:,iframe+1])

	angs_head_dir_male = np.zeros((num_frames-1,))
	for iframe in range(num_frames-1):
		angs_head_dir_male[iframe] = compute_angle_head_dir(X_tracks[imale,:,:,iframe], X_tracks[imale,:,:,iframe+1])

	angs_head_dir = np.concatenate([angs_head_dir_female[np.newaxis,:], angs_head_dir_male[np.newaxis,:]], axis=0)
	angs_head_dir[np.isnan(angs_head_dir)] = 0.
	angs_head_dir = np.max(angs_head_dir, axis=0)

	inds_change_head_dir = np.argsort(angs_head_dir)[::-1]



## check for large backwalking (signs of misoriented head/body)
if True:
	projs_head_dir_vs_vel = np.zeros((2,num_frames-1))

	## female
	for iframe in range(num_frames-1):
		head_dir = compute_head_dir(X_tracks[ifemale,:,:,iframe])
		vel_dir = X_tracks[ifemale,:,ibody,iframe+1] - X_tracks[ifemale,:,ibody,iframe]

		projs_head_dir_vs_vel[ifemale,iframe] = np.sum(head_dir * vel_dir)

	## male
	for iframe in range(num_frames-1):
		head_dir = compute_head_dir(X_tracks[imale,:,:,iframe])
		vel_dir = X_tracks[imale,:,ibody,iframe+1] - X_tracks[imale,:,ibody,iframe]

		projs_head_dir_vs_vel[imale,iframe] = np.sum(head_dir * vel_dir)

	forward_vels = np.max(projs_head_dir_vs_vel, axis=0)

	inds_backwards_walking = np.argsort(forward_vels)


## check for wing extension
if True:
	dists_between_wings = np.zeros((2,num_frames))

	dists_between_wings[0] = np.sqrt(np.sum((X_tracks[ifemale,:,2,:] - X_tracks[ifemale,:,3,:])**2,axis=0))
	dists_between_wings[1] = np.sqrt(np.sum((X_tracks[imale,:,2,:] - X_tracks[imale,:,3,:])**2,axis=0))

	if np.nanmax(dists_between_wings[0]) > np.nanmax(dists_between_wings[1]):
		print('!!! WARNING: Sex of fly may be mixed up...first fly appears to have wing extensions for song? !!!')

	dists_between_wings = np.max(dists_between_wings, axis=0)
	dists_between_wings[np.isnan(dists_between_wings)] = 0.

	inds_large_wing_extensions = np.argsort(dists_between_wings)[::-1]


## save inds in text file
if False:
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


## in single pdf, make helpful plots to track behavior
if True:
	f = plt.figure(figsize=(10,10))

	## male-female distance over time
	if True:
		plt.subplot(3,2,1)

		mf_dists = np.sqrt(np.sum((X_tracks[imale,:,ibody,:] - X_tracks[ifemale,:,ibody,:])**2, axis=0))

		plt.plot(np.arange(mf_dists.size)*(1/60)/60, mf_dists, '-k')

		plt.xlabel('time (min)')
		plt.ylabel('male-female distance (pixels)')

	## male/female velocity magnitude over time
	if True:
		 plt.subplot(3,2,2)

		 vel_mag_male = np.sqrt(np.sum((X_tracks[imale,:,ibody,1:] - X_tracks[imale,:,ibody,:-1])**2, axis=0)) * (1/60)
		 vel_mag_female = np.sqrt(np.sum((X_tracks[ifemale,:,ibody,1:] - X_tracks[ifemale,:,ibody,:-1])**2, axis=0)) * (1/60)
		 times = np.arange(vel_mag_male.size)*(1/60)/60

		 plt.plot(times, vel_mag_male, '-b', label='male', alpha=0.7)
		 plt.plot(times, vel_mag_female, '-r', label='female', alpha=0.7)

		 plt.xlabel('time (min)')
		 plt.ylabel('velocity magnitude (pixels/sec)')
		 plt.legend()

		 plt.ylim([0,0.5])

	## density position male
	if True:
		plt.subplot(3,2,3)

		X_tracks[np.isnan(X_tracks)] = 0.
		plt.hexbin(X_tracks[imale,0,ibody,:], X_tracks[imale,1,ibody,:], bins='log', cmap='Blues', edgecolor='none')

		plt.xlabel('x position')
		plt.ylabel('y position')
		plt.title('density of male body position')
		plt.axis('square')

		plt.xlim([200,900])
		plt.ylim([50,750])

	## density position female
	if True:
		plt.subplot(3,2,4)

		X_tracks[np.isnan(X_tracks)] = 0.
		plt.hexbin(X_tracks[ifemale,0,ibody,:], X_tracks[ifemale,1,ibody,:], bins='log', cmap='Reds', edgecolor='none')
		
		plt.xlabel('x position')
		plt.ylabel('y position')
		plt.title('density of female body position')
		plt.axis('square')

		plt.xlim([200,900])
		plt.ylim([50,750])

	## egocentric male density
	if True:
		plt.subplot(3,2,5)

		relative_positions = get_relative_positions(positions_centric_fly=X_tracks[imale], positions_noncentric_fly=X_tracks[ifemale])

		plt.hexbin(relative_positions[0,1,:], relative_positions[1,1,:], gridsize=50, cmap='Reds', edgecolor='none')

		plt.text(0,0,'^', horizontalalignment='center', verticalalignment='center', color='b')
		plt.text(0,0,'|', horizontalalignment='center', verticalalignment='center', color='b')

		plt.xlabel('x position')
		plt.ylabel('y position')
		plt.title('egocentric male')
		plt.axis('square')

		plt.xlim([-350,350])
		plt.ylim([-350,350])

	## egocentric female density
	if True:
		plt.subplot(3,2,6)

		relative_positions = get_relative_positions(positions_centric_fly=X_tracks[ifemale], positions_noncentric_fly=X_tracks[imale])

		plt.hexbin(relative_positions[0,1,:], relative_positions[1,1,:], gridsize=50, cmap='Blues', edgecolor='none')

		plt.text(0,0,'^', horizontalalignment='center', verticalalignment='center', color='b')
		plt.text(0,0,'|', horizontalalignment='center', verticalalignment='center', color='b')

		plt.xlabel('x position')
		plt.ylabel('y position')
		plt.title('egocentric female')
		plt.axis('square')

		plt.xlim([-350,350])
		plt.ylim([-350,350])

	plt.tight_layout()
	f.savefig('./' + h5_filename[:-3] + '_plots.pdf')
