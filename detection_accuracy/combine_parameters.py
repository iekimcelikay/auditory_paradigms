#! /usr/bin/env python
# Time-stamp: <2026-03-13 m.utrosa@bcbl.eu>

# TODO: check that actually all VALID_COMBOS are valid ... (testing & piloting)
# TODO: check units; timing is in milliseconds but frequency is in Hz
# TODO: check accuracy of trial duration calculation (depends on how sequences are constructed)
# TODO: randomize sampling deviations; sample integers not floats from log scale

"""
Creates a list of tuples with all possible valid combinations of parameters 
for a given auditory oddball experiment. The stimulus is a tone sequence
that can have frequency and/or timing deviants.

Plots the relationship between timing and frequency deviants to eliminate
any confounding factors or biases.

Returns:
- A list of randomzied parameter combinations for an experimental session
- Optionally, saves the combinations as csv
- Optionally, a barplot of timing deviants counts per frequency deviant
- Optionally, a histogram of timing deviant frequency per frequency deviant
"""
# 00. PREPARATION ---------------------------------------------------------------------------------
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt

# 01. DEFINE FUNCTIONS  ---------------------------------------------------------------------------
def create_deviations(num_values, min_val, max_val):
	"""
	Create a sample of deviations spaced evenly on a log scale with base 10.
	Excluding the maximum value.
	"""
	log_values = np.logspace(
		start = np.log10(min_val),
		stop  = np.log10(max_val),
		num   = num_values,
		endpoint=False)

	# Round and select unique values only
	selected_values = np.unique(np.round(log_values))
	
	# Change type to int
	log_values_list = [int(log_value) for log_value in selected_values]
	
	# Get the missing values by sampling from a normal distribution
	missing_values = num_values - len(log_values_list)
	random_sample = np.unique(np.random.choice(range(min_val, max_val), missing_values))
	random_sample_list = [int(rs) for rs in random_sample]
	log_values_list = np.sort(log_values_list + random_sample_list)
	return [int(lvl) for lvl in log_values_list]

##### ALTERNATIVE WAY TO DEFINE DEVIATIONS
# def create_deviations(num_values, min_val, max_val):

# 	# Generate logarithmic values between 1 and 500
# 	log_values = np.logspace(np.log10(min_val), np.log10(max_val), 2000)

# 	# Bias weights: strongly favor higher values, but keep small chance for low ones
# 	x = np.linspace(0, 1, len(log_values))
# 	weights = (x ** 4) + 0.001  # steeper bias (x**4 makes small values rarer)
# 	weights /= weights.sum()

# 	# Randomly sample up to 50 unique values
# 	chosen = np.random.choice(log_values, size=num_values, replace=False, p=weights)

# 	# Round to integers and sort
# 	chosen = np.sort(np.unique(np.rint(chosen).astype(int)))
# 	chosen = [ch for ch in chosen]
# 	return chosen

def calculate_trial_duration(combo, params):
	"""
	Calculates duration in ms for each trial combination.
	"""
	
	# durations = []
	nodev_dur = combo[0] * params["TONE_DURATION"] + (combo[0] - 1) * combo[4]
	# Below part is not needed if we correct for deviation in the second ISI
	# if combo[2] == "early":
	# 	duration =  nodev_dur - combo[1]
	# 	durations.append(duration)
	# elif combo[2] == "late":
	# 	duration = nodev_dur + combo[1]
	# 	durations.append(duration)
	# else:
		# durations.append(nodev_dur)

	return nodev_dur

def create_experimental_sessions(params, sesID, save_csv=True, plot_hist=False, plot_bar=False, MAX_BLOCK_DURATION_MIN=15):
	"""
	Combines specified parameters in valid combinations, then randomly samples trials per run to
	create all trial per one experimental session.

	HARDCODED RULES
	- All values are corrected for exclusiveness (+ 1), so parameters settings are inclusive
	- ISI values are randomly sampled from a given range of integers without replacement
	- Timing & frequency deviations are sampled from a log scale # TODO: no repetitions
	- Include zero as a frequency & timing deviation
	- Frequency and timing deviants cannot co-occur on the same tone
	- Maximum block duration: 15 min

	Raises:
	- ValueError if ITI is smaller than the max ISI and max DEV.
	- ValueError if the step between chosen ISI values is smaller than max DEV.
	- Warning if blocks are too long.
	"""
	
	# 01. SPECIFY THE PARAMETERS ------------------------------------------------------------------
	### a. Calculate all values needed for constructing trials
	ISI = random.sample(
		range(params["ISI_MIN"], params["ISI_MAX"] + 1, params["ISI_STEP"]),
		params["ISI_NO"])
	NO_TONES = list(range(params["MIN_TONES"], params["MAX_TONES"] + 1))

	# Define the absolute size of tone's timing deviation in milliseconds, including zero.
	DEV = create_deviations(params["DEV_NO"], params["DEV_MIN"], params["DEV_MAX"] + 1)
	DEV.insert(0, 0)

	# Type of timing deviation
	DEV_TYPE = ['early', 'on_time', 'late']

	# Set the position of the timing deviation in the sequence.
	# Include a location 0 for cases where there is no deviation in the sequence.
	DEV_LOC = list(range(params["FIRST_DEV_LOC"], params["LAST_DEV_LOC"] + 1))
	DEV_LOC.insert(0, 0)

	# Define the absolute size of tone's frequency deviation in Hz, including zero.
	FREQ = create_deviations(params["FREQ_NO"], params["FREQ_MIN"], params["FREQ_MAX"] + 1)
	FREQ.insert(0, 0)

	# Type of frequency deviation
	FREQ_TYPE = ['higher', 'standard', 'lower']

	# Set the position of the frequency deviation in the sequence.
	FREQ_LOC = list(range(params["FIRST_FREQ_LOC"], params["LAST_FREQ_LOC"] + 1))
	FREQ_LOC.insert(0, 0)

	### b. Check the relationships between parameters
	# ITI is longer than the longest ISI & DEV: for block separability
	ITI = [params["ITI"]] # TODO: randomize/counterbalance the ITI
	if min(ITI) <= max(ISI) or min(ITI) <= max(DEV):
		raise ValueError(
				f'ITI ({min(ITI)} ms) is too short given the '
				f'max ISI ({max(ISI)} ms) / DEV ({max(DEV)} ms).'
		)

	# For separability of a new block in terms of tempo
	# For change in tempo to not be perceived as a deviation
	if len(ISI) > 1:
		if params["ISI_STEP"] < max(DEV):
			raise ValueError(
				f'The step between chosen ISI values ({params["ISI_STEP"]}'
				f' ms) is too small given the max deviation ({max(DEV)} ms).'
				)

	# 03. CREATE THE COMBINATIONS -----------------------------------------------------------------
	# ALL_COMBOS: A list of tuples with all posible combinations of the above parameters.
	ALL_COMBOS = list(product(NO_TONES, DEV, DEV_TYPE, DEV_LOC, ISI, ITI, FREQ, FREQ_TYPE, FREQ_LOC))

	# VALID_COMBOS: removing invalid parameter combinations and adding constraints.
	VALID_COMBOS = []

	for combo in ALL_COMBOS:
	    no_tones  = combo[0]
	    deviation = combo[1]
	    dev_type  = combo[2]
	    dev_loc   = combo[3]
	    isi       = combo[4]
	    iti       = combo[5]
	    freq      = combo[6]
	    freq_type = combo[7]
	    freq_loc  = combo[8]

	    # dev_loc cannot exceed no_tones
	    if dev_loc >= no_tones:
	        continue

	    # freq_loc cannot exceed no_tones
	    if freq_loc >= no_tones:
	        continue

	    # If DEV == 0, then DEV_TYPE must be "on_time" and DEV_LOC must be 0
	    if deviation == 0:
	        if dev_type != "on_time" or dev_loc != 0:
	            continue

	    # If DEV != 0, then DEV_LOC cannot be 0 and type cannot be "on_time"
	    if deviation != 0:
	        if dev_loc == 0 or dev_type == "on_time":
	            continue

	    # If FREQ == 0, then FREQ_TYPE must be "standard" and FREQ_LOC must be 0
	    if freq == 0:
	        if freq_type != "standard" or freq_loc != 0:
	            continue

	    # If FREQ != 0, then FREQ_LOC cannot be 0 and type cannot be "standard"
	    if freq != 0:
	        if freq_loc == 0 or freq_type == "standard":
	            continue

	    # Frequency and timing deviants cannot occur on the same tone
	    if freq_loc == dev_loc:
	    	continue

	    # ISI has to be longer than the deviation
	    if isi < deviation: 
	        continue

	    # Fixed duration must remain positive
	    # duration = no_tones * tone_duration + (no_tones - 1) * ISI +/- dev
	    base_duration = (no_tones * params["TONE_DURATION"]) + ((no_tones - 1) * isi)
	    
	    # Check if the resulting duration is non-positive
	    if (base_duration - deviation) <= 0:
	        continue

	    # If all checks pass, add to list
	    VALID_COMBOS.append(combo)

	# Create a list of all trials for a single experimental session
	# and calculate their durations.
	RUN_COMBOS = []
	BLOCK_DURATIONS = []
	for run in range(params["NO_RUNS"]):
		TRIAL_COMBOS = random.sample(VALID_COMBOS, params["NO_TRIALS"])
		
		block_duration = 0
		for trial_no, trial in enumerate(TRIAL_COMBOS):

			# Add trial number (ID) to the tuples
			trial = trial + tuple((trial_no + 1,))

			# Add run number (ID) to the tuples
			trial = trial + tuple((run + 1,))

			# Add the tuple with added run & trial no
			RUN_COMBOS.append(trial)

			# Calculate trial duration
			trial_duration = calculate_trial_duration(trial, params)
			block_duration += trial_duration

		BLOCK_DURATIONS.append(block_duration)

	# Check average duration of one block
	block_duration_min = np.mean(BLOCK_DURATIONS) / 60000
	run_duration_min = sum(BLOCK_DURATIONS) / 60000
	if block_duration_min > MAX_BLOCK_DURATION_MIN:
		warning_msg = (
		f"WARNING: The average block duration ({block_duration_min:.2f} min) "
		f"exceeds the recommended maximum of {MAX_BLOCK_DURATION_MIN} min. "
		f"Ensure balance between challenge & exhaustion."
	    )
		warnings.warn(warning_msg)
	else:
		print(f"Block duration: {block_duration_min:.2f} min")
		print(f"Experiment duration: {run_duration_min:.2f} min")

	# 04. SAVE THE COMBINATIONS -------------------------------------------------------------------
	# Create the dataframe
	df = pd.DataFrame(
		RUN_COMBOS,
		columns=[
		"NO_TONES",
		"DEV",
		"DEV_TYPE",
		"DEV_LOC",
		"ISI",
		"ITI",
		"FREQ",
		"FREQ_TYPE",
		"FREQ_LOC",
		"TRIAL_NO",
		"RUN_NO"]
		)

	# Save the dataframe as csv
	out_path = Path(params["OUT_PATH"])
	out_path.mkdir(exist_ok=True, parents=True)
	filename =  f"exp_parameter_combo_ses-{sesID:003d}.csv"
	out_dir  = out_path / filename
	if save_csv:
		df.to_csv(out_dir, sep=",", index=False)
		print(f"\nSaved {filename} to {out_path}.")

	# 05. CALCULATE THE DISTRIBUTION --------------------------------------------------------------
	# Get the number of timing devs per category of frequency deviation
	groups = df.groupby(by=["FREQ", "DEV"])["DEV"].count()

	# Add zero occurrence if deviation doesn't exist with a frequency
	complete_groups = groups.unstack(fill_value=0).stack()
	plot_groups = complete_groups.reset_index(name='COUNT')

	# 06. PLOT THE DISTRIBUTION -------------------------------------------------------------------
	### a. Plotting a histogram from original data
	if plot_hist:
		sns.set_theme(style="white", palette="colorblind", font="sans-serif")
		h = sns.FacetGrid(
			data = df,
			col = "FREQ",
			col_wrap=3, # incompatible with row
			height=2,
			aspect=1,
			sharey = True,
			sharex = True,
			hue="FREQ",
			palette = "colorblind"
			)

		h.map_dataframe(
			sns.histplot,
			x="DEV",
			stat="frequency",
			bins=50,
			kde=False
			)

		h.set_axis_labels("", "")
		h.set_titles("{col_name} Hz", size=11)

		for ax in h.axes.flat:
			ax.tick_params(
				axis="x",
				which="both",
				bottom=True,
				top=False,
				labelbottom=True,
				labelsize=9
				)
			ax.tick_params(
				axis="y",
				which="both",
				left=True,
				right=False,
				labelleft=True,
				labelsize=9
				)
			ax.tick_params(labelbottom=True, labelleft=True)

		h.fig.supxlabel("Timing Deviation [msec]", fontsize=12)
		h.fig.supylabel("Occurrences [frequency]", fontsize=12)

		h.fig.suptitle(
			f"Distribution of Timing Deviants | Frequency Deviant for SES-{sesID:003d}",
			fontsize=12,
			weight="bold"
			)
		sns.despine()
		plt.show()

	### b. Plotting a barplot from subset data
	if plot_bar:
		sns.set_theme(style="white", palette="colorblind", font="sans-serif")
		g = sns.FacetGrid(
			plot_groups,
			col="FREQ",
			col_wrap=2,
			height=2,
			aspect=1.7,
			sharey=True,
			sharex=True,
			hue="FREQ",
			palette="colorblind"
		)

		g.map_dataframe(
			sns.barplot,
			x="DEV",
			y="COUNT",
			order=sorted(plot_groups["DEV"].unique()),
			dodge=False
		)

		g.set_axis_labels("", "")
		g.set_titles("{col_name} Hz", size=11)

		for ax in g.axes.flat:
			ax.tick_params(
				axis="x",
				which="both",
				bottom=True,
				top=False,
				labelbottom=True,
				rotation=45,
				labelsize=5
				)
			ax.tick_params(
				axis="y",
				which="both",
				left=True,
				right=False,
				labelleft=True,
				labelsize=8
				)
			ax.tick_params(labelbottom=True, labelleft=True)

		g.fig.supxlabel("Timing Deviation [msec]", fontsize=12)
		g.fig.supylabel("Occurrences [count]", fontsize=12)

		g.fig.suptitle(
			f"Occurrences of Timing Deviants | Frequency Deviant for SES-{sesID:003d}",
			fontsize=12,
			weight="bold"
			)

		sns.despine(trim=True)
		plt.show()

# 03. EXAMPLE USAGE  ------------------------------------------------------------------------------
if __name__ == "__main__":
	params = {
	
	"OUT_PATH" : "/home/mutrosa/Documents/projects/auditory_paradigms/detection_accuracy/out",

	# a. Overall structure of the experimental session
	"NO_RUNS"   : 4, 	   # Number of blocks, equal to no. of func scans in MRI protocol
	"NO_TRIALS" : 132,	   # Stimuli repetitions, where a trial is one tone sequence (stimulus)
	"ITI"       : 1500,    # Inter trial interval is the time between two sequences (trials)
	
	# b. Tone sequence
	"MIN_TONES"     : 7,    # Min. no. of tones in a single sequence
	"MAX_TONES"     : 7,    # Max. no. of tones in a single sequence
	"TONE_DURATION" : 50,   # Duration of a single tone in msec

	# c. Inter Stimulus Interval is the time between presentation of two sequential tones.
	"ISI_MIN"  : 700,        # Min. duration of ISI 
	"ISI_MAX"  : 700,        # Max. duration of ISI
	"ISI_STEP" : 300,    	 # Step size of the ISI range (population)
	"ISI_NO"   : 1,          # How many ISI values to test?
	
	# d. Timing deviants
	"DEV_MIN": 1,          # Min. tone timing deviation  
	"DEV_MAX": 300,        # Max. tone timing deviation
	"DEV_NO" : 66,         # How many deviations to test?

	"FIRST_DEV_LOC" : 4,  # The first tone that can be displaced:
						  # e.g.: if you want the first few tones to be timing standards

	"LAST_DEV_LOC"  : 6,  # The last tone to be displaced timing-wise

	# c. Frequency deviants
	"FREQ_MIN"  : 1200,     # Min. frequency deviation
	"FREQ_MAX"  : 1400,     # Max. frequency deviation
	"FREQ_NO"   : 5,		# How many frequencies to test?

	"FIRST_FREQ_LOC" : 4,  # The first tone that can be displaced frequency-wise
						   # e.g.: if you want the first few tones to be frequency standards

	"LAST_FREQ_LOC"  : 6,  # The last tone to be displaced frequency-wise
	}

	for session in range(1000):
		session = session + 1
		create_experimental_sessions(params, session, save_csv=True)