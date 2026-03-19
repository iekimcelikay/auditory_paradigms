#! /usr/bin/env python
# Time-stamp: <2026-03-13 m.utrosa@bcbl.eu>

# TODO: check that actually all VALID_COMBOS are valid ... (testing & piloting)
# TODO: check accuracy of trial duration calculation (depends on how sequences are constructed)
# TODO: randomize sampling deviations; sample integers not floats from log scale
# TODO: use durations to ensure approximately the same duration of runs (Important for MRI protocol!)
# COMBO_DURATIONS = calculate_trial_duration(VALID_COMBOS, params)
# paired_trials = list(zip(VALID_COMBOS, COMBO_DURATIONS))
# random.shuffle(paired_trials)

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

def calculate_trial_duration(combo, params):
	"""
	Calculates theoretical duration in ms for each trial combination.
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
	Creates trials for one experimental session.
	Ensures counterbalancing of independent variables and randomization of control variables.

	HARDCODED RULES
	- All values are corrected for exclusiveness (+ 1), so parameters settings are inclusive
	- ISI values are randomly sampled from a given range of integers without replacement
	- Each ITI value is different and randomly sampled with replacement (ensuring enough values)
	- Timing & frequency deviations are sampled from a log scale # TODO: no repetitions
	- Zero is included as in timing deviation values
	- Frequency and timing deviants cannot co-occur on the same tone
	- Maximum block duration: 15 min

	Raises:
	- ValueError if ITI is smaller than the max ISI and max DEV.
	- ValueError if the step between chosen ISI values is smaller than max DEV.
	- Warning if blocks are too long.
	"""
	
	# 01. GENERATE INDEPENDENT VARIABLE: Timing deviation -----------------------------------------
	# Define the absolute size of tone's timing deviation in milliseconds, including zero.
	DEV = create_deviations(params["DEV_NO"], params["DEV_MIN"], params["DEV_MAX"] + 1)
	DEV.insert(0, 0)

	# Type of timing deviation
	DEV_TYPE = ['early', 'on_time', 'late']

	# Set the position of the timing deviation in the sequence.
	# Include a location 0 for cases where there is no deviation in the sequence.
	DEV_LOC = list(range(params["FIRST_DEV_LOC"], params["LAST_DEV_LOC"] + 1))
	DEV_LOC.insert(0, 0)

	# 02. GENERATE COUNTERBALANCED TRIALS ---------------------------------------------------------
	# Generate a list of dictionaries with all possible combinations of the independent variables.
	# Timing deviants are counterbalanced for their type (direction) and location.
	TARGET_COMBOS = [{"dev" : d, "dev_type" : dt, "dev_loc" : dl} for d, dt, dl in product(DEV, DEV_TYPE, DEV_LOC)]

	# Remove invalid trial combinations 
	VALID_TARGET_COMBOS = []
	for combo in TARGET_COMBOS:

		# If DEV == 0, then DEV_TYPE must be "on_time" and DEV_LOC must be 0
		if combo["dev"] == 0:
			if combo["dev_type"] != "on_time" or combo["dev_loc"] != 0:
				continue

		# If DEV != 0, then DEV_LOC cannot be 0 and type cannot be "on_time"
		if combo["dev"] != 0:
			if combo["dev_type"] == "on_time" or combo["dev_loc"] == 0:
				continue

		# If all checks pass, add dict to list
		VALID_TARGET_COMBOS.append(combo)

	# Repeat the counterbalanced trials params["DEV_REP"]-times.
	# Multiple repetitions of dependent variable increase power.
	VALID_TARGET_COMBOS_REPS = VALID_TARGET_COMBOS * params["DEV_REP"]

	# Print update on the number of trials created.
	NO_TRIALS = len(VALID_TARGET_COMBOS_REPS)
	print(
		f"\nThere is a total of {NO_TRIALS} signal trials."
		f" These trials contain {params['DEV_REP']} repetitions of each timing deviation"
		f" for each possible position of the deviation {DEV_LOC[1:]}."
		f" There are {len(DEV)} unique timing deviation values (including zero)."
		f"\n\nAn MRI experiment would have {int((NO_TRIALS * (1 / 3))/(2/3) + NO_TRIALS)} trials in total."
		f" One third (n = {int((NO_TRIALS * (1 / 3))/(2/3))}) is silent (no signal trials) and"
		f" two thirds (n = {len(VALID_TARGET_COMBOS_REPS)}) are sound (signal trials)."
		)

	# Shuffle the trials to prevent order effects and predictability.
	random.shuffle(VALID_TARGET_COMBOS_REPS)

	# 03. ADD CONTROL VARIABLES -------------------------------------------------------------------
	# We're not interested in the effect of these variables on encoding of timing deviancy, so we
	# either fix (keep constant) or randomly define the. 
	# This keeps our dependent variable affected systematically or unaffected.
	
	# Each ITI is different to prevent locking into any periodic signals.
	# The total ITI sample is the (total number of trials) - (number of runs).
	ITI = random.choices(
		population=range(params["ITI_MIN"], params["ITI_MAX"] + 1),
		k=(NO_TRIALS - params["NO_RUNS"])
		)

	### -------- Randomly fixed parameters -------- 
	# Depending on how input parameters are set, these can:
	# 		- be fixed for each experimental session, or
	#		- can vary across sessions. 
	# In both cases, these parameters are fixed for all trials in one exp. session.
	ISI = random.sample(
			range(params["ISI_MIN"], params["ISI_MAX"] + 1, params["ISI_STEP"]),
			1
			)
	NO_TONES = random.sample(
		range(params["MIN_TONES"], params["MAX_TONES"] + 1),
		1
		)
	BASE_FREQUENCY = random.sample(
		range(params["FREQ_MIN"], params["FREQ_MAX"] + 1, params["FREQ_STEP"]),
		1
		)

	### -------- Randomly variable parameters -------- 
	# Define the pool of tone's possible frequency deviations in Hz
	# TODO: replace with naturalistic range (musical notes or soundscape freq.)
	FREQ = create_deviations(300, params["FREQ_MIN"], params["FREQ_MAX"] + 1)
	
	# Remove base frequency as a possible frequency deviation
	if BASE_FREQUENCY[0] in FREQ:
		FREQ.remove(BASE_FREQUENCY[0])

	# Probability of frequency deviation: how easy is counting frequencies?
	# Different number of deviants on each exp. session
	# A task is hard when it's close to uncertainty
	# TODO: Participants can't report a number because there's only 4 buttons in the MRI
	COUNT_MIN  = random.sample((40, 60), 1)
	FREQ_COUNT = random.sample(list(range(COUNT_MIN[0], NO_TRIALS + 1)), 1)
	freq = random.sample(FREQ, FREQ_COUNT[0])
	zeros = [0] * (NO_TRIALS - FREQ_COUNT[0])
	FREQUENCY_DEVIANTS = freq + list(zeros)
	random.shuffle(FREQUENCY_DEVIANTS)

	# ADD: NO_TONES, ISI, BASE_FREQ, FREQ_DEV, FREQ_LOC
	COMBOS_ALL_DEV = []
	for count, trial in enumerate(VALID_TARGET_COMBOS_REPS):

		# Add randomly fixed parameters to the dictionary
		trial["no_tones"] = NO_TONES[0]
		trial["isi"] = ISI[0]
		trial["base_freq"] = BASE_FREQUENCY[0]
		
		# Randomly choose varying parameters to the dictionary
		freq_dev = FREQUENCY_DEVIANTS[count]
		if freq_dev != 0:
			freq_loc = random.choices(
				list(range(0, trial["dev_loc"])) + list(range(trial["dev_loc"] + 1, NO_TONES[0] + 1))
				)[0]
		else:
			freq_loc = 0

		# Add varying parameters to the dictionary
		trial["freq_dev"] = freq_dev
		trial["freq_loc"] = freq_loc
		
		# Append the final structure
		COMBOS_ALL_DEV.append(trial)

	# 04. SPLIT INTO BLOCKS ----------------------------------------------------------------------
	# It's possible that the number of trials is not divisible by desired number of blocks.
	block_base = NO_TRIALS // params["NO_RUNS"]

	# Get number of blocks that need one extra item
	remainder = NO_TRIALS % params["NO_RUNS"]
	blocks = [block_base + 1] * remainder + [block_base] * (params["NO_RUNS"] - remainder)
	
	# Create a list of all trials for a single experimental session and calculate their durations.
	BLOCK_COMBOS = []
	for block_no, block in enumerate(blocks):
		for trial_no in range(block):

			cad = COMBOS_ALL_DEV.pop(0).copy()

			# Add trial and block ID
			cad["trial_no"] = trial_no + 1
			cad["block_no"] = block_no + 1

			# Add ITI
			if trial_no != block - 1:
				cad["ITI"] = ITI.pop(0)
			
			BLOCK_COMBOS.append(cad)

	# 05. ADD CONTROL CHECKS ----------------------------------------------------------------------
	# Check compatibility of the relationships between the parameters
	# To ensure trial separability, ITI must be longer than 2 x (max(ISI) + tone duration)
	if min(ITI) <= 2 * (max(ISI) + params["TONE_DURATION"]):
		raise ValueError(
				f'Min ITI ({min(ITI)} ms) is too short given the '
				f'max ISI ({max(ISI)} ms) & tone duration ({params["TONE_DURATION"]} ms).'
		)

	# To ensure trial separability, ITI must be longer than the longest ISI & DEV
	if min(ITI) <= max(ISI) or min(ITI) <= max(DEV):
		raise ValueError(
				f'Min ITI ({min(ITI)} ms) is too short given the '
				f'max ISI ({max(ISI)} ms) / max DEV ({max(DEV)} ms).'
		)

	# ISI should not be not be perceived as a deviation
	# To ensure separability of a new block / block in terms of tempo
	if len(ISI) > 1:
		if params["ISI_STEP"] < max(DEV):
			raise ValueError(
				f'The step between ISI values ({params["ISI_STEP"]} ms) '
				f'is too small given the max deviation ({max(DEV)} ms).'
				)

	    # # freq_loc cannot exceed no_tones
	    # if freq_loc >= no_tones:
	    #     continue

	    # # If FREQ == 0, then FREQ_LOC must be 0
	    # if freq == 0:
	    #     if freq_loc != 0:
	    #         continue

	    # # If FREQ != 0, then FREQ_LOC cannot be 0 and type cannot be "standard"
	    # if freq != 0:
	    #     if freq_loc == 0:
	    #         continue

	    # # Frequency and timing deviants cannot occur on the same tone
	    # if freq_loc == dev_loc:
	    # 	continue

	    # # ISI has to be longer than the deviation
	    # if isi < deviation: 
	    #     continue

	    # # Fixed duration must remain positive
	    # # duration = no_tones * tone_duration + (no_tones - 1) * ISI +/- dev
	    # base_duration = (no_tones * params["TONE_DURATION"]) + ((no_tones - 1) * isi)
	    
	    # # Check if the resulting duration is non-positive
	    # if (base_duration - deviation) <= 0:
	    #     continue
	
	# 06. ADD CONTROL CHECKS ----------------------------------------------------------------------

	# Check average duration of one block
	# block_duration_min = np.mean(BLOCK_DURATIONS) / 60000
	# run_duration_min = sum(BLOCK_DURATIONS) / 60000
	# if block_duration_min > MAX_BLOCK_DURATION_MIN:
	# 	warning_msg = (
	# 	f"WARNING: The average block duration ({block_duration_min:.2f} min) "
	# 	f"exceeds the recommended maximum of {MAX_BLOCK_DURATION_MIN} min. "
	# 	f"Ensure balance between challenge & exhaustion."
	#     )
	# 	warnings.warn(warning_msg)
	# else:
	# 	print(f"Block duration: {block_duration_min:.2f} min")
	# 	print(f"Experiment duration: {run_duration_min:.2f} min")

	# 07. SAVE THE COMBINATIONS -------------------------------------------------------------------
	# Create the dataframe
	df = pd.DataFrame(BLOCK_COMBOS)

	# Save the dataframe as csv
	out_path = Path(params["OUT_PATH"])
	out_path.mkdir(exist_ok=True, parents=True)
	filename =  f"exp_parameter_combo_ses-{sesID:003d}.csv"
	out_dir  = out_path / filename
	if save_csv:
		df.to_csv(out_dir, sep=",", index=False)
		print(f"\nSaved {filename} to {out_path}.")

	# 06. CALCULATE THE DISTRIBUTION --------------------------------------------------------------
	# Get timing devs per category of frequency deviation
	groups = df.groupby(by=["FREQ", "DEV"])["DEV"].count()

	# Add zero occurrence if deviation doesn't exist with a frequency
	complete_groups = groups.unstack(fill_value=0).stack()
	plot_groups = complete_groups.reset_index(name='COUNT')

	# 07. PLOT THE DISTRIBUTION -------------------------------------------------------------------
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
	
	"OUT_PATH" : "/home/mutrosa/Documents/projects/auditory_paradigms/detection_accuracy/plots",

	# a. Overall structure of the experimental session
	"NO_RUNS"   : 4, 	   # Number of blocks, equal to no. of func scans in MRI protocol
	"ITI_MIN"   : 1550,    # Must be noticeably larger than max(ISI) + tone_duration
	"ITI_MAX"   : 1950,    # Smaller than inter-block-interval (max rest time = 2 min)
	
	# b. Tone sequence
	"MIN_TONES"     : 7,    # Min. no. of tones in a single sequence
	"MAX_TONES"     : 7,    # Max. no. of tones in a single sequence
	"TONE_DURATION" : 50,   # Duration of a single tone in msec

	# c. Inter Stimulus Interval is the time between presentation of two sequential tones.
	"ISI_MIN"  : 700,        # Min. duration of ISI 
	"ISI_MAX"  : 700,        # Max. duration of ISI
	"ISI_STEP" : 300,    	 # Step size of the ISI range (population)
	
	# d. Timing deviants
	"DEV_MIN" : 1,          # Min. tone timing deviation  
	"DEV_MAX" : 300,        # Max. tone timing deviation
	"DEV_NO"  : 22,        	# How many deviations to test (excluding 0, + and - values)?
	"DEV_REP" : 2,			# How many times should each deviation repeat across the session?

	"FIRST_DEV_LOC" : 4,  # The first tone that can be displaced:
						  # e.g.: if you want the first few tones to be timing standards

	"LAST_DEV_LOC"  : 6,  # The last tone to be displaced timing-wise

	# c. Frequency deviants
	"FREQ_MIN"  : 200,     # Min. frequency deviation
	"FREQ_MAX"  : 750,     # Max. frequency deviation
	"FREQ_STEP" : 124,	   # The difference between frequencies TODO: melscaled
	"FIRST_FREQ_LOC" : 1,  # The first tone that can be displaced frequency-wise
						   # e.g.: if you want the first few tones to be frequency standards

	"LAST_FREQ_LOC"  : 7,  # The last tone to be displaced frequency-wise
	}

	for session in range(5):
		session = session + 1
		create_experimental_sessions(params, session, save_csv=True, plot_hist=True, plot_bar=True)