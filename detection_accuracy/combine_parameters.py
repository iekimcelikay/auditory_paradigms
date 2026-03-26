#! /usr/bin/env python
# Time-stamp: <2026-03-20 m.utrosa@bcbl.eu>
# TODO: replace frequency dev with naturalistic range (musical notes or soundscape freq.)

"""
Creates a list of dictionaries with all possible valid parameter combinations 
for a given auditory oddball experiment. The stimuli are tone sequences with
frequency and timing deviants.

Plots the relationship between independent and control variables to for visual
inspection of any confounding or biases.

Returns:
- A list of randomized parameter combinations for an experimental session
- Optionally, saves the list as csv
- Optionally, a shows and/or saves the plots
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
def create_deviations(num_values, min_val, max_val, zero=True, N=100):
	"""
	Generate `num_values` unique integers by sampling from a pool of log-spaced values (base 10).
	Sampling is random and without replacement.

	Parameters
	----------
	num_values (int): The total number of deviations desired.
	min_val (int): The smallest deviation in absolute terms. Must be larger than 0.
	max_val (int): The largest deviation in absolute terms.
	zero (bool): If True, includes 0 in the resulting deviation sample.
	N (int): The multiplier for the log-spaced pool size. Higher values are better.
	
	Returns
	-------
	selected_values (list of int): A sorted list of unique integer from a log space.
	
	Raises
	------
	ValueError:
		- if `min_val` <= 0
		- if `max_val` <= `min_val`, or
		- if the log pool cannot provide `num_values` unique integers.

	Notes
	-----
	Why logarithmic? 
	Human perception of magnitude is non-linear and logarithmic (Weber & Fechner).

	Relevant: 10.1126/science.1156540.
	- Innate intuition of number mapping is on a logarithmic scale.
	- The concept of linear number line is a cultural invention.
	- Linear mapping is observed only for small/symbolic numbers in educated participants.
	"""
	# Validate input arguments
	if min_val <= 0:
		raise ValueError("min_val must be greater than 0 for log10 calculation.")
	if max_val <= min_val:
		raise ValueError("max_val must be greater than min_val.")

	# Generate a pool of logarithmically spaced numbers
	log_pool = np.logspace(
		start = np.log10(min_val),
		stop  = np.log10(max_val),
		num   = num_values * N,
		endpoint=False
		)

	# Round, convert to int, and select unique values only
	int_values = np.unique(np.round(log_pool).astype(int))
	
	# Raise error if not possible to meet input arguments
	if len(int_values) < num_values:
		raise ValueError(
		f"Cannot generate {num_values} unique values. "
		f"Log-spaced pool size is {len(int_values)}. "
		"Try increasing the N multiplier or decreasing num_values."
		)

	# Randomly sample unique values
	if zero:
		# Sample one less than `num_values` because we're adding zero
		selected_values = np.random.choice(list(int_values), size=num_values-1, replace=False)
		# Add zero
		result = np.concatenate([[0], selected_values])

	else:
		result = np.random.choice(list(int_values), size=num_values, replace=False)

	return np.sort(result).tolist()

def calculate_trial_duration(combo, params):
	"""
	Calculates a theoretical trial duration in milliseconds.
	
	combo: a dictionary with information about the trial's ITI and ISI
	params: a dictionary with fixed input parameters (number of tones and their duration)
	"""
	tone_duration = combo["no_tones"] * params["TONE_DURATION"]
	isi_duration  = (combo["no_tones"] - 1) * combo["isi"]
	iti_duration  = combo["iti"]
	trial_duration = tone_duration + isi_duration + iti_duration
	
	return trial_duration

def create_experimental_sessions(params, sesID, save_csv=False, MAX_BLOCK_DURATION_MIN=15):
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
	- ValueError if trial durations are negative.
	- Warning if blocks are too long.
	"""
	
	# 01. GENERATE INDEPENDENT VARIABLES: Timing deviation size and location ----------------------
	# Create negative values of tone's timing deviation, including a "negative" zero.
	DEV_pos = np.array(params["DEVS"])
	DEV_neg = -DEV_pos

	# Combine positive and negative values into a sorted list.
	DEV_arr = np.concatenate([DEV_pos, DEV_neg])
	DEV     = np.sort(DEV_arr).tolist()

	# Determine the "type" of timing deviations (direction)
	if 0 in DEV:
		DEV_TYPE = ['on_time', 'early', 'late']
	else:
		DEV_TYPE = ['early', 'late']

	# Set the positions of the timing deviation in the tone sequences.
	DEV_LOC = list(range(params["FIRST_DEV_LOC"], params["LAST_DEV_LOC"] + 1))

	# 02. GENERATE COUNTERBALANCED TRIALS ---------------------------------------------------------
	# Generate a list of dictionaries with all possible combinations of the independent variables.
	# This will counterbalance the variables for their type (direction) and location.
	TARGET_COMBOS = [{"dev" : d, "dev_type" : dt, "dev_loc" : dl} for d, dt, dl in product(DEV, DEV_TYPE, DEV_LOC)]

	# Remove invalid trial combinations
	VALID_TARGET_COMBOS = []
	for combo in TARGET_COMBOS:

		# If DEV == 0, then DEV_TYPE must be "on_time"
		if combo["dev"] == 0:
			if combo["dev_type"] != "on_time":
				continue

		# If DEV != 0, then DEV_TYPE cannot be "on_time"
		if combo["dev"] != 0:
			if combo["dev_type"] == "on_time":
				continue

		# If DEV > 0, then type cannot be "early"
		if combo["dev"] > 0:
			if combo["dev_type"] == "early":
				continue

		# If DEV < 0, then type cannot be "late"
		if combo["dev"] < 0:
			if combo["dev_type"] == "late":
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
		f" for each possible position of the deviation {DEV_LOC}."
		f" There are {len(DEV)} unique timing deviation values (including zero, if applicable)."
		f"\n\nAn MRI experiment would have {int((NO_TRIALS * (1 / 3))/(2/3) + NO_TRIALS)} trials in total. "
		f"One third (n = {int((NO_TRIALS * (1 / 3))/(2/3))}) silent (no signal) trials and "
		f"two thirds (n = {len(VALID_TARGET_COMBOS_REPS)}) sound (signal) trials."
		)

	# Shuffle the trials to prevent order effects and predictability.
	random.shuffle(VALID_TARGET_COMBOS_REPS)

	# 03. ADD CONTROL VARIABLES -------------------------------------------------------------------
	# We're not interested in the effect of these variables on encoding of timing deviancy, so we
	# either fix (keep constant) or randomly define them. This keeps our dependent variable either
	# affected systematically or unaffected.
	
	### -------- Parameters that are constant across trials -------- 
	# Depending on how input parameters are set, these can:
	# 		- be fixed for each experimental session, or
	#		- can vary randomly across sessions. 
	# In both cases, these parameters are fixed for all trials in one exp. session.
	ISI = random.sample(
		range(params["ISI_MIN"], params["ISI_MAX"] + 1),
		1
		)
	NO_TONES = random.sample(
		range(params["MIN_TONES"], params["MAX_TONES"] + 1),
		1
		)

	### -------- Parameters that vary across trials -------- 
	ITI = random.choices(
		population=range(params["ITI_MIN"], params["ITI_MAX"] + 1),
		k=(NO_TRIALS - params["NO_BLOCKS"])
		)

	# To ensure trial separability, ITI must be longer than 2 x (max(ISI) + tone duration)
	if min(ITI) <= 2 * (max(ISI) + params["TONE_DURATION"]):
		raise ValueError(
				f'Min ITI ({min(ITI)} ms) could be too short given the '
				f'max ISI ({max(ISI)} ms) & tone duration ({params["TONE_DURATION"]} ms).'
		)

	# To ensure trial separability, ITI must be longer than the longest DEV
	if min(ITI) <= max(DEV):
		raise ValueError(
				f'Min ITI ({min(ITI)} ms) could be too short given the '
				f'max DEV ({max(DEV)} ms).'
		)

	### -------- Add control variables --------
	# Loop through each of the counterbalanced trials
	# Add NO_TONES, ISI, BASE_FREQ, FREQS, FREQ_LOC
	COMBOS_ALL_DEV = []
	for count, trial in enumerate(VALID_TARGET_COMBOS_REPS):

		# Add randomly fixed parameters to the trial dict
		trial["no_tones"] = NO_TONES[0]
		trial["isi"] = ISI[0]
		
		# Copy all possible frequency values
		FREQ = params["FREQS"].copy()

		# Randomly select the frequency standard
		BASE_FREQUENCY = random.sample(FREQ, 1)
		trial["base_freq"] = BASE_FREQUENCY[0]

		# Remove the standard as a possible frequency deviation
		FREQ.remove(BASE_FREQUENCY[0])

		# Generate a list of all possible FREQ_DEV locations in the tone sequence
		FREQ_LOC_ALL = list(range(params["FIRST_FREQ_LOC"], params["LAST_FREQ_LOC"] + 1))
		
		# Ensure that the dev_loc and freq_loc are not the same.
		# Freq_dev should never occur on the same tone as the tim_dev
		FREQ_LOC_ALL.remove(trial["dev_loc"])

		# Randomly determine the number of frequency deviants for the current trial
		FREQ_REP = random.sample(
			list(range(params["FREQ_REP_MAX"] + 1)),
			1
			)
		
		# Allow random sampling with replacement for deviants
		# This means that freq deviants are not fixed
		FREQ_DEVS = tuple(random.choices(FREQ, k=FREQ_REP[0]))

		# Randomly choose the location of the FREQ_DEVS
		# Without replacement
		FREQ_LOC = tuple(random.sample(FREQ_LOC_ALL, FREQ_REP[0]))

		# Add tuples to the trial dict. Tuples are immutable ;)
		# If no FREQ_DEVS occur in the current trial, the tuples are empty.
		trial["freq_dev"] = FREQ_DEVS
		trial["freq_loc"] = FREQ_LOC
		
		# Append the final structure
		COMBOS_ALL_DEV.append(trial)

	# 04. SPLIT INTO BLOCKS ----------------------------------------------------------------------
	# It's possible that the number of trials is not divisible by desired number of blocks.
	block_base = NO_TRIALS // params["NO_BLOCKS"]

	# Get number of blocks that need one extra item
	remainder = NO_TRIALS % params["NO_BLOCKS"]
	blocks = [block_base + 1] * remainder + [block_base] * (params["NO_BLOCKS"] - remainder)
	# TODO: write update about the number of trials per block and a warning if the number of trials
	# is not the same across blocks!!

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
				cad["iti"] = ITI.pop(0)
			else:
				cad["iti"] = 0
			
			BLOCK_COMBOS.append(cad)

	# 05. CALCULATE DURATIONS ---------------------------------------------------------------------
	# Get duration of trials
	trial_durs = []
	for b in BLOCK_COMBOS:
		t_dur = calculate_trial_duration(b, params)
		trial_durs.append(t_dur)
	
	# Get duration of blocks
	block_durations = {1: 0, 2: 0, 3: 0, 4: 0}
	for b, duration in zip(BLOCK_COMBOS, trial_durs):
		b_no = b["block_no"]
		block_durations[b_no] += duration

	# Get average duration of trials & blocks
	block_dur_avg = np.mean(list(block_durations.values()))
	trial_dur_avg = np.mean(trial_durs)

	# Convert to min
	block_dur_min = block_dur_avg / 60000
	exp_dur_min   = sum(list(block_durations.values())) / 60000

	# Raise warning if average block duration is too long
	if block_dur_min > MAX_BLOCK_DURATION_MIN:
		warning_msg = (
		f"WARNING: The average block duration ({block_dur_min:.2f} min) "
		f"exceeds the recommended maximum of {MAX_BLOCK_DURATION_MIN} min. "
		f"Ensure balance between challenge & exhaustion."
		)
		warnings.warn(warning_msg)
	else:
		print(
		f"\nExperiment duration: {exp_dur_min:.2f} min."
		f"\nAverage block duration: {block_dur_min:.2f} min."
		f"\nAverage trial duration: {int(trial_dur_avg)} msec."
		)

	# Ensure that duration of all trials is positive
	invalid_trials = [{"trial" : i, "duration": d} for i, d in enumerate(trial_durs) if d<= 0]
	if invalid_trials:
		raise ValueError(
			"Verify input parametes. Invalid trial durations found."
			f"\n{invalid_trials}."
			)
	
	# 06. SAVE TRIALS -----------------------------------------------------------------------------
	# Create a dataframe from the list of dictionaries
	df = pd.DataFrame(BLOCK_COMBOS)

	# Initialize a column for the difference between the standard and deviant frequency
	df['freq_diff'] = None
	for row, series in df.iterrows():
		std_freq = series.base_freq 
		f_diff   = tuple()
		for i in series.freq_dev:
			diff = np.abs(std_freq - np.abs(i))
			f_diff = f_diff + tuple([diff])
		df.at[row, 'freq_diff'] = f_diff

	# Save the dataframe as csv
	filename =  f"exp_parameter_combo_ses-{sesID:003d}.csv"
	out_path = Path(params["OUT_PATH"])
	out_path.mkdir(exist_ok=True, parents=True)
	out_dir  = out_path / filename
	if save_csv:
		df.to_csv(out_dir, sep=",", index=False)
		print(f"\nSaved {filename} to {out_path}.")

# 02. EXAMPLE USAGE  & SIMULATION OF EXPERIMENTAL SESSIONS ----------------------------------------
if __name__ == "__main__":
	params = {
	
	"OUT_PATH" : "/home/mutrosa/Documents/projects/auditory_paradigms/detection_accuracy/trials",

	# a. Overall structure of the experimental session
	"NO_BLOCKS" : 4, 	   # Number of blocks, equal to the number of funcional scans in the MRI protocol
	"ITI_MIN"   : 1550,    # Must be noticeably larger than max(ISI) + tone_duration
	"ITI_MAX"   : 1950,    # Smaller than inter-block-interval (max rest time = 2 min)
	
	# b. Tone sequence
	"TONE_DURATION" : 50,   # Duration of a single tone in msec

	# If kept constant, the length of tone sequences is the same for each experimental session.
	# If not, length of tone sequence is chosen randomly from the range and kept constant across sesions.
	"MIN_TONES"     : 7,    # Min. no. of tones in a single sequence
	"MAX_TONES"     : 7,    # Max. no. of tones in a single sequence

	# c. Inter Stimulus Interval is the time between presentation of two sequential tones.
	"ISI_MIN"  : 700,        # Min. duration of ISI 
	"ISI_MAX"  : 700,        # Max. duration of ISI
	"ISI_STEP" : 300,    	 # Step size of the ISI range (population) TODO: this may be obsolete for MRI (not for beh)
	
	# d. Timing deviants
	"DEVS"    : [0, 4, 8, 13, 19, 27, 36, 48, 63, 80, 100, 125], # Absolute timing deviants in msec
	"DEV_REP" : 4,	   # How many times should each deviation repeat across the exp. session?

	"FIRST_DEV_LOC" : 4,  # The first tone that can be displaced:
						  # e.g.: if you want the first few tones to be timing standards
	"LAST_DEV_LOC"  : 6,  # The last tone to be displaced timing-wise

	# c. Frequency deviants
	"FREQS" : [440, 185, 392, 880, 98],	 # Absolute frequency deviants in Hz
	"FREQ_REP_MAX" : 3, # How many times max can frequency deviations occur per trial?
						# If 3, it means there could be 0, 1, 2, or 3 deviants in one trial
						# That's good, because we have 4 buttons in the MRI

	"FIRST_FREQ_LOC" : 2,  # The first tone that can be displaced frequency-wise
						   # e.g.: if you want the first few tones to be frequency standards

	"LAST_FREQ_LOC"  : 7,  # The last tone to be displaced frequency-wise
	}

	for session in range(1000):
		session = session + 1
		create_experimental_sessions(params, session, save_csv=True)