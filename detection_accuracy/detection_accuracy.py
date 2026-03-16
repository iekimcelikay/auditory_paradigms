#! /usr/bin/env python
# Time-stamp: <2026-03-16 m.utrosa@bcbl.eu>

# 00. PREPARATION ---------------------------------------------------------------------------------
## Start by importing the neccesary modules and packages. If you do not have the python packages
## installed on your laptop, you can install them with: pip install {package name}.
import random
import numpy as np
from itertools import product
from expyriment import design, control, stimuli, io, misc

control.set_develop_mode(on=True) # developping == True / testing == False.

# 01. DEFINE FUNCTIONS  ---------------------------------------------------------------------------
def create_sequences(tone, param_combos, iti):
	"""
	tone is an expyriment tone stimulus.
	null is a null sound from Moerell's sound database.
	param_combos is a random sample of valid parameter combinations.

	"""
	# Loop through runs (experimental blocks)
	soundtrack = []
	for cr, run in enumerate(param_combos):
		run_id = cr + 1

		sequence = []
		# For each combination of parameters in a block
		for ct, trial in enumerate(run):
			trial_id = ct + 1

			# Identify which part of the combination refers to which parameter
			# TODO: better if dictionary (instead of tuple)
			no_tones = trial[0]
			dev      = trial[1]
			dev_type = trial[2]
			dev_loc  = trial[3]
			isi      = trial[4]

			for count in range(no_tones):
				
				# Add the tone ----
				sequence.append(tone)
				tone_id = count + 1
				# tone.save(f"tone-{tone_id:02d}_trial-{trial_id:02d}_run-{run_id:02d}.wav")

				# Add the isi ----
				current_isi = isi

				# For late tones, the ISI before the diplaced tone is longer, ISI after shorter.
				if dev_type == "late":
					if count == (dev_loc - 1): # ISI before
						current_isi = isi + dev
					elif count == dev_loc: # ISI after
						current_isi = isi - dev

				# For early tones, the ISI before the diplaced tone is shorter, ISI after longer.
				elif dev_type == "early":
					if count == (dev_loc - 1):
						current_isi = isi - dev
					elif count == dev_loc:
						current_isi = isi + dev

				isi_null = stimuli.Tone(
						current_isi, # duration of the null tone
						40000,       # inaudible frequency (outside the human range)
						params["TONE_SAMPLERATE"],
						params["TONE_BITDEPTH"]
						)
				isi_id = count + 1
				# isi_null.save(f"isi-{isi_id:02d}_len-{current_isi}_trial-{trial_id:02d}_run-{run_id:02d}.wav")

				# There is one less null tone (isi) in the sequence
				if count < (no_tones - 1):
					sequence.append(isi_null)

			# Add an iti ----
			# There is one less ITI than the number of trials
			if ct < len(run) - 1:
				iti_null = stimuli.Tone(
						iti,         # duration of the null tone
						40000,       # inaudible frequency (outside the human range)
						params["TONE_SAMPLERATE"],
						params["TONE_BITDEPTH"]
						)
				iti_id = count + 1
				# iti_null.save(f"iti-{iti_id:02d}_len-{iti}_trial-{trial_id:02d}_run-{run_id:02d}.wav")

			sequence.append(iti_null)
			
		soundtrack.append(sequence)

	return soundtrack

# 02. SET PARAMETERS ------------------------------------------------------------------------------
params = {

	# Visual
	"CANVAS_SIZE" : (1920, 1080), # Monitor resolution. PC: 1920, 1080. MRI: 1024, 768.
	"FIXATION_CROSS_SIZE"     : (20, 20),
	"FIXATION_CROSS_POSITION" : (0, 0),
	"FIXATION_CROSS_WIDTH"    : 4,
	"HEADING_SIZE" : 30, 
	"TEXT_SIZE"    : 20,
	"TEXT" 		   : f"Identify the sound that is displaced when hearing it. \n"
					  "Press a button to start.",

	# Audio
	"MIN_TONES": 3,             # min. no. of tones in a single sequence
	"MAX_TONES" : 7,            # max. no. of tones in a single sequence
	"TONE_DURATION" : 50,       # msec
	"TONE_FREQUENCY" : 440,     # TO-DO: changing frequency
	"TONE_SAMPLERATE" : 48000,  # Change depending on the speakers
	"TONE_BITDEPTH" : 16,       # Change depending on the speakers

	# Colors in RGB
	"BLACK"  : (0, 0, 0),	    # screen background
	"WHITE"  : (255, 255, 255), # fixation cross
	"GREEN"  : (50, 205, 50),   # correct response
	"RED"    : (204, 0, 0),     # incorrect reponse
	"CYAN"   : (0,255,255),     # correct for colorblind
	"ORANGE" : (255,165,0),     # incorrect for colorblind

	# Experiment Structure
	"NO_RUNS" : 4,
	"ITI" : 1500,     		 # Inter Trial Interval (time between two sequences, i.e., trials)       
					  		 # Should be longer than the largest ISI and DEV.
	"NO_TRIALS" : 10,		 # The number of trial repetitions. A trial is a single tone sequence.
	
	# ISI: Inter Stimulus Interval (time between presentation of two sequential tones).
	"ISI_MIN": 700, # Minimum duration of ISI (inclusive)
	"ISI_MAX": 701, # Maximum duration of ISI (exclusive)
	"ISI_NO" : 1,   # How many ISI values to test?
	
	# Deviations
	"DEV_MIN": 1,   # Minimum tone timing deviation (inclusive) 
	"DEV_MAX": 300, # Maximum tone timing deviation ISI (exclusive)
	"DEV_NO" : 64,  # How many deviations to test? !!! TODO: this is not accurate as we correct for no signal trials to calculate sensitivity!!
	}

# 03. STRUCTURE THE EXPERIMENT --------------------------------------------------------------------
# TODO: use durations to ensure approximately the same duration of runs (Important for MRI protocol!)
# COMBO_DURATIONS = calculate_trial_duration(VALID_COMBOS, params)
# paired_trials = list(zip(VALID_COMBOS, COMBO_DURATIONS))
# random.shuffle(paired_trials)

# 04. INITIALIZE THE EXPERIMENT -------------------------------------------------------------------
exp = design.Experiment(name="timingDev")
control.initialize(exp)

# 05. CREATE & PRELOAD THE STIMULI ----------------------------------------------------------------
keyboard = io.Keyboard()
instructions = stimuli.TextScreen(
	"Instructions", params["TEXT"],
	heading_size = params["HEADING_SIZE"],
	text_size = params["TEXT_SIZE"]
	)
instructions.preload()
canvas = stimuli.Canvas(size = params["CANVAS_SIZE"], colour = params["BLACK"])
cross  = stimuli.FixCross(
	size = params["FIXATION_CROSS_SIZE"],
	position = params["FIXATION_CROSS_POSITION"],
	line_width = params["FIXATION_CROSS_WIDTH"],
	colour = params["WHITE"]
	)
cross.preload()
cross.plot(canvas) # Plot on canvas now, less to do later
canvas.preload()

tone = stimuli.Tone(
		params["TONE_DURATION"],
		params["TONE_FREQUENCY"],
		params["TONE_SAMPLERATE"],
		params["TONE_BITDEPTH"]
		)

soundtrack = create_sequences(tone, RUN_COMBOS, params["ITI"])

# Set up data storage
exp.add_data_variable_names(['TRIAL_NO', 'NO_TONES', 'DEV', 'DEV_TYPE', 'DEV_LOC', 'ISI', 'RESPONSE'])

# 06. RUN THE EXPERIMENT ---------------------------------------------------------------------------
control.start(skip_ready_screen=True)

# Show the instructions and wait for the participant to read the instructions.
instructions.present()
keyboard.wait(keys=[misc.constants.K_2])

canvas.present()

# Wait for the MRI scanner signal the start of functional sequence to sync the task.
# keyboard.wait(keys=[misc.constants.K_s])

# # Start the task after the 4th trigger. One "s" trigger per TR "trigger".
# keyboard.wait(keys=[misc.constants.K_s])
# keyboard.wait(keys=[misc.constants.K_s])
# keyboard.wait(keys=[misc.constants.K_s])
# keyboard.wait(keys=[misc.constants.K_s])

for cr, r in enumerate(soundtrack):
	
	for cs, s in enumerate(r):

		# Enable ESCAPE
		keyboard.check(keys=[misc.constants.K_ESCAPE])
		
		# s.save(f"sound-{cs+1}_run-{cr+1}.wav")
		s.play()
		s.wait_end()
	
	exp.clock.wait(5000) # pause between runs

# 07. END THE EXPERIMENT --------------------------------------------------------------------------
control.end()