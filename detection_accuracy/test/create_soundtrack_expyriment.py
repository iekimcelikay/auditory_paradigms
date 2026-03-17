#! /usr/bin/env python
# Time-stamp: <2026-03-16 m.utrosa@bcbl.eu>
'''
Test script for create_soundtrack_expyriment() function.
Instead of adding the tone stimuli to the soundtrack, strings are added.
Strings represent the stimuli that would be added to the sequence to
check consistency.
'''

import pandas as pd
from pathlib import Path
from expyriment import design, control, stimuli, io, misc

def create_soundtrack_expyriment(tone, df, null_frequency, null_samplerate, null_bitdepth, save_audio=False):
	"""
	Generate a list of tone sequences using Expyriment's Tone class.

	Args:
		tone (expyriment.stimuli.Tone): The base tone stimulus.
		df (pd.DataFrame): Dataframe with trial parameters.
		null_frequency (int): Frequency for null audio generation.
		null_samplerate (int): Sampling rate for null audio generation.
		null_bitdepth (int): Bit depth for null audio generation.
		save_audio (bool): If true, saves individual .wav files.

	Returns:
		soundtrack: a list of lists, where each sublist is a tone sequence
	"""

	no_trials = len(df["TRIAL_NO"].unique())
	soundtrack = [] # audio across the entire experimental session

	# Loop through all trials of the experimental session
	# Each trial is a linear combination of parameters
	for trial in df.itertuples():
		
		sequence = [] # all audio stimuli per trial

		# Identify parts of the combination
		no_tones = trial.NO_TONES
		dev      = trial.DEV
		dev_type = trial.DEV_TYPE
		dev_loc  = trial.DEV_LOC
		isi      = trial.ISI
		iti      = trial.ITI
		run_id   = trial.RUN_NO
		trial_id = trial.TRIAL_NO

		for i in range(no_tones):

			# Correct for zero indexing
			tone_count = i + 1

			# ---- Add the TONE ----
			# sequence.append(tone)
			sequence.append("TONE")

			# Save tone and isi segments of the sequence as wav
			if save_audio:
				tone.save(f"tone-{tone_count:02d}_trial-{trial_id:02d}_run-{run_id:02d}.wav")

			# ---- Add the ISI ----
			current_isi = isi

			# Late tones: the ISI before this tone is longer, ISI after shorter.
			if dev_type == "late":
				if tone_count == (dev_loc - 1): # ISI before
					current_isi = isi + dev
				elif tone_count == dev_loc:     # ISI after
					current_isi = isi - dev

			# Early tones: the ISI before this tone is shorter, ISI after longer.
			elif dev_type == "early":
				if tone_count == (dev_loc - 1): # ISI before
					current_isi = isi - dev
				elif tone_count == dev_loc:     # ISI after
					current_isi = isi + dev
			
			# Create and add the ISI (null tone)
			# Note: there's one less isi in the sequence than tones.
			if tone_count < no_tones:
				isi_null = stimuli.Tone(
					current_isi,          
					null_frequency,      
					null_samplerate,
					null_bitdepth
					)
				# sequence.append(isi_null)
				sequence.append("ISI")
				
				# Save tone and isi segments of the sequence as wav
				if save_audio:
					isi_null.save(f"isi-{tone_count:02d}_trial-{trial_id:02d}_run-{run_id:02d}_len-{current_isi}.wav")

		# ---- Add the ITI ----
		# Note: there's one less ITI than trials.
		if trial_id < no_trials:
			iti_null = stimuli.Tone(
					iti,          
					null_frequency,
					null_samplerate,
					null_bitdepth
					)
			# sequence.append(iti_null)
			sequence.append("ITI")

			if save_audio:
				iti_null.save(f"iti-{tone_count:02d}_len-{iti}_trial-{trial_id:02d}_run-{run_id:02d}.wav")
		
		# Add the trial sequence to the list of all trials in the exp. session
		soundtrack.append(sequence)

	return soundtrack

# TEST: example usage -----------------------------------------------------------------------------
if __name__ == "__main__":
	
	# Set the parameters
	params = {
		"PROJECT_ROOT"    : "/home/mutrosa/Documents/projects/auditory_paradigms/detection_accuracy/",
		"TONE_DURATION"   : 50,      # msec
		"NULL_FREQUENCY"  : 40000,   # Null sound frequency in Hz
		"TONE_FREQUENCY"  : 440,     # Base frequency in Hz
		"TONE_SAMPLERATE" : 48000,   # Change depending on the audio devices
		"TONE_BITDEPTH"   : 16,      # Change depending on the audio devices
			}
	
	# Load the trial parameters
	homePath  = Path(params["PROJECT_ROOT"])
	paramPath = homePath / "test" / "test_trials.csv"
	df        = pd.read_csv(paramPath)
	
	# Create a standard tone
	tone = stimuli.Tone(
			params["TONE_DURATION"],
			params["TONE_FREQUENCY"],
			params["TONE_SAMPLERATE"],
			params["TONE_BITDEPTH"]
			)

	# Create the soundtrack
	soundtrack = create_soundtrack_expyriment(
		tone,
		df,
		params["NULL_FREQUENCY"],
		params["TONE_SAMPLERATE"],
		params["TONE_BITDEPTH"]
		)

	print(f"The experiment has {len(soundtrack)} trials.")
	print(f"\nThe first trial in a block: {soundtrack[0]}")
	print(f"\nThe last trial in a block: {soundtrack[-1]}")