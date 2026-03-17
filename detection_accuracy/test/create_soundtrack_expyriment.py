#! /usr/bin/env python
# Time-stamp: <2026-03-16 m.utrosa@bcbl.eu>
'''
Test script for create_soundtrack_expyriment() function.
'''

import pandas as pd
from pathlib import Path
from expyriment import design, control, stimuli, io, misc

def create_soundtrack_expyriment(df, samplerate, bitdepth, tone_duration, tone_frequency, null_frequency, save_audio=False):
	"""
	Generate a list of tone sequences using Expyriment's Tone class.

	Args:
		df (pd.DataFrame): Dataframe with trial parameters.
		samplerate (int): Sampling rate for null audio generation.
		bitdepth (int): Bit depth for null audio generation.
		tone_duration (int): Base tone duration in milliseconds.
		tone_frequency (int): Base tone frequency in Hz for tone audio generation.
		null_frequency (int): Inaudible frequency in Hz for null audio generation.
		save_audio (bool): If true, saves individual .wav files (segments of the tone sequence).

	Returns:
		soundtrack: a list of lists, where each sublist is a tone sequence
	"""
   	# Initialize a list to store all tone sequences for this experimental session.
	soundtrack = []

    # Ensure that the trials are ordered by run & trial IDs
    df.sort_values(by=["RUN_NO", "TRIAL_NO"], inplace=True)
	
	# Get number of trials per run
	no_trials = len(df["TRIAL_NO"].unique())

	# Loop through all trials of the experimental session
	# Each trial is a linear combination of parameters
	for trial in df.itertuples():
		
		# Initialize a sequence
		sequence = []

		# Identify parts of the combination
		no_tones = trial.NO_TONES
		dev      = trial.DEV
		dev_type = trial.DEV_TYPE
		dev_loc  = trial.DEV_LOC
		freq_dev = trial.FREQ
		freq_loc = trial.FREQ_LOC
		isi      = trial.ISI
		iti      = trial.ITI
		trial_id = trial.TRIAL_NO
		run_id   = trial.RUN_NO

		# Raise error if timing and frequency deviants occur on the same tone
		if freq_loc == dev_loc:
			raise ValueError(
				f"\nFor trial-{trial_id:02d} run-{run_id:02d}."
				" Frequency and timing deviations "
				f"occur on the same tone (no. {dev_loc})."
				"\nCheck your input dataframe."
				" Parameter combinations may be incorrectly set."
			)

		for i in range(no_tones):

			# Correct for zero indexing
			tone_count = i + 1

			# ----------------- Adding TONES ------------------
			if freq_dev != 0:

				# Generate frequency deviant at the right location
				if tone_count == freq_loc:
					tone = stimuli.Tone(
						tone_duration,
						freq_dev,
						samplerate,
						bitdepth
					)

				# Generate frequency standard tone at other locations
				else:
					tone = stimuli.Tone(
						tone_duration,
						tone_frequency,
						samplerate,
						bitdepth
					)
			
			# Generate frequency standard tone sequence
			else:
				tone = stimuli.Tone(
					tone_duration,
					tone_frequency,
					samplerate,
					bitdepth
				)

			# Add the sound to the sequence
			sequence.append(tone)

			# Save tone segment of the sequence as a .wav file
			if save_audio:
				tone.save(f"tone-{tone_count:02d}_trial-{trial_id:02d}_run-{run_id:02d}.wav")

			# ----------------- Adding ISI --------------------
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
					samplerate,
					bitdepth
				)
				sequence.append(isi_null)
				
				# Save isi segment of the sequence as a .wav file
				if save_audio:
					isi_null.save(f"isi-{tone_count:02d}_trial-{trial_id:02d}_run-{run_id:02d}_len-{current_isi}.wav")

		# ----------------- Adding ITI --------------------
		# Note: there's one less ITI than trials.
		if trial_id < no_trials:
			iti_null = stimuli.Tone(
				iti,          
				null_frequency,
				samplerate,
				bitdepth
			)
			sequence.append(iti_null)

			if save_audio:
				iti_null.save(f"iti-{tone_count:02d}_len-{iti}_trial-{trial_id:02d}_run-{run_id:02d}.wav")
		
		# -------------- Join all segments ----------------
		soundtrack.append(sequence)

	df['AUDIO_SEQUENCE'] = soundtrack

	return soundtrack, df

# TEST: example usage -----------------------------------------------------------------------------
if __name__ == "__main__":
	
	# Set the parameters
	params = {
		"PROJECT_ROOT"   : "/home/mutrosa/Documents/projects/auditory_paradigms/detection_accuracy/",
		"TONE_DURATION"  : 50,      # msec
		"BASE_FREQUENCY" : 440,     # Base frequency in Hz
		"NULL_FREQUENCY" : 44000,   # Null sound frequency in Hz
		"SAMPLERATE"     : 48000,   # Change depending on the audio devices
		"BITDEPTH"       : 16,      # Change depending on the audio devices
		}
	
	# Load the trial parameters
	homePath  = Path(params["PROJECT_ROOT"])
	paramPath = homePath / "test" / "test_trials.csv"
	df        = pd.read_csv(paramPath)
	
	# Generate the soundtrack of the experimental session
	soundtrack, df = create_soundtrack_expyriment(
		df,
		params["SAMPLERATE"],
		params["BITDEPTH"],
		params["TONE_DURATION"],
		params["TONE_FREQUENCY"],
		params["NULL_FREQUENCY"],
		)

	# Play five trials

	# Print info about the return results
	print(df.head())
	print(f"The experiment has {len(soundtrack)} trials.")