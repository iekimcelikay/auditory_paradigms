#! /usr/bin/env python
# Time-stamp: <2026-03-16 m.utrosa@bcbl.eu>

"""
Creates a "soundtrack", which is a list of "sequences".

soundtrack represents all trials in a single experimental session
sequence represents a single stimulus/trial
"""
import pandas as pd
from pathlib import Path
from expyriment import design, control, stimuli, io, misc

def create_sequences(tone, df, tone_samplerate, tone_bitdepth, save_audio=False):
	"""
	tone is an expyriment tone stimulus.
	df is a dataframe with valid parameter combinations.
	"""

	soundtrack = [] # audio across the entire experimental session
	# Loop through all trials of the experimental session
	# Each trial is a linear combination of parameters
	no_trials = len(df["TRIAL_NO"].unique())
	for trial in df.itertuples():
		
		sequence = [] # audio per trial

		# Identify parts of the combination
		no_tones = trial.NO_TONES
		dev      = trial.DEV
		dev_type = trial.DEV_TYPE
		dev_loc  = trial.DEV_LOC
		isi      = trial.ISI
		iti      = trial.ITI
		run_id   = trial.RUN_NO
		trial_id = trial.TRIAL_NO

		for tone_count in range(no_tones):

			# Correct for zero indexing
			tone_count = tone_count + 1

			# ---- Add the TONE ----
			sequence.append(tone)
			
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

			# Create the ISI as a null tone (inaudible)
			isi_null = stimuli.Tone(
					current_isi,        # duration of the null tone
					40000,              # inaudible frequency (outside the human range)
					tone_samplerate,
					tone_bitdepth
					)
			
			# Note: there's one less null tone (isi) in the sequence than tones.
			if tone_count < (no_tones - 1):
				sequence.append(isi_null)

			# Save tone and isi segments of the sequence as wav
			if save_audio:
				tone.save(f"tone-{tone_count:02d}_trial-{trial_id:02d}_run-{run_id:02d}.wav")
				isi_null.save(f"isi-{tone_count:02d}_trial-{trial_id:02d}_run-{run_id:02d}_len-{current_isi}.wav")

		# ---- Add the ITI ----
		# Note: there's one less ITI than trials.
		print(trial.TRIAL_NO)
		if trial.TRIAL_NO < no_trials - 1:
			iti_null = stimuli.Tone(
					iti,         # duration of the null tone
					40000,       # inaudible frequency (outside the human range)
					tone_samplerate,
					tone_bitdepth
					)
			iti_id = tone_count + 1
			
			if save_audio:
				iti_null.save(f"iti-{iti_id:02d}_len-{iti}_trial-{trial_id:02d}_run-{run_id:02d}.wav")

			sequence.append(iti_null)
		
		# Add the trial sequence to the list of all trials in the exp. session
		soundtrack.append(sequence)

	return soundtrack

# Example usage -------------------------------------------------------------------------------
if __name__ == "__main__":
	params = {
			"TONE_DURATION"   : 50,      # msec
			"TONE_FREQUENCY"  : 440,     # TO-DO: changing frequency
			"TONE_SAMPLERATE" : 48000,   # Change depending on the speakers
			"TONE_BITDEPTH"   : 16,      # Change depending on the speakers
			}
	homePath  = Path("/home/mutrosa/Documents/projects/auditory_paradigms/detection_accuracy/")
	paramPath = homePath / "out" / "exp_parameter_combo_ses-002.csv"
	df   = pd.read_csv(paramPath)
	tone = stimuli.Tone(
			params["TONE_DURATION"],
			params["TONE_FREQUENCY"],
			params["TONE_SAMPLERATE"],
			params["TONE_BITDEPTH"]
			)
	soundtrack = create_sequences(tone, df, 48000, 16)