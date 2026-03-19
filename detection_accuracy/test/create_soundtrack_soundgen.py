#! /usr/bin/env python
# Time-stamp: <2026-03-17 m.utrosa@bcbl.eu>
'''
Test script for create_soundtrack_soundgen() module.
'''
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import sounddevice as sd

# TODO: function from Alejandro as a replacement for thorns
def set_dbspl(sound, dbspl, ref=20e-6):
    """
    Normalize waveform to target dB SPL.

    :param sound : np.array, input waveform
    :param dbspl: float, desired sound level in dB SPL
    :param ref: float, reference pressure (default 20 µPa)
    """

    # Apply dB SPL scaling (RMS based)
    rms = np.sqrt(np.mean(sound**2))
    target_rms = ref * (10 ** (dbspl / 20))
    scale = target_rms / rms
    scaled_sound = sound * scale

    return scaled_sound

class SoundGen:
    def __init__(self, sample_rate, tau):
        """
        Initialize the CreateSound instance.

        :param sample_rate: Sample rate of sounds in Hz.
        :param tau: The ramping window in milliseconds.
        """
        self.sample_rate = sample_rate
        self.tau = tau / 1000 # convert to sec
    
    def sound_maker(self, freq, max_amplitude, num_harmonics, tone_duration, harmonic_factor, dbspl):
        """
        Make a single normalized sound.

        :param freq: Tone frequency in Hz.
        :param max_amplitude: Maximum amplitude to avoid clipping.
        :param num_harmonics: Number of harmonic tones.
        :param tone_duration: Duration of the tone in seconds.
        :param harmonic_factor: Harmonic amplitude decay factor for the tone.
        :param dbspl: Desired dB SPL (loudness) level (cannot change post sound creation).
        
        :return: normalized_sound: an array of audio samples representing a harmonic complex tone.
        """
        
        # Create a time array: each sample represents one event per second 
        t = np.linspace(
            0,                                     # start
            tone_duration,                         # stop
            int(self.sample_rate * tone_duration), # number of samples
            endpoint = False                       # stop is not the last sample
            )
        
        # Initialize a sound array
        sound = np.zeros_like(t)

        # Generate the harmonics
        for k in range(1, num_harmonics + 1):
            harmonic  = np.sin(2 * np.pi * freq * k * t)
            amplitude = max_amplitude * (harmonic_factor ** (k - 1)) / num_harmonics
            sound += amplitude * harmonic

        # Normalize the sound
        normalized_sound = set_dbspl(sound, dbspl)
        
        return normalized_sound

    def sine_ramp(self, sound):
        """ Apply ramping to the start and end of the sound """

        L = int(self.tau * self.sample_rate)
        t = np.linspace(0, L / self.sample_rate, L)
        sine_window = np.sin(np.pi * t / (2 * self.tau)) ** 2  # Sine fade-in

        sound = sound.copy()
        sound[:L] *= sine_window         # Apply fade-in
        sound[-L:] *= sine_window[::-1]  # Apply fade-out

        return sound

    def generate_soundtrack(self, df, base_freq, max_amplitude, num_harmonics, tone_duration, harmonic_factor, dbspl):
        """
        Generate tone sequences with timing deviants.

        :param df: A dataframe with tone sequence parameters with msec as the time unit.
        :param freq: Standard tone frequency in Hz.
        :param max_amplitude: Maximum amplitude to avoid clipping.
        :param num_harmonics: Number of harmonic tones.
        :param tone_duration: Duration of the tone in milliseconds.
        :param harmonic_factor: Harmonic amplitude decay factor for the tone.
        :param dbspl: Desired dB SPL (loudness) level (cannot change post sound creation).
        
        :return: soundtrack: a list of arrays (audio samples), representing harmonic complex tone sequences.
        """
        # Initialize a list to store all tone sequences for this experimental session.
        soundtrack = []

        # Ensure that the trials are ordered by run & trial IDs
        df.sort_values(by=["RUN_NO", "TRIAL_NO"], inplace=True)
        
        # Get number of trials per run
        no_trials = len(df["TRIAL_NO"].unique())

        # Reminder to yourself that we're assuming msec as unit for ISI, ITI, and DEV
        sample_isi = df["ISI"].iloc[0] if not df.empty else "N/A"
        sample_iti = df["ITI"].iloc[0] if not df.empty else "N/A"
        sample_dev = df["DEV"].iloc[0] if not df.empty else "N/A"
        message = (
            "\nThe script converts TONE_DURATION, ISI, ITI, and DEV to seconds, assuming input values are in milliseconds."
            "\nPlease Verify units in the input dataset."
            f"\nUnconverted sample values -> TONE_DURATION: {tone_duration}, ISI: {sample_isi}, ITI: {sample_iti}, DEV: {sample_dev}"
            )
        warnings.warn(message, UserWarning)

        # Convert to sec only once
        tone_duration = tone_duration / 1000

        # Loop through all trials of the experimental session
        # Each trial is a linear combination of parameters
        for trial in df.itertuples():
        
            # Initialize a sequence
            sequence = []

            # Identify parts of the combination
            no_tones = trial.NO_TONES
            dev      = trial.DEV / 1000 # convert to sec
            dev_type = trial.DEV_TYPE
            dev_loc  = trial.DEV_LOC
            freq_dev = trial.FREQ
            freq_loc = trial.FREQ_LOC
            isi      = trial.ISI / 1000 # convert to sec
            iti      = trial.ITI / 1000 # convert to sec
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
            
            # Calculate how many events/samples occur per event
            iti_samples   = int(iti * self.sample_rate)
            isi_samples   = int(isi * self.sample_rate)
            dev_samples   = int(dev * self.sample_rate)

            for i in range(no_tones):

                # Correct for zero indexing
                tone_count = i + 1
                
                # ----------------- Adding TONES ------------------
                if freq_dev != 0:

                    # Generate frequency deviant at the right location
                    if tone_count == freq_loc:
                        sound = self.sound_maker(
                            freq_dev,
                            max_amplitude,
                            num_harmonics,
                            tone_duration,
                            harmonic_factor,
                            dbspl
                            )

                    # Generate frequency standard tone at other locations
                    else:
                        sound = self.sound_maker(
                        base_freq,
                        max_amplitude,
                        num_harmonics,
                        tone_duration,
                        harmonic_factor,
                        dbspl
                        )

                # Generate frequency standard tone sequence
                else:
                    sound = self.sound_maker(
                        base_freq,
                        max_amplitude,
                        num_harmonics,
                        tone_duration,
                        harmonic_factor,
                        dbspl
                        )

                # Apply ramp to start and end using the sine_ramp method
                ramped_sound = self.sine_ramp(sound)
                
                # Add the sound to the sequence
                sequence.append(ramped_sound)

                # ----------------- Adding ISI --------------------
                current_isi = isi_samples

                # Late tones: the ISI before this tone is longer, ISI after shorter.
                if dev_type == "late":
                    if tone_count == (dev_loc - 1): # ISI before
                        current_isi = isi_samples + dev_samples
                    elif tone_count == dev_loc:     # ISI after
                        current_isi = isi_samples - dev_samples

                # Early tones: the ISI before this tone is shorter, ISI after longer.
                elif dev_type == "early":
                    if tone_count == (dev_loc - 1): # ISI before
                        current_isi = isi_samples - dev_samples
                    elif tone_count == dev_loc:     # ISI after
                        current_isi = isi_samples + dev_samples

                # Add the ISI
                # Note: there's one less isi in the sequence than tones.
                if tone_count < no_tones:
                    sequence.append(np.zeros(current_isi))
                
            # ----------------- Adding ITI --------------------
            # Note: there's one less ITI than trials.
            if trial_id < no_trials:
                sequence.append(np.zeros(iti_samples))

            # -------------- Join all segments ----------------
            final_sequence = np.concatenate(sequence)
            soundtrack.append(final_sequence)

        df['AUDIO_SEQUENCE'] = soundtrack
            
        return soundtrack, df
        
# TEST: example usage -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Set the parameters
    params = {
        "PROJECT_ROOT"    : "/home/mutrosa/Documents/projects/auditory_paradigms/detection_accuracy/",  
        "TONE_LOUDNESS"   : 70,     #dB SPL
        "TONE_DURATION"   : 50,     # msec
        "BASE_FREQUENCY"  : 392,    # Hz
        "NUM_HARMONICS"   : 5,      # Number of harmonics
        "HARMONIC_FACTOR" : 0.7,    # Harmonic amplitude decay factor
        "MAX_AMPLITUDE"   : 1.14,   # Defined through a simulation
        "SAMPLE_RATE"     : 48000,  # Hz
        "TAU"             : 5,      # Ramping window in msec
        }

    # Load the trial parameters
    homePath  = Path(params["PROJECT_ROOT"])
    paramPath = homePath / "test" / "test_trials.csv"
    df        = pd.read_csv(paramPath)
    
    # Initialize the class
    sound_gen = SoundGen(params["SAMPLE_RATE"], params["TAU"])

    # Generate the soundtrack of the experimental session
    soundtrack, df = sound_gen.generate_soundtrack(
       df,
       params["BASE_FREQUENCY"],
       params["MAX_AMPLITUDE"],
       params["NUM_HARMONICS"], 
       params["TONE_DURATION"], 
       params["HARMONIC_FACTOR"],
       params["TONE_LOUDNESS"]
       )

    # Play five trials
    for i in range(5):
        sd.play(soundtrack[i], samplerate = params["SAMPLE_RATE"])
        sd.wait()   
    
    # Print info about the return results
    print(df.head())
    print(f"The experiment has {len(soundtrack)} trials.")