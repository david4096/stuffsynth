import os
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from pydub.playback import play
import time
import sys

# Function to load MP3 files using pydub
def load_audio(path):
    audio = AudioSegment.from_mp3(path)
    audio = audio.set_channels(1).set_frame_rate(44100)  # Mono and 44.1 kHz
    return audio

# Function to detect the tempo (BPM) using pydub's built-in beat detection
def detect_bpm(audio):
    # Apply a low-pass filter to emphasize bass drum frequencies (focus on low end)
    audio = audio.low_pass_filter(120.0)

    # Get the average loudness of the entire audio
    beat_loudness = audio.dBFS

    # The fastest tempo allowed is 240 bpm (60000ms / 240beats)
    minimum_silence = int(60000 / 240.0)

    # Detect non-silent parts in the audio
    nonsilent_times = detect_nonsilent(audio, minimum_silence, beat_loudness)

    # Calculate the spaces between peaks (beats)
    spaces_between_beats = []
    last_t = nonsilent_times[0][0]
    
    for peak_start, _ in nonsilent_times[1:]:
        spaces_between_beats.append(peak_start - last_t)
        last_t = peak_start

    # Calculate the median space between beats
    spaces_between_beats = sorted(spaces_between_beats)
    median_space = spaces_between_beats[len(spaces_between_beats) // 2]

    # Estimate the BPM based on the median space
    bpm = 60000 / median_space
    print(f"🔍 Detected BPM: {bpm:.2f}")
    return bpm

# Function to stretch or speed up the audio based on BPM
def stretch_audio(audio, original_bpm, target_bpm):
    # Calculate speed change ratio
    rate = target_bpm / original_bpm
    new_length_ms = len(audio) * (rate)  # Adjust duration based on the BPM ratio
    audio_stretched = audio.speedup(playback_speed=rate)
    print(f"⏱️ Time-stretched to {target_bpm} BPM.")
    return audio_stretched

# Function to split the audio into chunks based on measures
def split_into_measures(audio, bpm, measures=2, output_dir="chunks"):
    beats_per_second = bpm / 60.0
    seconds_per_measure = 4 / beats_per_second  # Assuming 4/4 time signature
    chunk_duration = seconds_per_measure * measures * 1000  # In milliseconds

    total_chunks = int(np.floor(len(audio) / chunk_duration))
    print(f"✂️ Splitting into {total_chunks} chunks of {measures} measure(s) each...")

    os.makedirs(output_dir, exist_ok=True)

    for i in range(total_chunks):
        start_ms = i * chunk_duration
        end_ms = start_ms + chunk_duration
        chunk = audio[start_ms:end_ms]
        output_path = os.path.join(output_dir, f"chunk_{i:03d}.wav")
        chunk.export(output_path, format="wav")
        print(f"🎧 Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bpm_resample_split.py input.mp3")
        sys.exit(1) 
    input_path = sys.argv[1]  
    audio = load_audio(input_path)

    # Detect original BPM
    original_bpm = detect_bpm(audio)

    # Stretch the audio to the target BPM
    target_bpm = 80
    audio_stretched = stretch_audio(audio, original_bpm, target_bpm)

    # Split the audio into 2-measure chunks
    split_into_measures(audio_stretched, target_bpm, measures=2)