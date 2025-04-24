import os
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence_from_file(input_path, output_path, silence_thresh=-40, min_silence_len=1000):
    """
    Removes silence from the beginning and end of an audio file.

    Parameters:
    - input_path: Path to the input .wav file.
    - output_path: Path where the trimmed file will be saved.
    - silence_thresh: The silence threshold in dBFS. Default is -40 dBFS.
    - min_silence_len: The minimum length (in milliseconds) of silence to be removed.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(input_path)
        
        # Split on silence (this will remove silence at the beginning and end)
        segments = split_on_silence(audio, 
                                    min_silence_len=min_silence_len, 
                                    silence_thresh=silence_thresh)
        
        # Join the segments (remove the silence parts)
        trimmed_audio = sum(segments)
        
        # Export the processed file
        trimmed_audio.export(output_path, format="wav")
        print(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_directory(directory_path):
    """
    Processes all .wav files in the given directory by removing silence from their beginning and end.
    
    Parameters:
    - directory_path: Path to the directory containing .wav files.
    """
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return
    
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(directory_path, f"trimmed_{filename}")
            
            remove_silence_from_file(input_path, output_path)

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Remove silence from the beginning and end of .wav files.")
    parser.add_argument("directory", help="Path to the directory containing .wav files.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Process the directory
    process_directory(args.directory)

if __name__ == "__main__":
    main()
