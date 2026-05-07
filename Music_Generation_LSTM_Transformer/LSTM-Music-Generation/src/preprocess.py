from pathlib import Path
import pretty_midi

# Create tokens from MIDI files by extracting note pitches in order of their start times. 
# Each token is a MIDI pitch (0-127). 
# We ignore drum tracks and only consider melodic instruments. 
# We also filter out sequences that are too short to be useful for training.
def midi_to_tokens(midi_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    events = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            events.append({
                "start": note.start,
                "pitch": note.pitch,
                "program": instrument.program 
            })

    # Sort notes by their start time to maintain the correct order of events
    events.sort(key=lambda e: e["start"])
    # Create tokens by combining the instrument program number and pitch into a single integer.
    # This allows us to represent both the note and the instrument in a single token
    tokens = [event["program"] * 128 + event["pitch"] for event in events]

    return tokens


# Load in MIDI files from a directory structure where each subfolder is a genre. 
# We return token sequences, genre ids, and a mapping of genre to id for later use
def load_midi_dataset(data_dir, min_tokens=129):
    data_dir = Path(data_dir)

    token_sequences = []
    genre_ids = []

    genres = sorted([
        folder.name
        for folder in data_dir.iterdir()
        if folder.is_dir()
    ])

    # Create genre ids from folder names
    genre_to_id = {genre: i for i, genre in enumerate(genres)}

    for genre in genres:
        genre_folder = data_dir / genre

        for midi_path in genre_folder.rglob("*.mid"):
            
            try:
                tokens = midi_to_tokens(midi_path)
                #check the length of the token sequence and only keep those that are long enough for training
                # Dont want to keep sequences that are too short, since we need at least 128 tokens for input and 128 for target
                # this helps ensure we have enough data to train on and that the model can try to learn meaningful patterns rather than just memorizing short sequences
                if len(tokens) >= min_tokens:
                    token_sequences.append(tokens)
                    genre_ids.append(genre_to_id[genre])

            #skip files that can't be processed and print an error message
            except Exception as e:
                print(f"Skipping {midi_path}: {e}")

    print(f"Loaded {len(token_sequences)} MIDI files across {len(genres)} genres.")
    print("Preprocessing complete.")
               

    return token_sequences, genre_ids, genre_to_id
