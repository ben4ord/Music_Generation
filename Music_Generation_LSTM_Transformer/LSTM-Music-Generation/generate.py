import torch
import torch.nn.functional as F
import pretty_midi
import random
from argparse import ArgumentParser
from pathlib import Path

from src.model import LSTMModel
from src.preprocess import midi_to_tokens

def generate(
    model,
    start_tokens,
    genre_id,
    max_length=500,
    temperature=1.0,
    device="cuda"
    ):
    model.eval()

    # Start with the initial tokens and genre conditioning, then iteratively sample the next token and feed it back in
    generated = list(start_tokens)
    tokens = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)
    genre = torch.tensor([genre_id], dtype=torch.long).to(device)

    hidden = None
    # dont need gradients for generation, since we are not training
    with torch.no_grad():
        #send the initial tokens and genre conditioning through the model to get the initial hidden state
        logits, hidden = model(tokens, genre, hidden)
        # Iteratively sample tokens until we reach the max length
        for _ in range(max_length):
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)

            next_token = torch.multinomial(probs, 1)
            generated.append(next_token.item())

            logits, hidden = model(next_token, genre, hidden)

    return generated

# Utility function to convert a sequence of tokens back into a MIDI file
def tokens_to_midi(
    tokens,
    output_path,
    note_duration=0.25,
    velocity=90,
):
    midi = pretty_midi.PrettyMIDI()

    instruments = {}
    for i, token in enumerate(tokens):
        token = int(token)
        # need to decode both the instrument program number and pitch from the token
        program = token // 128
        pitch = token % 128 

        if program < 0 or program > 127:
            print(f"Warning: Invalid program number {program} decoded from token {token}. Skipping note.")
            continue

        if program not in instruments:
            instruments[program] = pretty_midi.Instrument(program=program)
        
        # Calculate the start and end times for the note based on its position in the sequence and the specified note duration
        start = i * note_duration
        end = start + note_duration

        # Create a PrettyMIDI Note object with the decoded pitch, 
        # specified velocity, and calculated start/end times, 
        # and add it to the appropriate instrument track
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instruments[program].notes.append(note)

    for instrument in instruments.values():
        midi.instruments.append(instrument)

    midi.write(str(output_path))


# get a mapping of genre names to ids based on the subfolder structure in the data directory
def load_genres(data_dir):
    data_dir = Path(data_dir)
    genres = sorted(folder.name for folder in data_dir.iterdir() if folder.is_dir())
    return {genre: i for i, genre in enumerate(genres)}

# Load a random seed token sequence from a MIDI file in the specified genre, 
# to use as the starting point for generation. 
def load_dataset_start_tokens(data_dir, genre, start_length):
    genre_dir = Path(data_dir) / genre
    midi_paths = sorted(genre_dir.rglob("*.mid"))

    if not midi_paths:
        raise ValueError(f"No .mid files found for genre '{genre}' in {genre_dir}")

    random.shuffle(midi_paths)

    for midi_path in midi_paths:
        try:
            tokens = midi_to_tokens(midi_path)
        except Exception as e:
            print(f"Skipping seed MIDI {midi_path}: {e}")
            continue

        if len(tokens) < start_length:
            continue
        
        # Randomly select a contiguous sequence of tokens from the MIDI file to use as the seed for generation.
        max_start = len(tokens) - start_length
        start_index = random.randint(0, max_start)
        print(f"Using dataset seed: {midi_path}")
        print(f"Seed token range: {start_index}:{start_index + start_length}")
        return tokens[start_index:start_index + start_length]

    raise ValueError(
        f"No usable .mid files with at least {start_length} tokens found for genre '{genre}'"
    )



def main():

    # We use argument parser to allow us to specify various parameters for generation, 
    # such as the genre to condition on, the length of the generated sequence,
    # the temperature for sampling, and the starting token sequence.
    # This makes it easy to experiment with different generation settings without having to modify the code.
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="data/midi")
    parser.add_argument("--genre", default="Pop")
    parser.add_argument("--output", default="generated.mid")
    # Generation parameters
    # Length is how many tokens to generatete
    # Temperature controls randomness higher is more random
    # start is the initial token sequence to condition on, seed is for reproducibility
    # note-duration controls the length of each note in the generated MIDI file
    parser.add_argument("--length", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.5)
    # Start with a few tokens to prime the model. These could be randomly selected from the training data or set to specific values to try to steer the generation in a certain direction.
    parser.add_argument("--start", type=int, nargs="+", default=[10, 65, 35, 34])
    parser.add_argument(
        "--start-source",
        choices=["dataset", "manual"],
        default="dataset",
        help="dataset picks a seed from a real MIDI in the selected genre; manual uses --start."
    )
    parser.add_argument(
        "--start-length",
        type=int,
        default=32,
        help="Number of tokens to use when --start-source dataset is selected."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--note-duration", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genre_to_id = load_genres(args.data_dir)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if args.genre not in genre_to_id:
        available = ", ".join(genre_to_id)
        raise ValueError(f"Unknown genre '{args.genre}'. Available genres: {available}")

    genre_id = genre_to_id[args.genre]
    print(f"Using genre: {args.genre} (id {genre_id})")

    if args.start_source == "dataset":
        start_tokens = load_dataset_start_tokens(
            data_dir=args.data_dir,
            genre=args.genre,
            start_length=args.start_length
        )
    else:
        start_tokens = args.start

    print(f"Start tokens: {start_tokens[:24]}")

    # Load the trained model checkpoint and set up the model for generation
    # Multply vocab_size by 128 to account for the fact that we are encoding both instrument and pitch into a single token
    model = LSTMModel(vocab_size=128 * 128, genre_count=len(genre_to_id))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Generate a new sequence of tokens conditioned on the specified genre and starting tokens
    tokens = generate(
        model=model,
        start_tokens=start_tokens,
        genre_id=genre_id,
        max_length=args.length,
        temperature=args.temperature,
        device=device
    )

    print(f"First generated tokens: {tokens[:24]}")

    # Convert the generated token sequence back into a MIDI file and save it
    tokens_to_midi(
        tokens,
        args.output,
        note_duration=args.note_duration
    )
    print(f"Saved generated MIDI to {args.output}")


if __name__ == "__main__":
    main()
