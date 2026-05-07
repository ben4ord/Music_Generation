"""
Pre-tokenization pipeline — runs ONCE before training begins.

What this script does:
  1. Reads raw .mid files from  Genres/<genre>/
  2. Tokenizes each file with midi_to_tokens() (converts notes -> integer tokens)
  3. Slices the resulting token stream into fixed-length windows of SEQ_LEN tokens
     using a sliding window with overlap (so long songs produce multiple training examples)
  4. Saves each window as a .npy file in  midi_data/<genre>/
  5. Writes midi_data/metadata.csv listing every saved file and its genre label

Why do this up front instead of tokenizing during training?
  Tokenizing a MIDI file involves floating-point time math, sorting, and looping —
  it's slow. If we did it inside the DataLoader at training time, the GPU would sit
  idle waiting for the CPU. Pre-tokenizing lets training read simple .npy arrays,
  which is orders of magnitude faster.
"""
import os
import csv
import random
import numpy as np
from tqdm import tqdm

from Transformer_Music.genres import GENRES, GENRES_ROOT
from Transformer_Music.tokenizer import midi_to_tokens, VOCAB_SIZE, PAD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_ROOT = r"C:\CS-421\Final_Project\midi_data"

SEQ_LEN = 1024  # Every saved sequence is exactly this many tokens long.
                # Must match MAX_SEQ_LEN in model.py.

STRIDE = 768    # Step size of the sliding window.
                # Overlap = SEQ_LEN - STRIDE = 256 tokens.
                # Overlapping windows give the model more training examples
                # and expose it to sequences that start mid-song.

MAX_FILES_PER_GENRE = 250   # Cap on files used per genre.
                            # Prevents heavily-represented genres (e.g. Rock with
                            # 7,000+ files) from dominating training. With this cap
                            # each genre contributes a similar number of sequences.

MAX_SEQS_PER_FILE = 6    # Maximum number of windows extracted from a single file.
                         # Prevents one unusually long song from flooding the dataset.

MAX_BARS = 96            # Tokenizer stops after this many bars. Caps sequence length
                         # for very long pieces so they don't exceed SEQ_LEN.

RANDOM_SEED = 42         # Fixed seed so the file selection is reproducible.

os.makedirs(OUT_ROOT, exist_ok=True)


def slice_sequences(tokens, seq_len, stride, max_seqs):
    """
    Split a long token stream into overlapping fixed-length windows.

    If the token stream is shorter than seq_len, pad it with PAD tokens to
    reach exactly seq_len and return that single padded sequence.

    If it's longer, slide a window of size seq_len over it with step=stride,
    collecting up to max_seqs windows.

    Example with seq_len=8, stride=4:
        tokens = [A B C D E F G H I J]
        windows: [A B C D E F G H]   (start=0)
                     [E F G H I J ?]  ... etc.
    """
    seqs = []
    if len(tokens) <= seq_len:
        # Short sequence: pad with PAD tokens on the right to reach seq_len
        seqs.append(tokens + [PAD] * (seq_len - len(tokens)))
        return seqs

    # Long sequence: sliding window
    for start in range(0, len(tokens) - seq_len + 1, stride):
        seqs.append(tokens[start : start + seq_len])
        if len(seqs) >= max_seqs:
            break  # don't extract more windows than allowed
    return seqs


def main():
    random.seed(RANDOM_SEED)
    rows = []   # accumulates metadata for the CSV
    skipped = 0    # count of files that failed tokenization

    for genre in GENRES:
        genre_dir = os.path.join(GENRES_ROOT, genre)
        if not os.path.isdir(genre_dir):
            print(f"  Warning: {genre} folder not found, skipping")
            continue

        # Create output directory for this genre's .npy files
        out_dir = os.path.join(OUT_ROOT, genre)
        os.makedirs(out_dir, exist_ok=True)

        # Gather all MIDI files in this genre's folder
        mid_files = [
            os.path.join(genre_dir, f)
            for f in os.listdir(genre_dir)
            if f.lower().endswith((".mid", ".midi"))
        ]

        # Shuffle so the cap selects a random representative subset, not just
        # alphabetically first files. The seed ensures reproducibility.
        # This was added to reduce the total number of midi files I was training on.
        random.shuffle(mid_files)
        if len(mid_files) > MAX_FILES_PER_GENRE:
            mid_files = mid_files[:MAX_FILES_PER_GENRE]

        print(f"[{genre}] using {len(mid_files)} files")
        saved = 0

        for path in tqdm(mid_files, desc=genre, unit="file"):
            # --- Tokenize the MIDI file ---
            # Returns a list of integers, or None if the file is unusable.
            tokens = midi_to_tokens(path, max_bars=MAX_BARS)
            if tokens is None or len(tokens) < 32:
                skipped += 1
                continue  # skip empty / too-short files

            # --- Slice into overlapping SEQ_LEN windows ---
            for i, seq in enumerate(slice_sequences(tokens, SEQ_LEN, STRIDE, MAX_SEQS_PER_FILE)):
                # Save as int16 to halve disk usage (token IDs fit in 16 bits since VOCAB_SIZE=187)
                arr = np.array(seq, dtype=np.int16)

                # Build a safe filename: strip special characters, cap at 80 chars
                stem = os.path.splitext(os.path.basename(path))[0]
                stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:80]
                out_name = f"{stem}_seq{i:03d}.npy"
                out_path = os.path.join(out_dir, out_name)

                # Handle rare filename collisions (same stem, different source dirs)
                counter = 1
                while os.path.exists(out_path):
                    out_name = f"{stem}_seq{i:03d}_{counter}.npy"
                    out_path = os.path.join(out_dir, out_name)
                    counter += 1

                np.save(out_path, arr)
                # Record this sequence in the metadata CSV
                rows.append({"file": f"{genre}/{out_name}", "genre": genre})
                saved += 1

        print(f"  -> {saved} sequences saved")

    # --- Write metadata CSV ---
    # The training script reads this file to find all sequences and their labels.
    meta_path = os.path.join(OUT_ROOT, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "genre"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal sequences : {len(rows)}")
    print(f"Skipped files   : {skipped}")
    print(f"Vocab size      : {VOCAB_SIZE}")
    print(f"Metadata        : {meta_path}")


if __name__ == "__main__":
    main()
