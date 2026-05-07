"""
This comment block was created by ChatGPT, it can write a lot better than me.

Autoregressive music generation with constrained sampling.

--- How generation works ---
The trained model was taught (via teacher forcing) to predict the next token
given all previous tokens. At generation time we run that process one step at
a time, feeding each predicted token back as input for the next step:

    [START] → model → predict BAR
    [START, BAR] → model → predict POS_0
    [START, BAR, POS_0] → model → predict TRACK_DRUMS
    ...and so on until END is emitted or max_tokens is reached.

This loop produces an entire token sequence token-by-token.

--- Constrained sampling (structural state machine) ---
The raw model logits sometimes assign probability to tokens that would create
an invalid sequence (e.g. a PITCH token right after a BAR token, with no POS
or TRACK in between). To prevent this, allowed_token_mask() implements a
finite-state machine that returns a boolean mask of which tokens are legal
at each step. Illegal tokens have their logit set to -1e9 (effectively zero
probability) before sampling.

State machine:
    START → BAR only
    BAR   → POS_0 only (every bar must start at beat 0)
    POS   → TRACK tokens only
    TRACK → PITCH tokens only
    PITCH → DUR tokens only
    DUR   → TRACK (another note at same position)
              | POS (move to a later beat in this bar)
              | BAR (start new bar)
              | END (finish piece)

--- Sampling strategies ---
Temperature:  logits /= temperature. Higher → more random; lower → more repetitive.
Top-k:        zero out all logits except the k highest before sampling.
Top-p (nucleus): zero out logits past the cumulative probability threshold p.
Combined top-k + top-p gives diverse but coherent output.
"""
import torch
import torch.nn.functional as F

from Transformer_Music.model import GenreTransformer, MAX_SEQ_LEN
from Transformer_Music.genres import GENRES, GENRE_TO_IDX
from Transformer_Music.tokenizer import (
    tokens_to_midi, VOCAB_SIZE,
    PAD, START, END, BAR,
    POS_OFFSET, TRACK_OFFSET, PITCH_OFFSET, DUR_OFFSET,
    is_pos, is_track, is_pitch, is_dur,
    STEPS_PER_BAR,
)

CKPT_PATH = r"C:\CS-421\Final_Project\checkpoints\best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model handle — loaded once on first call to get_model()
model = None


def get_model():
    """Load the trained model from disk (once) and return it in eval mode."""
    global model
    if model is None:
        model = GenreTransformer().to(device)
        # weights_only=False because the checkpoint was saved with torch.save(state_dict)
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        model.eval()  # disable dropout for deterministic generation
    return model


def allowed_token_mask(prev_tok, cur_pos):
    """
    Return a boolean mask of shape (VOCAB_SIZE,) where True = this token is allowed.

    prev_tok : the most recently generated token
    cur_pos : the current beat position within the bar (0-15), needed to
              enforce that new POS tokens only move forward, never backward

    The mask enforces the structural grammar of the token format:
    BAR → POS → TRACK → PITCH → DUR → (TRACK | POS | BAR | END)
    """
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)

    if prev_tok == START:
        # The very first token after START must be a BAR marker
        mask[BAR] = True

    elif prev_tok == BAR:
        # Every bar starts at position 0 (we don't allow starting a bar mid-beat)
        mask[POS_OFFSET + 0] = True

    elif is_pos(prev_tok):
        # After a position token, only a track (instrument group) token makes sense
        mask[TRACK_OFFSET : PITCH_OFFSET] = True

    elif is_track(prev_tok):
        # After choosing a track, we need the pitch of the note
        mask[PITCH_OFFSET : DUR_OFFSET] = True

    elif is_pitch(prev_tok):
        # After a pitch, we need the duration to complete the note
        mask[DUR_OFFSET : VOCAB_SIZE] = True

    elif is_dur(prev_tok):
        # After completing a note we can:
        #   - play another note on a different track at the same beat
        mask[TRACK_OFFSET : PITCH_OFFSET] = True
        #   - move to a later beat in this bar (never earlier — time flows forward)
        for p in range(cur_pos + 1, STEPS_PER_BAR):
            mask[POS_OFFSET + p] = True
        #   - start a new bar
        mask[BAR] = True
        #   - end the piece
        mask[END] = True

    else:
        # Unknown / unexpected state: allow everything except bookkeeping tokens
        mask[:] = True
        mask[PAD] = False  # PAD is never a valid generated token
        mask[START] = False  # START only appears at the very beginning

    return mask


def sample_tokens(genre_name, max_tokens=1024, temperature=1.0, top_k=40, top_p=0.95, min_bars=8):
    """
    Generate a token sequence one token at a time.

    genre_name : which genre to generate (must be a key in GENRE_TO_IDX)
    max_tokens : hard limit on sequence length (stops generation early if hit)
    temperature : controls randomness — 1.0 = neutral, <1 = conservative, >1 = creative
    top_k : only consider the k most probable tokens at each step
    top_p : nucleus sampling — only consider tokens summing to probability p
    min_bars : don't allow END until at least this many bars have been generated

    Returns a list of integer token IDs.
    """
    model = get_model()
    # Genre is passed as a single-element batch: shape (1,)
    genre_idx = torch.tensor([GENRE_TO_IDX[genre_name]], device=device)

    tokens = [START]   # seed the sequence with the START token
    cur_pos_in_bar = 0         # track beat position for the state machine
    bar_count = 0         # track how many bars we've produced

    with torch.no_grad():
        for _ in range(max_tokens - 1):
            # Feed the last MAX_SEQ_LEN tokens to the model (sliding window if longer)
            inp = torch.tensor([tokens[-MAX_SEQ_LEN:]], device=device, dtype=torch.long)
            logits = model(inp, genre_idx)

            # We only care about the logits at the LAST position (next-token prediction)
            logits = logits[0, -1] / max(1e-6, temperature)

            # Check if the token is allowed
            mask = allowed_token_mask(tokens[-1], cur_pos_in_bar).to(device)
            if bar_count < min_bars:
                mask[END] = False  # force model to keep generating until min_bars
            # Set masked-out logits to a very large negative number → ~0 probability
            logits = logits.masked_fill(~mask, -1e9)

            # Keep only the k highest-scoring tokens; zero out the rest.
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -1e9

            # Sort tokens by probability, accumulate until cumulative prob exceeds p,
            # then zero out everything beyond that cutoff.
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs  = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                # Shift right: we want to include the token that first pushes cumulative > p
                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0]  = False
                sorted_logits[cutoff] = -1e9
                logits = torch.full_like(logits, -1e9).scatter(0, sorted_idx, sorted_logits)

            # --- Sample from the filtered distribution ---
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() <= 0:
                break  # safety: stop if the distribution collapses
            tok = torch.multinomial(probs, 1).item()

            tokens.append(tok)

            # Stop at the END token
            if tok == END:
                break

            # Update position tracking for the state machine
            if tok == BAR:
                bar_count += 1
                cur_pos_in_bar = 0
            elif is_pos(tok):
                cur_pos_in_bar = tok - POS_OFFSET

    return tokens


def generate_midi(genre_name, output_path, temperature=1.1, top_k=40, top_p=0.95, min_bars=8, max_tokens=1024):
    """
    Full pipeline: genre name → MIDI file on disk.

    1. Validates the genre name
    2. Generates a token sequence with sample_tokens()
    3. Converts tokens back to a pretty_midi object with tokens_to_midi()
    4. Writes the MIDI file to output_path

    Returns output_path on success.
    """
    if genre_name not in GENRE_TO_IDX:
        raise ValueError(f"Unknown genre: {genre_name}. Available: {GENRES}")

    tokens = sample_tokens(
        genre_name,
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = top_k,
        top_p = top_p,
        min_bars = min_bars,
    )

    # Decode token list back to a PrettyMIDI object with real instrument tracks
    pm = tokens_to_midi(tokens)

    # Write the .mid file to disk
    pm.write(output_path)
    return output_path
