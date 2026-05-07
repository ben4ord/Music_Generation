"""
This comment block was written by ChatGPT

Multi-track REMI-style tokenizer for the Lakh_MIDI / Genres dataset.

--- What is tokenization? ---
A neural network can only process numbers, not raw MIDI data.
Tokenization converts each MIDI file into a flat sequence of integer "tokens"
that the model can learn to predict one at a time (like words in a sentence).

--- Token scheme (REMI multi-track) ---
Every piece becomes a stream like:
    START BAR POS_0 TRACK_DRUMS PITCH_36 DUR_4 TRACK_BASS PITCH_36 DUR_8
          POS_4 TRACK_DRUMS PITCH_38 DUR_2 ... BAR POS_0 ... END

Each group of four tokens (POS, TRACK, PITCH, DUR) represents one note:
  - POS   : when in the bar the note starts (16th-note grid, 0-15)
  - TRACK : which instrument group plays the note (DRUMS, BASS, PIANO, etc.)
  - PITCH : MIDI pitch number (0-127)
  - DUR   : how long the note lasts (1-32 sixteenth-notes)

Time grid: 16 positions per bar = 16th-note resolution, 4/4 time assumed.

--- Instrument grouping ---
128 GM programs are collapsed into 7 instrument groups so the vocabulary stays
small and the model generalizes across different piano patches, guitar sounds, etc.

GM program -> track group:
    0-23   Piano / Organ    -> TRACK_PIANO
    24-31  Guitar           -> TRACK_GUITAR
    32-39  Bass             -> TRACK_BASS
    40-55  Strings          -> TRACK_STRINGS
    56-87  Brass/Reed/Lead  -> TRACK_LEAD
    88-127 Synth Pad/Ethnic -> TRACK_PAD
    drum channel 9          -> TRACK_DRUMS
"""
from __future__ import annotations
import pretty_midi

STEPS_PER_BAR = 16   # 16 positions per bar (16th-note resolution in 4/4)
MAX_DURATION  = 32   # Longest allowed note: 32 sixteenth-notes = 2 full bars

# Seven broad categories that cover all 128 GM programs + drums.
TRACK_NAMES  = ["DRUMS", "BASS", "PIANO", "GUITAR", "STRINGS", "LEAD", "PAD"]
N_TRACKS     = len(TRACK_NAMES)
TRACK_TO_IDX = {n: i for i, n in enumerate(TRACK_NAMES)}

# When decoding tokens back to a MIDI file, each group is assigned one
# representative GM program number so the output has real-sounding instruments.
TRACK_DEFAULT_PROGRAM = {
    "DRUMS":   0,    # program is ignored for drum tracks; is_drum=True handles it
    "BASS":    33,   # Electric Bass (finger)
    "PIANO":   0,    # Acoustic Grand Piano
    "GUITAR":  25,   # Acoustic Guitar (steel)
    "STRINGS": 48,   # String Ensemble 1
    "LEAD":    56,   # Trumpet (used as a generic lead voice)
    "PAD":     88,   # New Age Pad
}


def program_to_track(program: int, is_drum: bool) -> str:
    """Map a GM program number (0-127) and drum flag to one of the 7 track groups."""
    if is_drum:
        return "DRUMS"
    if program <= 23:
        return "PIANO"     # Grand piano, electric piano, organ, etc.
    if program <= 31:
        return "GUITAR"    # Nylon, steel, electric guitar
    if program <= 39:
        return "BASS"      # Acoustic, electric, fretless bass
    if program <= 55:
        return "STRINGS"   # Violin, viola, cello, string ensemble
    if program <= 87:
        return "LEAD"      # Trumpet, sax, flute, synth lead
    return "PAD"           # Synth pad, choir, ethnic, effects


# ---------------------------------------------------------------------------
# Vocabulary layout
# ---------------------------------------------------------------------------
# All token IDs are packed into fixed contiguous ranges so that the model
# (and the sampling state machine) can identify a token's type by checking
# which range it falls in — no lookup table needed.
#
# ID   token
# ---  -----
#  0   PAD    (padding — ignored by the loss function)
#  1   START  (prepended to every sequence; marks the beginning)
#  2   END    (marks the end of the piece)
#  3   BAR    (marks the start of a new bar)
#  4-19  POS_0 .. POS_15   (position within the bar, 16th-note grid)
# 20-26  TRACK_0 .. TRACK_6 (instrument group index)
# 27-154 PITCH_0 .. PITCH_127 (MIDI pitch)
# 155-186 DUR_1 .. DUR_32  (note duration in 16th-notes)
PAD = 0
START = 1
END = 2
BAR = 3

POS_OFFSET = 4                             # IDs 4-19  → 16 position tokens
TRACK_OFFSET = POS_OFFSET + STEPS_PER_BAR  # IDs 20-26 → 7 track tokens
PITCH_OFFSET = TRACK_OFFSET + N_TRACKS     # IDs 27-154 → 128 pitch tokens
DUR_OFFSET = PITCH_OFFSET + 128            # IDs 155-186 → 32 duration tokens

VOCAB_SIZE = DUR_OFFSET + MAX_DURATION     # 187 total tokens in the vocabulary


# --- Encoder helpers: value -> token ID ---
def pos_token(p: int) -> int: return POS_OFFSET + p
def track_token(t: int) -> int: return TRACK_OFFSET + t
def pitch_token(p: int) -> int: return PITCH_OFFSET + p
def dur_token(d: int) -> int: return DUR_OFFSET + (d - 1)  # DUR_1 stored at offset 0

# --- Type predicates: is this token ID in a given range? ---
def is_pos(tok): return POS_OFFSET   <= tok < TRACK_OFFSET
def is_track(tok): return TRACK_OFFSET <= tok < PITCH_OFFSET
def is_pitch(tok): return PITCH_OFFSET <= tok < DUR_OFFSET
def is_dur(tok): return DUR_OFFSET   <= tok < VOCAB_SIZE

# --- Decoder helpers: token ID -> original value ---
def pos_value(tok): return tok - POS_OFFSET
def track_value(tok): return tok - TRACK_OFFSET
def pitch_value(tok): return tok - PITCH_OFFSET
def dur_value(tok): return tok - DUR_OFFSET + 1  # undo the -1 from encoding


# ===========================================================================
# ENCODE: MIDI file -> integer token list
# ===========================================================================
def midi_to_tokens(path: str, max_bars: int = 64) -> list[int] | None:
    """
    Read a .mid file and convert it to a flat list of integer tokens.

    The output follows the pattern:
        [START, BAR, POS, TRACK, PITCH, DUR, TRACK, PITCH, DUR, ..., BAR, ..., END]

    Returns None if the file is unreadable, empty, or too short.
    """
    # --- 1. Load the MIDI file ---
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None  # corrupt or unsupported MIDI files are silently skipped

    # --- 2. Determine tempo and convert to 16th-note grid step size ---
    tempos = pm.get_tempo_changes()[1]
    tempo = float(tempos[0]) if len(tempos) else 120.0
    # Clamp extreme tempos that would distort the grid
    if tempo < 40 or tempo > 240:
        tempo = 120.0
    step_sec = 60.0 / (tempo * 4)  # duration of one 16th-note in seconds

    # --- 3. Collect all notes from all instruments as (step, dur, track, pitch) ---
    events = []
    for inst in pm.instruments:
        # Map this instrument to one of the 7 track groups
        track_name = program_to_track(inst.program, inst.is_drum)
        track_idx = TRACK_TO_IDX[track_name]
        for note in inst.notes:
            # Convert note start time to grid step (rounded to nearest 16th)
            s = int(round(note.start / step_sec))
            # Convert note duration to grid steps, clamp to [1, MAX_DURATION]
            d = int(round((note.end - note.start) / step_sec))
            d = max(1, min(MAX_DURATION, d))
            if note.pitch < 0 or note.pitch > 127:
                continue  # skip out-of-range pitches
            events.append((s, d, track_idx, int(note.pitch)))

    if not events:
        return None  # file has no usable notes

    # --- 4. Sort by (time, track group, pitch) for a stable, learnable ordering ---
    # Sorting ensures the model sees notes in a consistent order: drums first,
    # then bass, piano, etc. This makes it easier for the model to predict patterns.
    events.sort(key=lambda e: (e[0], e[2], e[3]))

    # --- 5. Emit tokens bar-by-bar ---
    tokens = [START]
    cur_bar = -1   # which bar we're currently emitting
    cur_pos = -1   # last position token emitted within the current bar
    cur_track = -1   # last track token emitted at the current position

    for step, dur, track_idx, pitch in events:
        bar = step // STEPS_PER_BAR  # which bar this note belongs to
        pos = step %  STEPS_PER_BAR  # position within that bar (0-15)
        if bar >= max_bars:
            break  # truncate very long files

        # Emit BAR tokens to advance to the correct bar
        while cur_bar < bar:
            tokens.append(BAR)
            cur_bar += 1
            cur_pos = -1   # reset position context for new bar
            cur_track = -1

        # Emit a POS token only when the position changes within the bar
        # (multiple notes at the same position share one POS token)
        if pos != cur_pos:
            tokens.append(pos_token(pos))
            cur_pos = pos
            cur_track = -1  # reset track context for new position

        # Emit a TRACK token only when the instrument group changes
        # (multiple notes on the same track at the same position share one TRACK token)
        if track_idx != cur_track:
            tokens.append(track_token(track_idx))
            cur_track = track_idx

        # Every note always gets its own PITCH and DUR tokens
        tokens.append(pitch_token(pitch))
        tokens.append(dur_token(dur))

    tokens.append(END)

    # Reject sequences that are too short to be meaningful
    if len(tokens) < 8:
        return None
    return tokens


# ===========================================================================
# DECODE: integer token list -> pretty_midi.PrettyMIDI object
# ===========================================================================
def tokens_to_midi(tokens: list[int], tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """
    Convert a token list (produced by the model) back into a playable MIDI file.

    This is the inverse of midi_to_tokens. The decoder walks through the
    token stream, tracking which bar / position / track it's currently in,
    and builds a Note object whenever it sees a complete PITCH+DUR pair.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    step_sec = 60.0 / (tempo * 4)  # 16th-note duration in seconds

    # --- Create one Instrument object per track group upfront ---
    # We keep them in a dict keyed by name and add notes as we decode.
    instruments: dict[str, pretty_midi.Instrument] = {}
    for name in TRACK_NAMES:
        is_drum = (name == "DRUMS")
        prog = TRACK_DEFAULT_PROGRAM[name]
        inst = pretty_midi.Instrument(program=prog, is_drum=is_drum, name=name)
        instruments[name] = inst

    # --- Decoder state ---
    cur_bar = -1          # current bar index (incremented on each BAR token)
    cur_pos = 0           # current position within the bar (0-15)
    cur_track = None      # current instrument group name (set by TRACK token)
    pending_pitch = None  # pitch seen but not yet paired with a duration

    # --- Walk through tokens and reconstruct notes ---
    for tok in tokens:
        if tok == START or tok == PAD:
            continue  # structural tokens with no musical meaning
        if tok == END:
            break     # stop at the explicit end marker

        if tok == BAR:
            # Start a new bar: reset position and track context
            cur_bar += 1
            cur_pos = 0
            cur_track = None
            pending_pitch = None
            continue

        if is_pos(tok):
            # Move to a new position within the current bar
            cur_pos = pos_value(tok)
            pending_pitch = None
            continue

        if is_track(tok):
            # Switch to a different instrument group
            cur_track = TRACK_NAMES[track_value(tok)]
            pending_pitch = None
            continue

        if is_pitch(tok):
            # Store the pitch; we need the next DUR token to complete the note
            pending_pitch = pitch_value(tok)
            continue

        if is_dur(tok):
            # We have pitch + duration: build the note and add it to the instrument
            if pending_pitch is None or cur_track is None or cur_bar < 0:
                pending_pitch = None
                continue  # incomplete state — skip this token

            dur = dur_value(tok)
            start_step = cur_bar * STEPS_PER_BAR + cur_pos
            start_sec = start_step * step_sec
            end_sec = (start_step + dur) * step_sec

            note = pretty_midi.Note(
                velocity=90,              # fixed velocity (loudness)
                pitch=int(pending_pitch),
                start=start_sec,
                end=end_sec,
            )
            instruments[cur_track].notes.append(note)
            pending_pitch = None  # clear state; next token starts a new note
            continue

    # --- Only attach instruments that actually have notes ---
    # (avoids empty tracks in the output MIDI file)
    for inst in instruments.values():
        if inst.notes:
            pm.instruments.append(inst)

    return pm
