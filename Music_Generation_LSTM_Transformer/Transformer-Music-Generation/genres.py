"""
Every other module (tokenizer, pipeline, model, train, generate) imports
from here so that adding or removing a genre only requires editing this one file.
"""

# Absolute path to the folder that holds one sub-folder per genre,
# each containing .mid files (e.g. Genres/Jazz/song.mid).
GENRES_ROOT = r"C:\CS-421\Final_Project\Genres"

# Sorted list of all 13 genres in the dataset.
# Sorting guarantees that the integer index assigned to each genre is stable
# across runs — if you add a genre later, existing checkpoints still decode
# the old genres correctly because their indices don't shift.
GENRES = sorted([
    "Blues", "Classical",
    "Country", "Disco", "Electronic",
    "Folk", "Hip_Hop",
    "Jazz", "Metal", "Pop",
    "R_and_B", "Reggae", "Rock", 
])

# Total number of genres — used by the model to size the genre embedding table.
N_GENRES = len(GENRES)

# Dict: genre string -> integer index  (e.g. "Blues" -> 0)
# Used during training to look up the label for each sequence.
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

# Dict: integer index -> genre string  (e.g. 0 -> "Blues")
# Used during evaluation / generation to convert a predicted index back to a name.
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}
