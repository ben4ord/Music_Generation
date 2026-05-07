import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import MusicDataset
from src.model import LSTMModel
from src.preprocess import load_midi_dataset
import os
from torch.utils.data import random_split
from tqdm import tqdm

CHECKPOINT_DIR = "checkpoints"
RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
VOCAB_SIZE = 128 * 128
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Validation loop to evaluate the model on the validation set after each training epoch, and return the average validation loss
def validate(model, dataloader, criterion, device, epoch, epochs):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{epochs} [val]",
            leave=False
        )

        for x, y, genre in progress_bar:
            x, y, genre = x.to(device), y.to(device), genre.to(device)

            logits, _ = model(x, genre)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)

# Load a model checkpoint if it exists, and return the starting epoch and best validation loss for resuming training
def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0, float("inf")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        print(f"Skipping incompatible checkpoint: {checkpoint_path}")
        print(f"Reason: {e}")
        return 0, float("inf")

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("val_loss", float("inf"))
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Resuming after epoch {start_epoch} with best val loss {best_loss:.4f}")

    return start_epoch, best_loss


# save a generated sequence of tokens as a MIDI file, using pretty_midi to convert token sequences back into MIDI format
def save_checkpoint(path, model, optimizer, epoch, val_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)


def train(model, train_loader, val_loader, epochs, lr, device, resume_checkpoint=None):
    print("Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    start_epoch, best_loss = load_checkpoint(model, optimizer, resume_checkpoint, device)

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        total_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{start_epoch + epochs} [train]"
        )

        # Iterate through training batches and perform forward/backward passes
        for x, y, genre in progress_bar:
            x, y, genre = x.to(device), y.to(device), genre.to(device)

            optimizer.zero_grad()

            # Forward pass through the model to get logits for the next token prediction at each time step
            logits, _ = model(x, genre)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation 
        val_loss = validate(model, val_loader, criterion, device, epoch, start_epoch + epochs)

        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
        save_checkpoint(latest_checkpoint_path, model, optimizer, epoch, val_loss)

        # Saving best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss)

            print("Saved new best model.")

        print(f"Epoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")


# Load and preprocess MIDI dataset
token_sequences, genre_ids, genre_to_id = load_midi_dataset("data/midi", min_tokens=129)

dataset = MusicDataset(token_sequences, genre_ids, seq_len=128)

# Split dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Create DataLoaders for training and validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
# Initialize model and start training
# Need to multiply vocab_size by 128 to account for the fact that we are encoding both instrument and pitch into a single token
# So we have 128 possible pitches for each of the 128 possible instruments, resulting in a total vocabulary size of 128 * 128 = 16384
model = LSTMModel(vocab_size=VOCAB_SIZE, genre_count=len(genre_to_id))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Train the model and save checkpoints based on validation loss
train(
    model,
    train_loader,
    val_loader,
    epochs=30,
    lr=3e-4,
    device=device,
    resume_checkpoint=RESUME_CHECKPOINT
)
