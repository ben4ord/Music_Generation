import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from Transformer_Music.generate import generate_midi, GENRES

app = Flask(__name__)

MIDI_OUT = os.path.join(app.static_folder, "generated")
os.makedirs(MIDI_OUT, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html", genres=GENRES)


@app.route("/generate", methods=["POST"])
def generate():
    genre = request.json.get("genre")
    if genre not in GENRES:
        return jsonify({"error": f"Unknown genre: {genre}"}), 400

    filename = f"{genre}_{uuid.uuid4().hex[:8]}.mid"
    out_path = os.path.join(MIDI_OUT, filename)
    generate_midi(genre, out_path)

    return jsonify({"file": filename})


@app.route("/midi/<filename>")
def midi(filename):
    return send_from_directory(MIDI_OUT, filename, mimetype="audio/midi")


if __name__ == "__main__":
    app.run(debug=False, port=8080)
