import os
import numpy as np
import torch
import librosa
import faiss
from speechbrain.inference.speaker import SpeakerRecognition
from TTS.api import TTS
import warnings

import logging

warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# =========================
#  CONFIG
# =========================
THRESHOLD = 0.5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
GENERATE_AUDIO = False

# =========================
#  LOAD MODELS
# =========================
print("Loading Speaker Model...")
speechbrain_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models"
)

print("Loading XTTS...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

# =========================
#  EMBEDDING FUNCTION
# =========================
def get_embedding(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)
    signal = torch.tensor(signal).unsqueeze(0)

    with torch.no_grad():
        embedding = speechbrain_model.encode_batch(signal)

    emb = embedding.squeeze().cpu().numpy()

    # Normalize for cosine similarity
    emb = emb / np.linalg.norm(emb)

    return emb.astype("float32")

# =========================
#  LOAD DATASETS
# =========================
def load_audio_folder(folder_path, is_protected):
    embeddings = []
    metadata = []

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            path = os.path.join(folder_path, file)

            print(f"Processing: {file}")
            emb = get_embedding(path)

            embeddings.append(emb)
            metadata.append({
                "filename": file,
                "protected": is_protected,
                "path": path
            })

    return embeddings, metadata

print("\nLoading datasets...")

protected_embs, protected_meta = load_audio_folder("protected", True)
normal_embs, normal_meta = load_audio_folder("normal", False)

# Combine
embeddings = np.array(protected_embs + normal_embs).astype("float32")
metadata = protected_meta + normal_meta

print(f"\nTotal embeddings: {len(embeddings)}")

# =========================
#  BUILD FAISS INDEX
# =========================
dimension = embeddings.shape[1]

# Inner Product index (for cosine similarity since normalized)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(" FAISS index built")

# =========================
# 🔍 CHECK FUNCTION (FAISS)
# =========================
def check_audio(test_audio_path, top_k=5):
    test_emb = get_embedding(test_audio_path).reshape(1, -1)

    # Search top-k similar embeddings
    similarities, indices = index.search(test_emb, top_k)

    max_protected_sim = 0

    print("\n FAISS Similarity Results:")

    for sim, idx in zip(similarities[0], indices[0]):
        speaker_info = metadata[idx]

        print(f"Matched: {speaker_info['filename']}")
        print(f"Similarity: {sim:.4f}")
        print(f"Protected: {speaker_info['protected']}")
        print("------")

        if speaker_info["protected"]:
            max_protected_sim = max(max_protected_sim, sim)

    print(f"\nMax protected similarity: {max_protected_sim:.4f}")

    if max_protected_sim >= THRESHOLD:
        print(" BLOCKED")
        return "BLOCK"
    else:
        print(" ALLOWED")
        return "ALLOW"

# =========================
# 🔊 GENERATION FUNCTION
# =========================
def generate_if_allowed(test_audio, text):
    decision = check_audio(test_audio)

    if decision == "BLOCK":
        print(" Generation skipped")
        return

    print("\n🎤 Generating speech...")

    output_file = "output.wav"

    with torch.no_grad():
        tts.tts_to_file(
            text=text,
            speaker_wav=test_audio,
            language="en",
            file_path=output_file
        )

    print(f" Generated: {output_file}")

def process_input_folder(folder_path, text, generate_audio=False):
    print(f"\n📂 Processing folder: {folder_path}")

    if not os.path.exists(folder_path):
        print(" Folder not found")
        return

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            print("\n==============================")
            print(f"🎧 Processing: {file}")
            print("==============================")

            decision = check_audio(file_path)

            if decision == "BLOCK":
                print(" Skipped (Protected Voice)")
                continue

            print("✅ ALLOWED")

            # 🔥 CONDITIONAL GENERATION
            if generate_audio:
                print("🎤 Generating speech...")

                output_file = f"output_{os.path.splitext(file)[0]}.wav"

                with torch.no_grad():
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=file_path,
                        language="en",
                        file_path=output_file
                    )

                print(f" Generated: {output_file}")
            else:
                print(" Generation skipped (disabled)")

# =========================
#  METRICS FUNCTION
# =========================
def get_ground_truth(filename):
    filename = filename.lower()

    # Protected speakers
    if any(name in filename for name in ["amitabh", "salman", "srk", "wanted"]):
        return "BLOCK"
    else:
        return "ALLOW"


def evaluate_model(folder_path, threshold):
    print(f"\n Evaluating Model (Threshold = {threshold})")

    TP = TN = FP = FN = 0

    for file in os.listdir(folder_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, file)

        print("\n------------------------------")
        print(f" File: {file}")

        # Prediction using your existing function
        pred = check_audio(file_path)

        # Ground truth
        actual = get_ground_truth(file)

        print(f"Prediction: {pred} | Actual: {actual}")

        # Confusion matrix update
        if pred == "BLOCK" and actual == "BLOCK":
            TP += 1
        elif pred == "ALLOW" and actual == "ALLOW":
            TN += 1
        elif pred == "BLOCK" and actual == "ALLOW":
            FP += 1
        elif pred == "ALLOW" and actual == "BLOCK":
            FN += 1

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print("\n==============================")
    print(" FINAL METRICS")
    print("==============================")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")


# =========================
#  TEST
# =========================
if __name__ == "__main__":

    # Run pipeline (existing)
    process_input_folder(
        folder_path="input_voice",
        text="This is a privacy aware voice cloning system.",
        generate_audio=GENERATE_AUDIO
    )

    #  NEW: Evaluate for threshold = 0.5
    evaluate_model("input_voice", threshold=THRESHOLD)

    #  OPTIONAL: Evaluate for 0.4 (comparison)
    THRESHOLD = 0.4
    evaluate_model("input_voice", threshold=THRESHOLD)
