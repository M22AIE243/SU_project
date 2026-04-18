import os
import numpy as np
import torch
import librosa
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.inference.speaker import SpeakerRecognition
from TTS.api import TTS

# =========================
# 🔧 CONFIG
# =========================
THRESHOLD = 0.4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# =========================
# 🎤 LOAD MODELS
# =========================
print("Loading Speaker Model...")
speechbrain_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models"
)

print("Loading XTTS...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

# =========================
# 🎧 EMBEDDING FUNCTION
# =========================
def get_embedding(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)
    signal = torch.tensor(signal).unsqueeze(0)

    with torch.no_grad():
        embedding = speechbrain_model.encode_batch(signal)

    emb = embedding.squeeze().cpu().numpy()
    emb = emb / np.linalg.norm(emb)  # normalize

    return emb

# =========================
# 📁 LOAD DATASETS
# =========================
def load_audio_folder(folder_path, is_protected):
    embeddings = []
    metadata = []

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            path = os.path.join(folder_path, file)

            emb = get_embedding(path)

            embeddings.append(emb)
            metadata.append({
                "filename": file,
                "protected": is_protected,
                "path": path
            })

    return embeddings, metadata

print("Loading datasets...")

protected_embs, protected_meta = load_audio_folder("protected", True)
normal_embs, normal_meta = load_audio_folder("normal", False)

embeddings = np.array(protected_embs + normal_embs).astype("float32")
metadata = protected_meta + normal_meta

# =========================
# 🔍 CHECK FUNCTION
# =========================
def check_audio(test_audio_path):
    test_emb = get_embedding(test_audio_path).reshape(1, -1)

    max_protected_sim = 0

    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(test_emb, emb.reshape(1, -1))[0][0]

        if metadata[i]["protected"]:
            max_protected_sim = max(max_protected_sim, sim)

    print(f"Max protected similarity: {max_protected_sim:.4f}")

    if max_protected_sim >= THRESHOLD:
        print("🚫 BLOCKED")
        return "BLOCK"
    else:
        print("✅ ALLOWED")
        return "ALLOW"

# =========================
# 🔊 GENERATION FUNCTION
# =========================
def generate_if_allowed(test_audio, text):
    decision = check_audio(test_audio)

    if decision == "BLOCK":
        print("❌ Generation skipped")
        return

    print("🎤 Generating speech...")

    output_file = "output4.wav"

    with torch.no_grad():
        tts.tts_to_file(
            text=text,
            speaker_wav=test_audio,
            language="en",
            file_path=output_file
        )

    print(f"✅ Generated: {output_file}")

# =========================
# 🚀 TEST
# =========================
generate_if_allowed(
    #test_audio="clean_speaker.wav",
    test_audio="Audio3.wav",
    text="This is a privacy aware voice cloning system."
)