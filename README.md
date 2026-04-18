# SU_project
#  Privacy-Aware Voice Cloning Pipeline (XTTS + Speaker Verification)

This project implements a **privacy-preserving voice cloning system** where:

* If input voice matches a **protected speaker (≥ 0.4 similarity)** →  BLOCK
* Otherwise →  Generate cloned speech using XTTS

---

#  1. Environment Setup (Local Machine)

##  Step 1: Create Virtual Environment

```bash
conda create -n voice_pipeline python=3.10 -y
conda activate voice_pipeline
```

---

##  Step 2: Install Dependencies (IMPORTANT: exact versions)

```bash
pip install torch==2.4.0 torchaudio==2.4.0
pip install transformers==4.38.2
pip install TTS==0.22.0
pip install speechbrain faiss-cpu librosa soundfile scikit-learn
pip install setuptools==68.2.2
```

---

## ✅ Step 3: Install FFmpeg (for audio conversion)

```bash
brew install ffmpeg
```

---

# 🎧 2. Prepare Dataset

## Folder Structure

```
project/
│
├── protected/        # Protected voices (blocked)
│   ├── p1.wav
│   ├── p2.wav
│
├── normal/           # Allowed voices
│   ├── n1.wav
│   ├── n2.wav
│
├── clean_speaker.wav # Test input
├── AudioGenPipeline.py
```

---

## 🔄 Convert `.m4a` → `.wav` (if needed)

```bash
for f in *.m4a; do 
  ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.m4a}.wav"
done
```

---

#  3. Pipeline Logic

1. Extract speaker embeddings using **SpeechBrain ECAPA**
2. Compare input audio with:

   * Protected set
   * Normal set
3. Compute **cosine similarity**
4. Decision:

   * ≥ 0.4 →  BLOCK
   * < 0.4 →  ALLOW
5. If allowed → generate speech using XTTS

---

#  4. Run the Pipeline

```bash
python AudioGenPipeline.py
```

---

#  5. Test Cases

##  Case 1: Protected Voice

**Output:**

```
Max protected similarity: 0.9829
 BLOCKED
 Generation skipped
```

 Voice matched protected dataset → generation blocked

---

##  Case 2: Normal Voice

**Output:**

```
Max protected similarity: 0.1077
ALLOWED
 Generating speech...
 Generated: output4.wav
```

 Voice is safe → speech generated

---

# ⚙️ 6. Configuration

```python
THRESHOLD = 0.4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

---

#  Notes

* Use **16kHz mono WAV** audio for best results
* Keep speaker audio **3–6 seconds**
* Ignore `FutureWarning` messages (safe)
* First run will download models (~2GB)

---

#  Final Output

* `output4.wav` → Generated speech (if allowed)
* Console logs → Similarity + decision

---

#  Summary

✔ Privacy-aware voice cloning
✔ Prevents misuse of protected voices
✔ Works locally on Mac (MPS/CPU)https://github.com/M22AIE243/SU_project/blob/main/README.md
✔ Fully reproducible pipeline

<img width="1004" height="1352" alt="image" src="https://github.com/user-attachments/assets/a14257ec-fcfe-4bcb-b9a5-4a029aaa2392" />

# OUTPUT Case 1: When generation of Audio not allowed
#(xtts) kumarroushan@kumars-MacBook-Air genai % python AudioGenPipeline.py 
Loading Speaker Model...
hyperparams.yaml: 1.92kB [00:00, 2.08MB/s]
embedding_model.ckpt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83.3M/83.3M [00:04<00:00, 18.9MB/s]
mean_var_norm_emb.ckpt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.92k/1.92k [00:00<00:00, 8.25MB/s]
classifier.ckpt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.53M/5.53M [00:00<00:00, 9.15MB/s]
label_encoder.txt: 129kB [00:00, 21.5MB/s]

Loading XTTS...
 > tts_models/multilingual/multi-dataset/xtts_v2 is already downloaded.
 > Using model: xtts

Loading datasets...
Max protected similarity: 0.9829
🚫 BLOCKED
❌ Generation skipped

# OUTPUT Case 2: When generation of Audio is allowed

#(xtts) kumarroushan@kumars-MacBook-Air genai % python AudioGenPipeline.py
Loading Speaker Model...

Loading XTTS...
 > tts_models/multilingual/multi-dataset/xtts_v2 is already downloaded.
 > Using model: xtts

Loading datasets...
Max protected similarity: 0.1077
✅ ALLOWED
🎤 Generating speech...
 > Text splitted to sentences.
['This is a privacy aware voice cloning system.']
 > Processing time: 12.82853388786316
 > Real-time factor: 3.174595666046223
✅ Generated: output4.wav



#  Future Improvements

* Replace loop with **FAISS ANN search**
* Improve voice quality (denoise + trim)
* Add UI (Streamlit demo)
* Dynamic threshold tuning

---



