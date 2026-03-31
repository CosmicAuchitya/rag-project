# Local Video RAG Project Documentation

## 1. Project Summary | Project ka Summary

This project builds a **production-style local RAG system** that can take documents or a video transcript, convert the content into embeddings, store it in FAISS, retrieve relevant chunks, and answer questions from the retrieved context.

Ye project ek **production-style local RAG system** banata hai jo documents ya video transcript ko process karta hai, embeddings banata hai, FAISS me store karta hai, relevant chunks retrieve karta hai, aur unhi chunks ke context se answer deta hai.

---

## 2. Main Problem | Asli Problem Kya Thi

The user did not want to depend on any paid API or cloud service.

User kisi bhi paid API ya cloud service par depend nahi karna chahte the.

That created a few practical problems:

Isse kuch practical problems aaye:

1. We needed a **fully local answer generation path**.
2. We needed **local transcription** for the MP4 video.
3. The original video is around **16 minutes long**, so direct full processing could be slow on CPU.
4. The machine did not have `ffmpeg` installed globally.
5. Local models can be heavy, so we needed a balanced choice between **speed, memory, and quality**.

---

## 3. Why RAG Here | Yahan RAG Kyun Use Kiya

The goal was not just transcription.

Goal sirf transcription karna nahi tha.

We wanted a system that could:

Hum aisa system chahte the jo:

1. understand the source content,
2. search semantically,
3. answer user questions from relevant context,
4. track the source of the answer.

Video ya long transcript ko directly model me daalne se context limit aur quality issues aa sakte hain. RAG approach me transcript ko chunks me tod kar indexed form me store kiya jata hai, jisse targeted retrieval hota hai aur answer zyada grounded hota hai.

---

## 4. Final Architecture | Final Architecture Kya Hai

Pipeline:

1. Data Loading
2. Data Cleaning
3. Chunking
4. Embedding
5. Vector Storage in FAISS
6. Semantic Retrieval
7. LLM-based Answer Generation
8. Evaluation / Inspection

Video workflow:

1. MP4 video input
2. Sample extraction
3. WAV audio conversion
4. Whisper transcription
5. Transcript saved as text and JSON
6. Transcript indexed into FAISS
7. User query answered with local RAG

---

## 5. Project Structure | Project Structure Samajhna

- [rag_system/config.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\config.py)
  Central config for pipeline settings.

- [rag_system/loaders.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\loaders.py)
  Loads TXT, CSV, and PDF files.

- [rag_system/preprocess.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\preprocess.py)
  Cleans and normalizes text.

- [rag_system/chunking.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\chunking.py)
  Splits documents into overlapping chunks.

- [rag_system/embeddings.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\embeddings.py)
  Generates embeddings and caches them in SQLite.

- [rag_system/vector_store.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\vector_store.py)
  Builds and saves FAISS indexes.

- [rag_system/retriever.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\retriever.py)
  Retrieves top-k relevant chunks.

- [rag_system/generator.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\generator.py)
  Generates answers using extractive mode, OpenAI-compatible mode, or local Hugging Face mode.

- [rag_system/transcription.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\transcription.py)
  Handles local video sample extraction and Whisper transcription.

- [rag_system/video_rag.py](C:\Users\91945\OneDrive\Documents\New project\rag_system\video_rag.py)
  Shared reusable local video-to-RAG workflow.

- [run_local_video_rag.py](C:\Users\91945\OneDrive\Documents\New project\run_local_video_rag.py)
  CLI runner for local video RAG.

- [app/streamlit_app.py](C:\Users\91945\OneDrive\Documents\New project\app\streamlit_app.py)
  Streamlit UI for the same local workflow.

---

## 6. Problems We Faced | Humne Kaun Kaun Si Problems Face Ki

### Problem 1: No API usage allowed
The user explicitly wanted a local setup.

Isliye OpenAI ya kisi hosted API par dependency hataani padi.

### Problem 2: No ffmpeg installed globally
Video ko audio me convert karne ke liye ffmpeg chahiye tha.

System me ffmpeg available nahi tha, isliye `imageio-ffmpeg` use kiya gaya jo Python ke through bundled ffmpeg executable deta hai.

### Problem 3: Long video processing time
16-minute video ko directly process karna CPU par slow ho sakta tha.

Isliye pehle **60-second sample** par workflow verify kiya gaya.

### Problem 4: Local models can be heavy
Bade local models CPU RAM aur time dono consume karte hain.

Isliye humne lighter but useful models choose kiye.

### Problem 5: Library/runtime issues
Environment me dependency visibility aur model task compatibility jaisi issues aaye.

Humne generator ko more stable local seq2seq loading path par shift kiya.

---

## 7. Choices We Made | Humne Kaun Se Choices Kiye

### Choice 1: 60-second sample first
Why:

- Faster validation
- Lower memory use
- Easier debugging

Hindi:
Pehle 1-minute sample lena safe tha kyunki isse poora pipeline jaldi test ho gaya aur full 16-minute run se pehle issues pakad liye.

### Choice 2: `openai/whisper-tiny.en` for transcription
Why:

- Small and lightweight
- Good for quick local testing
- Faster than larger Whisper variants

Tradeoff:
Accuracy `base` ya `small` model se thodi kam ho sakti hai.

### Choice 3: `sentence-transformers/all-MiniLM-L6-v2` for embeddings
Why:

- Fast
- Memory efficient
- Strong enough for semantic retrieval

### Choice 4: FAISS as vector database
Why:

- Fast similarity search
- Lightweight
- Good local deployment option

### Choice 5: `google/flan-t5-small` as local answer model
Why:

- Fully local
- Small enough for CPU use
- Easier to run than larger instruction models

Tradeoff:
Answer quality useful hai, but bahut advanced ya highly detailed nahi hogi.

### Choice 6: Similarity filtering
Why:

- Low-quality irrelevant chunks ko avoid karna
- Better grounding

### Choice 7: Embedding cache in SQLite
Why:

- Repeated runs faster ho jate hain
- Same text ke embeddings dubara compute nahi karne padte

---

## 8. What We Did Step By Step | Humne Step By Step Kya Kiya

1. Existing modular RAG codebase banaya.
2. Loaders, preprocess, chunking, embeddings, FAISS, retrieval, generation modules add kiye.
3. Local generator support add kiya.
4. Video processing ke liye audio extraction utility banayi.
5. Whisper transcription module add kiya.
6. Transcript ko TXT and JSON artifacts me save kiya.
7. Transcript par FAISS index build kiya.
8. Local question answering test ki.
9. Streamlit UI add kiya.
10. Documentation add ki.

---

## 9. Output Artifacts | Kaunse Files Generate Hui

- Sample video clip
- Sample WAV audio
- Transcript TXT
- Transcript JSON metadata
- FAISS index
- Chunk metadata JSON

Important generated paths:

- [artifacts/video_processing](C:\Users\91945\OneDrive\Documents\New project\artifacts\video_processing)
- [data/video_transcripts](C:\Users\91945\OneDrive\Documents\New project\data\video_transcripts)
- [artifacts/faiss_index](C:\Users\91945\OneDrive\Documents\New project\artifacts\faiss_index)

---

## 10. Current Limitations | Abhi System Ki Limitations

1. The local answer model is small, so answers can be short or imperfect.
2. Whisper tiny model is fast but not the most accurate.
3. Full 16-minute video processing will take more time than the 60-second sample.
4. CPU-only setup will be slower than GPU.
5. Better summarization may need a stronger local model.

---

## 11. Recommended Improvements | Aage Kya Improve Kar Sakte Hain

1. Use `openai/whisper-base.en` or `openai/whisper-small.en` for better transcription.
2. Use a stronger local LLM such as a quantized instruct model via Ollama or LM Studio.
3. Add transcript chunk timestamps inside retrieval metadata.
4. Add batch processing for full video segments.
5. Add conversation memory in Streamlit.
6. Add transcript search and source preview UI.

---

## 12. Can This Run On Streamlit? | Kya Ye System Streamlit Par Chal Sakta Hai

Yes, absolutely.

Haan, bilkul chal sakta hai.

We already added a Streamlit app:

- [app/streamlit_app.py](C:\Users\91945\OneDrive\Documents\New project\app\streamlit_app.py)

### How Streamlit fits here | Streamlit ka role

Streamlit ek browser-based UI deta hai jahan user:

1. video ka naam de sakta hai,
2. sample duration choose kar sakta hai,
3. query enter kar sakta hai,
4. transcript dekh sakta hai,
5. final answer aur retrieved chunks dekh sakta hai.

### How to run Streamlit | Streamlit kaise chalana hai

```powershell
streamlit run app/streamlit_app.py
```

If `streamlit` command PATH me na ho, tab:

```powershell
python -m streamlit run app/streamlit_app.py
```

### What the Streamlit app does

1. Reads the local video from project root
2. Extracts sample clip
3. Transcribes locally
4. Builds FAISS index
5. Answers your question
6. Shows transcript, answer, and retrieved chunks in browser

### When Streamlit is useful

- Demo dene ke liye
- Non-technical users ke liye
- Quick experimentation ke liye
- Manual question-answer workflow ke liye

### When CLI is better

- Batch runs
- Debugging
- Automation
- Long processing jobs

---

## 13. Recommended Command Set | Recommended Commands

### CLI run

```powershell
python run_local_video_rag.py
```

### Longer sample

```powershell
python run_local_video_rag.py --duration-seconds 300
```

### Different start offset

```powershell
python run_local_video_rag.py --start-seconds 120 --duration-seconds 180
```

### Streamlit UI

```powershell
python -m streamlit run app/streamlit_app.py
```

---

## 15. OpenAI API Demo Option | Interview Me API Key Kaise Dikhani Hai

Yes, this project now supports an interview-friendly API-key flow too.

Haan, ab is project me API key ka clear demo option bhi diya gaya hai.

### Where to put the key | Key kahan daalni hai

Use this file:

- `.env.example`

Copy it to:

- `.env`

Then fill this line:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Main switch | API mode on kaise karna hai

In `.env`, set:

```env
USE_OPENAI=true
```

Recommended interview/demo values:

```env
VIDEO_EMBEDDING_PROVIDER=openai
VIDEO_EMBEDDING_MODEL=text-embedding-3-small
VIDEO_LLM_PROVIDER=openai
VIDEO_LLM_MODEL=gpt-4o-mini
```

### Which files use the API settings | Kaun si files ye settings padhti hain

- `run_local_video_rag.py`
- `notebooks/rag_step_by_step.py`
- `app/api.py`
- `app/streamlit_app.py`
- `rag_system/video_rag.py`

### What to say in interview | Interview me kaise explain karna hai

You can say:

"System local mode me bhi chal sakta hai aur OpenAI API mode me bhi. API key `.env` me jaati hai, aur providers environment variables se switch hote hain. Isliye same codebase local aur hosted dono modes support karta hai."

Hindi:

"Ye system local mode me bhi chalta hai aur API key ke saath bhi. Key `.env` file me daali jaati hai, aur provider switch env variables se hota hai. Isliye interview me main dono modes dikha sakta hoon."

### Exact demo command | Seedha run kaise karna hai

```powershell
python run_local_video_rag.py
```

Ya Streamlit me:

```powershell
python -m streamlit run app/streamlit_app.py
```

If `.env` me `USE_OPENAI=true` aur valid `OPENAI_API_KEY` hai, to system OpenAI mode use karega.

## 14. Final Conclusion | Final Conclusion

This project now supports:

1. modular RAG,
2. local video transcription,
3. local embeddings,
4. FAISS retrieval,
5. local answer generation,
6. CLI execution,
7. Streamlit UI.

Simple words me:

Ab ye system bina API ke local machine par video ko samajhne aur us par question-answer karne ke liye ready hai. Streamlit par bhi chal sakta hai, aur CLI se bhi.
