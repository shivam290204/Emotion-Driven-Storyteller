# Emotion-Driven Storyteller

Emotion-Driven Storyteller is a privacy-first Streamlit experience that blends facial, vocal, and textual emotion analysis with adaptive storytelling, recommendations, and analytics. The app fuses multi-modal signals, personalizes every narrative, and keeps sensitive data encrypted at rest.
<img width="488" height="686" alt="Screenshot 2025-10-27 221924" src="https://github.com/user-attachments/assets/53327c2e-a9b0-4b30-a9f6-1f002199f0a1" />
<img width="628" height="771" alt="Screenshot 2025-10-27 221840" src="https://github.com/user-attachments/assets/2b87ff97-f192-4185-9c4c-28e8d44f44b6" />
<img width="1889" height="819" alt="Screenshot 2025-10-27 221814" src="https://github.com/user-attachments/assets/7f7e69e3-63d0-415f-bf8e-ca6fcc1a85c0" />
<img width="1915" height="861" alt="Screenshot 2025-10-27 221711" src="https://github.com/user-attachments/assets/7708556f-911d-40e8-8351-190a83b0d057" />
<img width="1917" height="859" alt="Screenshot 2025-10-27 221632" src="https://github.com/user-attachments/assets/c5dfa967-9f2e-4bde-beeb-a10a1c3f6941" />


## ✨ Features
- **Multi-modal emotion detection** using DeepFace (webcam), Hugging Face audio classifiers (voice), and transformer-based text analysis.
- **AI-assisted storytelling** combining curated templates, cultural personalization, and optional Hugging Face text generation.
- **Emotion fusion dashboard** displaying dominant moods, confidence scores, and trend forecasting.
- **Adaptive wellness layer** with mood-aware recommendations and an emotion-sensitive mini-game.
- **Privacy controls** including AES-GCM encrypted logs, profile protection, and one-click purge utilities.
- **Voice narration** via pyttsx3 with adjustable rate, volume, and voice preferences.

## 🚀 Quick Start (PowerShell)
```powershell
# 1. Clone the repository and enter the project folder
# git clone https://github.com/<your-username>/emotion-storyteller.git
cd "Emotion_storyTeller"

# 2. Create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

Grant webcam and microphone permissions when prompted. Press `q` to exit the live webcam scan.

## 🧩 Project Structure
```
app.py                 # Streamlit UI & orchestration
emotion_detector.py    # DeepFace webcam scanner and helpers
voice_detector.py      # Audio capture & Hugging Face classifier
text_analyzer.py       # Transformer-based text emotion analysis
story_generator.py     # Story templates, personalization, TTS, AI generation
recommendations.py     # Mood-based activity recommendations
game_engine.py         # Emotion-adaptive mini-game
security_utils.py      # AES-GCM encryption utilities
emotion_forecaster.py  # Short-term emotion forecasting insights
analytics_logger.py    # Encrypted SQLite logging and analytics helpers
```

## ⚙️ Configuration Highlights
- Tune **confidence thresholds**, **scan duration**, and **story strategies** directly from the sidebar.
- Switch between **template** and **AI-generated** stories on demand.
- Manage **user profiles** with optional cultural context for deeper personalization.
- Purge encrypted logs or profiles at any time via the **Data Privacy** panel.

## 🛠 Troubleshooting
- **Webcam not detected:** Close other applications using the camera and allow Streamlit access in browser prompts.
- **Audio capture issues:** Ensure `sounddevice` has permission to use the default microphone.
- **Model downloads slow:** The first run fetches DeepFace weights and Hugging Face models; keep the console open until complete.
- **No narration audio:** Confirm pyttsx3 is using the desired output device or disable narration.

## 📄 License
This project is provided for educational and demonstrative use. Review the licenses for any bundled datasets or third-party models you employ.

## 🙌 Acknowledgements
- [DeepFace](https://github.com/serengil/deepface)
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
