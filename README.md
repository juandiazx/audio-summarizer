# Audio Summarizer Pro

A Python application that transcribes audio files and generates text summaries using AI-powered models.

## Features

- **Audio Transcription:** Converts audio files (e.g., MP3, WAV) into text using OpenAI's Whisper model.
- **Text Summarization:** Summarizes the transcribed text using a pre-trained Pegasus model from Hugging Face Transformers.
- **Customizable Summary Length:** Allows users to control the level of detail in the summary.
- **Gradio Interface:** Provides a user-friendly web interface for uploading audio and viewing results.

## Setup and Installation

1.  **Clone the repository (if applicable):**

    ```bash
    # git clone <repository-url>
    # cd audio-summarizer
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    You may also need to install `ffmpeg` if not already present:
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```

## Usage

1.  **Run the application:**
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to the URL provided by Gradio (usually `http://127.0.0.1:7860`).
3.  Upload an audio file.
4.  Adjust the "Summary Level" slider to control the desired conciseness.
5.  Click "Process Audio".
6.  View the transcription, summary, and processing statistics.

## Technologies Used

- Python
- Gradio
- OpenAI Whisper
- Hugging Face Transformers (Pegasus)
- PyTorch
