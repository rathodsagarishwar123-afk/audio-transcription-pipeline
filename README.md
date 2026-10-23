# Audio Transcription Pipeline

A Python-based audio transcription tool built with Faster-Whisper.

## Features
- Speech-to-text transcription
- Supports WAV, MP3, M4A, FLAC, and OGG
- Automatic language detection
- Segment-level timestamps
- JSON output
- Input validation and error handling
- Voice Activity Detection (VAD)

## Requirements
- Python 3.10 or a compatible version
- FFmpeg or compatible audio decoding support as required by the environment

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python transcribe.py sample_audio.mp3
```

The transcription output is saved as JSON in the `output` directory.

## Technology
- Python
- Faster-Whisper
- Whisper speech recognition model
