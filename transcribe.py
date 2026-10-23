
import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Configuration
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Load the model once
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


def validate_audio(audio_path: Path) -> None:
    """Validate the audio file before processing."""

    if not audio_path.exists() or not audio_path.is_file():
        raise ValueError("Audio file does not exist.")

    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if audio_path.stat().st_size == 0:
        raise ValueError("Audio file is empty.")

    if audio_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("Audio file exceeds the 100 MB limit.")


def clean_text(text: str) -> str:
    """Normalize whitespace in the transcript."""
    return re.sub(r"\s+", " ", text).strip()


def transcribe_audio(audio_path: Path) -> dict:
    """Transcribe audio and return structured JSON-compatible data."""

    validate_audio(audio_path)

    start_time = time.perf_counter()
    logger.info("Transcription started: %s", audio_path.name)

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True
    )

    transcript_parts = []
    timestamped_segments = []

    # Consume the generator to complete transcription
    for segment in segments:
        text = clean_text(segment.text)

        if text:
            transcript_parts.append(text)

            timestamped_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })

    transcript = clean_text(" ".join(transcript_parts))
    processing_time = round(time.perf_counter() - start_time, 2)

    result = {
        "status": "success",
        "filename": audio_path.name,
        "language": info.language,
        "language_probability": round(
            info.language_probability, 4
        ),
        "transcript": transcript,
        "segments": timestamped_segments,
        "processing_time_seconds": processing_time,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    logger.info("Transcription completed in %.2f seconds", processing_time)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audio-to-text transcription pipeline"
    )

    parser.add_argument(
        "audio",
        help="Path to the input WAV or MP3 file"
    )

    parser.add_argument(
        "--output",
        default="output/transcription.json",
        help="Path to save the JSON output"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_path = Path(args.output)

    try:
        result = transcribe_audio(audio_path)

        # Create output directory if necessary
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save transcription as JSON
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("Output saved to %s", output_path)

    except (ValueError, OSError) as error:
        logger.error("Input/output error: %s", error)

        print(json.dumps({
            "status": "error",
            "message": str(error)
        }, indent=2), file=sys.stderr)

        sys.exit(1)

    except Exception:
        logger.exception("Unexpected transcription error")

        print(json.dumps({
            "status": "error",
            "message": "Transcription failed. Check logs for details."
        }, indent=2), file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()
