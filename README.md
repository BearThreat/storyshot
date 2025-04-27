```markdown
# StoryShot ✨ - AI Transcript-to-Short-Movie Generator

StoryShot is a command-line application designed to automate the creation of short video sequences from text transcripts. It leverages a suite of cutting-edge AI models for text-to-speech, text-to-image, and image-to-video generation, aiming to produce a visual storyboard-like movie synced with the transcript's audio.

The project heavily utilizes:
*   **Large Language Models (LLMs):** Accessed via OpenRouter for tasks like generating descriptive image prompts from text.
*   **Text-to-Image:** OpenAI's `gpt-image-1` (GPT-4o's image model) known for excellent prompt adherence.
*   **Image-to-Video:** Primarily Kling via Fal.ai (`fal-ai/kling-video/v1.6`), potentially others like `fal-ai/wan-i2v`.
*   **Text-to-Speech:** Fal.ai's TTS service (`fal-ai/playai/tts/v3`).

**Note:** Running the full pipeline can be **time-consuming and expensive** due to the costs associated with the image and video generation APIs. The built-in caching helps mitigate this significantly on subsequent runs.

## Key Features

*   **Automated Pipeline:** Converts raw text transcripts into basic video sequences.
*   **Sentence-Based Processing:** Breaks down the transcript into sentences, processing each one individually.
*   **AI-Powered Generation:** Uses state-of-the-art models for audio, image prompts, images, and video clips.
*   **Modular Structure:** Code is organized into distinct Python modules for each major processing step (`text2speech_.py`, `sentence2storyboard_.py`, etc.).
*   **Robust Caching:** Saves intermediate results (audio, prompts, images, videos) to avoid redundant processing and API calls, significantly reducing cost and time on re-runs.
*   **Cost & Timing Metrics:** Tracks estimated costs and execution times for each API call and the overall run, aiding in performance analysis and optimization.
*   **Command-Line Interface:** Easy to run with simple commands and arguments.
*   **Asynchronous Operations:** Leverages `asyncio` for concurrent processing of independent tasks (like TTS or image generation for different sentences/prompts).

## Technology Stack

*   **Core:** Python 3.10+
*   **Text Processing:** NLTK (for sentence tokenization)
*   **AI APIs:**
    *   OpenAI (`openai`, `tiktoken`)
    *   Fal.ai (`fal-client`, `aiohttp`)
    *   OpenRouter (`httpx`)
*   **Video Assembly:** FFmpeg (command-line tool)
*   **Configuration:** `python-dotenv`
*   **Utilities:** `Pillow` (for image handling), `watchdog` (for `sprompt.py`)

## Workflow

The main script (`storyshot_.py`) orchestrates the following steps:

1.  **Load Transcript:** Reads the input `.txt` file.
2.  **Sentence Tokenization:** Splits the transcript into sentences using NLTK.
3.  **Cache Load/Initialization:** Loads existing results from `cache_data/cache.json`. Normalizes sentences (lowercase, trimmed) to use as keys.
4.  **Text-to-Speech (`text2speech_.py`):**
    *   For each *new* or *force-regenerated* sentence:
    *   Calls Fal.ai TTS API to generate an audio clip (.mp3).
    *   Stores audio path, duration, cost, and time in the cache entry.
5.  **Storyboard Prompt Generation (`sentence2storyboard_.py`):**
    *   For each sentence with *successful* TTS:
    *   Calculates the number of required image "shots" (~2 seconds each) based on audio duration.
    *   Calls an LLM via OpenRouter Engine (`openrouter_engine_.py`) to generate a list of descriptive image prompts for these shots.
    *   Stores the prompt list, cost, time, and token info in the cache entry.
6.  **Prompt-to-Image Generation (`prompt2image_openai_.py`):**
    *   For each *new* or *force-regenerated* prompt generated in the previous step:
    *   Calls OpenAI's `gpt-image-1` API to generate an image (.png).
    *   Stores image path, cost, time, revised prompt, and token info in the cache entry (nested under the corresponding prompt).
7.  **Image-to-Video Generation (`image2video_.py`):**
    *   For each *successful* image generation:
    *   Calls Fal.ai Image-to-Video API (e.g., Kling) to generate a short video clip (~2 seconds) based on the image and original prompt.
    *   Stores video path, cost, and time in the cache entry (nested under the corresponding image).
8.  **Video Assembly (`ffmpeg_assembler_.py`):**
    *   For each sentence with *successful* video clips:
    *   Uses FFmpeg to concatenate the short video clips for that sentence sequentially.
    *   Overlays the original sentence audio (from step 4) onto the concatenated video, trimming to the shorter duration. Creates a final clip per sentence.
    *   Concatenates all the final sentence clips into the final movie file (`storyshot_output.mp4`).
9.  **Cache Save:** Saves the updated cache dictionary to `cache_data/cache.json`.
10. **Summary:** Logs a detailed summary of the run, including costs and timings for each stage.

## Project Structure


.
├── cache_data/               # Directory for storing cache metadata
│   └── cache.json            # (Generated) Cache metadata file
├── temp_files/               # Directory for storing generated media files (persistent cache)
│   ├── audio/                # Generated audio files
│   ├── images/               # Generated image files
│   ├── videos/               # Generated video files
│   ├── ffmpeg_concat_lists/  # Temporary lists for FFmpeg concatenation
│   └── final_clips/          # Intermediate assembled clips per sentence
├── storyshot_.py             # Main orchestrator script
├── text2speech_.py           # Handles Text-to-Speech via Fal.ai
├── openrouter_engine_.py     # Client for interacting with OpenRouter LLMs
├── sentence2storyboard_.py   # Generates image prompts using an LLM
├── prompt2image_openai_.py   # Generates images via OpenAI
├── image2video_.py           # Generates video clips via Fal.ai
├── ffmpeg_assembler_.py      # Assembles video clips and audio using FFmpeg
├── sprompt.py                # Dev utility server to concatenate files for LLM input
├── project_superprompt.md    # The high-level design document/prompt
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env                      # (Required, not committed) API keys and config (see below)
└── .gitignore                # Git ignore file


## Setup and Installation

**1. Prerequisites:**

*   **Python:** Version 3.10 or higher recommended.
*   **Pip:** Python package installer.
*   **Git:** For cloning the repository.
*   **FFmpeg:** **Required.** Must be installed and accessible in your system's PATH.
    *   On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
    *   On macOS (using Homebrew): `brew install ffmpeg`
    *   On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` directory to your system's PATH environment variable.
    *   Verify installation by running `ffmpeg -version` in your terminal.
*   **NLTK Data:** The script requires the 'punkt' tokenizer data. It will attempt to download it automatically on the first run if missing.

**2. Clone the Repository:**

```bash
git clone https://github.com/BearThreat/storyshot.git
cd storyshot
```

**3. Create a Virtual Environment (Recommended):**

```bash
python -m venv .venv
# Activate the environment:
# On Windows (cmd.exe):
#   .\.venv\Scripts\activate.bat
# On Windows (PowerShell):
#   .\.venv\Scripts\Activate.ps1
# On macOS/Linux:
#   source .venv/bin/activate
```

**4. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**5. Configure API Keys (.env file):**

*   Create a file named `.env` in the root directory of the project.
*   **Do not commit this file to Git.** Add `.env` to your `.gitignore` file if it's not already there.
*   Add your API keys and any desired model overrides to the `.env` file. Use the following format:

    ```env
    # Required API Keys
    OPENAI_KEY="sk-..."
    FAL_KEY="key_id:key_secret" # Note format: key_id:key_secret
    OPENROUTER_KEY="sk-or-..."

    # Optional Model Overrides (Defaults are shown)
    # FALAI_TTS_MODEL="fal-ai/playai/tts/v3"
    # FALAI_TTS_VOICE="Jennifer (English (US)/American)" # Ensure voice is valid for the model
    # OPENAI_IMAGE_MODEL="gpt-image-1"
    # FALAI_VIDEO_MODEL="fal-ai/kling-video/v1.6/standard/image-to-video" # Or "fal-ai/wan-i2v"
    ```

## Usage

Run the main orchestrator script from your terminal, providing the path to your transcript file:

```bash
python storyshot_.py path/to/your_transcript.txt [options]
```

**Arguments:**

*   `transcript_file` (Required): Path to the input transcript text file.
*   `-f`, `--force-regenerate` (Optional): Ignores the cache and re-runs all generation steps for all sentences.
*   `--skip-images` (Optional): Skips the prompt-to-image generation stage (and consequently, video generation and assembly).
*   `--skip-videos` (Optional): Skips the image-to-video generation stage (and consequently, assembly).
*   `--skip-assembly` (Optional): Skips the final FFmpeg video/audio assembly stage.

**Example:**

```bash
python storyshot_.py sample_transcript.txt --force-regenerate
```

**Output:**

*   Generated media files will be stored in the `temp_files/` subdirectories.
*   Cache metadata will be stored in `cache_data/cache.json`.
*   The final assembled video (if assembly is not skipped) will be saved as `storyshot_output.mp4` in the project's root directory.
*   Detailed logs, including cost and timing summaries, will be printed to the console.

## Development & Testing

*   **Modular Testing:** Each `_*.py` script is designed with an `if __name__ == "__main__":` block containing test logic for that specific module. You can run these tests individually:
    ```bash
    python text2speech_.py
    python sentence2storyboard_.py
    # etc.
    ```
*   **Superprompt Concatenation:** The `sprompt.py` script runs a `watchdog` server. When changes are detected in any `.py` or `.md` file specified, it concatenates their contents into `project_superprompt_output.md`. This is useful for providing updated context to an LLM during development. Run it via `python sprompt.py`.

## Caching

*   **Purpose:** To save time and API costs on subsequent runs.
*   **Metadata:** `cache_data/cache.json` stores a dictionary where keys are normalized sentences. Each entry holds the results (paths, durations, costs, etc.) for TTS, storyboard prompts, image generation (nested list), and video generation (nested within image outputs).
*   **Assets:** Large media files (audio, images, videos) are stored in the `temp_files/` directory. The cache file links to these assets.
*   **Persistence:** Both the cache file and the temp files are persistent. To clear the cache, delete `cache_data/cache.json` and the contents of `temp_files/`.

## Cost and Performance

*   **API Costs:** Be aware that generating images (especially with `gpt-image-1`) and videos (with models like Kling) can be **expensive**. Monitor the cost summaries printed at the end of each run.
*   **Execution Time:** Generation steps, particularly image and video, can take a significant amount of time (minutes per item). Caching is crucial for iterative development.
*   **Rate Limits:** APIs have rate limits (e.g., `gpt-image-1` is ~5 images/minute). The `prompt2image_openai_.py` module includes basic handling and the orchestrator runs tasks concurrently, but very large transcripts might hit limits.

## Future Work / Roadmap

*   **Visual Cohesion:** Improve prompt engineering in `sentence2storyboard_.py` to generate sequences that maintain better visual consistency across shots within a sentence, and potentially across sentences. This might involve passing previous prompts or even image feedback to the LLM.
*   **More Model Options:** Easily configure and switch between different LLM, image, video, and TTS models.
*   **Advanced FFmpeg:** More sophisticated editing, transitions, or effects during assembly.
*   **Error Recovery:** More granular error handling and potential recovery/retry logic within the orchestrator.
*   **User Interface:** Develop a web or desktop UI instead of being purely command-line based.
*   **Voice Cloning (TTS):** Integrate TTS models that support voice cloning.

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss bugs or feature requests, or submit a pull request.

## License

```
MIT License

Copyright (c) [Year] [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```