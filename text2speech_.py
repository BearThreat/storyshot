# File: text2speech_.py
"""
Text-to-Speech Module (text2speech_.py)

Handles converting text sentences to speech audio files using Fal.ai's TTS API.
Designed with async readiness and includes estimated cost and inference timing tracking.

This module is part of the 'storyshot' project, aiming to create short movies
from transcripts. It is called by `storyshot_.py` to generate audio for each sentence.

API Info (Fal.ai):
- Website: https://fal.ai/
- Authentication: Uses API Key (key_id:key_secret format). **Requires FAL_KEY environment variable to be set.**
  `fal-client` library picks this up automatically.
- Client Library: `fal-client` (pip install fal-client)
- Target Model (Configurable): Uses model specified by FALAI_TTS_MODEL env var,
  defaulting to "fal-ai/playai/tts/v3".
- Key Input Parameters for the Fal Model:
    - "input": (Required) The text sentence to synthesize.
    - "voice": (Required) The voice name/ID. *Must be valid for the specific Fal model being used.*
               Configured via FALAI_TTS_VOICE env var, defaulting to "Jennifer (English (US)/American)".
               **Note: There is no known standard Fal API endpoint to list available voices dynamically for this model.**
    - "response_format": Defaults to "url", which is expected by this script.
- Key Output (from API): A dictionary containing an "audio" object with:
    - "url": URL to the generated audio file (e.g., MP3).
    - "duration": Duration of the audio in seconds. Used for cost calculation.
    - Other metadata like "content_type", "file_size".

Workflow:
1. Receives sentence text and output path from the orchestrator (`storyshot_.py`).
2. Submits the TTS request asynchronously to Fal.ai using `fal_client.submit_async`.
3. Waits for the Fal.ai processing to complete using `await handler.get()`. Records this wait time.
4. Parses the response to get the audio URL and duration.
5. Calculates the estimated cost based on the audio duration ($0.03/min).
6. Downloads the audio file from the URL asynchronously using `aiohttp`. Records download time.
7. Saves the downloaded audio to the specified path (using standard sync file I/O).
8. Returns a dictionary containing the audio path, duration, estimated cost, and inference time.

Key Features & Outputs:
- Async Ready: Uses `async`/`await` and `aiohttp` for non-blocking I/O where practical.
- Cost Tracking: Calculates estimated cost per request based on $0.03/min rate.
  Maintains a running total cost within the class instance.
- Timing Tracking: Measures and returns the time spent waiting for the Fal.ai inference to complete.
- Result Dictionary: The `process_sentence` method returns a dictionary on success:
    {
        'audio_path': str,          # Local path to the saved audio file
        'duration': float,          # Audio duration in seconds (from API)
        'estimated_cost': float,    # Estimated cost in USD for this generation
        'inference_time': float     # Time waiting for Fal results in seconds
    }
- Retries: Implements basic exponential backoff for API calls.

Usage Notes / Integration with storyshot_.py:
- Environment Variables: Requires `FAL_KEY`. Optional: `FALAI_TTS_MODEL`, `FALAI_TTS_VOICE`.
- Dependencies: Requires `fal-client`, `aiohttp`, `python-dotenv`.
- Caching: This module does *not* handle caching; caching based on sentence text and output paths
  is expected to be managed by the calling script (`storyshot_.py`).
- Concurrency: Designed to be callable concurrently (e.g., using `asyncio.gather` in the caller).
- Error Handling: Raises `TextToSpeechError` on failure after retries.
- Initialization: A `TextToSpeech` instance should be created by the orchestrator.
"""

import os
import logging
import time # Import time for monotonic()
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from dotenv import load_dotenv
import fal_client
import aiohttp

# --- Configuration ---
load_dotenv()
FAL_KEY = os.getenv("FAL_KEY")
FALAI_TTS_MODEL_DEFAULT = "fal-ai/playai/tts/v3"
FALAI_TTS_MODEL = os.getenv("FALAI_TTS_MODEL", FALAI_TTS_MODEL_DEFAULT)
DEFAULT_VOICE = "Jennifer (English (US)/American)"
FALAI_TTS_VOICE = os.getenv("FALAI_TTS_VOICE", DEFAULT_VOICE)

DEFAULT_API_RETRY_COUNT = 1
COST_PER_MINUTE = Decimal("0.03")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class TextToSpeechError(Exception):
    """Custom exception for TextToSpeech errors."""
    pass

class TextToSpeech:
    """
    Handles text-to-speech conversion using Fal.ai (Async Ready).
    Tracks estimated costs and inference times.
    """

    def __init__(self, model: str = FALAI_TTS_MODEL, voice: str = FALAI_TTS_VOICE, api_retries: int = DEFAULT_API_RETRY_COUNT):
        # ... (init remains the same) ...
        self.model = model
        self.voice = voice
        self.api_retries = api_retries
        self.cost_per_minute = COST_PER_MINUTE
        self.total_estimated_cost = Decimal("0.0")
        self.available_voices = None # Still no dynamic fetch
        logger.info(f"TextToSpeech initialized. Model: {self.model}, Voice: {self.voice}. Cost: ${self.cost_per_minute}/min. Async ready.")


    def get_total_estimated_cost(self) -> float:
        return float(self.total_estimated_cost)

    def _calculate_cost(self, duration_seconds: float) -> Decimal:
        if duration_seconds <= 0: return Decimal("0.0")
        cost = (Decimal(str(duration_seconds)) / Decimal("60")) * self.cost_per_minute
        return cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


    async def process_sentence(self, sentence_text: str, output_audio_path: str, force_regenerate: bool = False) -> dict | None:
        """
        Converts a sentence to speech asynchronously, saves it, and returns info including cost and timing.

        Args:
            sentence_text (str): The text to convert.
            output_audio_path (str): The full path to save the generated audio file.
            force_regenerate (bool): If True, bypasses cache check (caching logic is in storyshot_.py).

        Returns:
            dict: {
                    'audio_path': str,
                    'duration': float,          # Audio duration in seconds
                    'estimated_cost': float,    # Estimated cost in USD
                    'inference_time': float     # Time waiting for Fal results in seconds
                  } on success.
            None: On failure after retries.

        Raises:
            TextToSpeechError: If API call fails definitively after retries or download fails.
        """
        logger.info(f"Requesting TTS: \"{sentence_text[:60]}...\" -> {os.path.basename(output_audio_path)}")
        output_dir = os.path.dirname(output_audio_path)
        os.makedirs(output_dir, exist_ok=True)

        attempts = 0
        last_exception = None
        while attempts <= self.api_retries:
            # --- Start timing before the API interaction ---
            request_start_time = time.monotonic()
            inference_duration = -1.0 # Default value if we fail before getting result

            try:
                logger.debug(f"Fal.ai TTS API submit attempt {attempts + 1}/{self.api_retries + 1}")
                arguments = {"input": sentence_text, "voice": self.voice}
                logger.debug(f"Submitting with arguments: {arguments}")

                handler = await fal_client.submit_async(self.model, arguments=arguments)

                # --- Measure time spent waiting for Fal.ai ---
                get_result_start_time = time.monotonic()
                result = await handler.get()
                get_result_end_time = time.monotonic()
                inference_duration = get_result_end_time - get_result_start_time # More focused timing

                # --- Validate Response and Extract Info ---
                if not isinstance(result, dict) or "audio" not in result:
                    raise TextToSpeechError(f"Unexpected API response format (missing 'audio'): {result}")
                audio_info = result["audio"]
                if not isinstance(audio_info, dict) or "url" not in audio_info or "duration" not in audio_info:
                     raise TextToSpeechError(f"Unexpected API response format (missing 'url' or 'duration' in 'audio'): {audio_info}")

                audio_url = audio_info["url"]
                api_duration = float(audio_info["duration"])
                estimated_cost = self._calculate_cost(api_duration)

                logger.info(f"API success. Duration: {api_duration:.3f}s. Est Cost: ${estimated_cost:.4f}. Inference Wait: {inference_duration:.3f}s")

                # --- Download the Audio File Asynchronously ---
                download_start_time = time.monotonic()
                logger.debug(f"Downloading audio from {audio_url}...")
                async with aiohttp.ClientSession() as session:
                    async with session.get(audio_url) as response:
                        response.raise_for_status()
                        with open(output_audio_path, 'wb') as f:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk: break
                                f.write(chunk)
                download_duration = time.monotonic() - download_start_time
                logger.info(f"Audio downloaded ({download_duration:.3f}s) and saved to: {output_audio_path}")

                # Update total cost
                self.total_estimated_cost += estimated_cost

                return {
                    'audio_path': output_audio_path,
                    'duration': api_duration,
                    'estimated_cost': float(estimated_cost),
                    'inference_time': inference_duration # Add inference time
                }

            # --- Error Handling & Retry Logic ---
            # ... (Error handling remains the same as before)...
            except aiohttp.ClientError as e:
                 logger.error(f"AIOHTTP Error during download (Attempt {attempts + 1}): {e}")
                 last_exception = e
            except fal_client.RequestError as e: # Catch documented RequestError
                logger.error(f"Fal Client RequestError (Attempt {attempts + 1}): {e}")
                last_exception = e
            except Exception as e:
                error_type_name = type(e).__name__
                if error_type_name == 'FalClientError': # Check specific error type if needed
                     logger.error(f"Fal Client Error (Attempt {attempts + 1}) [{error_type_name}]: {e}")
                else:
                    logger.error(f"Unexpected error during TTS (Attempt {attempts + 1}) [{error_type_name}]: {e}", exc_info=False)
                last_exception = e

            attempts += 1
            if attempts <= self.api_retries:
                wait_time = 2 ** (attempts - 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Max retries ({self.api_retries}) reached for TTS processing. Giving up on: \"{sentence_text[:60]}...\"")
                if os.path.exists(output_audio_path):
                    try: os.remove(output_audio_path)
                    except OSError as rm_err: logger.warning(f"Could not remove incomplete file {output_audio_path}: {rm_err}")
                raise TextToSpeechError(f"Fal.ai TTS failed for sentence '{sentence_text[:60]}...' after {self.api_retries + 1} attempts.") from last_exception
        return None # Should not be reached


# --- Testing Block (Concurrent Async) ---
async def run_concurrent_test(num_concurrent=3):
    """Runs multiple TTS requests concurrently."""
    print(f"--- Testing text2speech_.py (Concurrent Async x{num_concurrent}) ---")
    # ... (pre-computation check remains the same) ...
    if not os.path.exists(".env") or not FAL_KEY or ':' not in FAL_KEY:
        print("Error: `.env` file missing, `FAL_KEY` not found, or key format incorrect.")
        print("--- Pre-computation checks failed. Aborting test. ---")
        return

    test_sentences = [
        "This is the first test sentence running concurrently.",
        "Here comes the second sentence, processed potentially in parallel.",
        "And finally, the third sentence joins the asynchronous fun.",
        "Number four adds a little more work.",
        "Sentence five, are we there yet?"
    ]
    test_sentences = test_sentences[:num_concurrent]

    test_output_dir = os.path.join("temp_files", "audio_test_concurrent")
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"\nConfiguration:...") # (Config printout same as before)

    print("\nInstantiating TextToSpeech handler...")
    tts_handler = TextToSpeech(api_retries=1)

    tasks = []
    print(f"\nCreating {len(test_sentences)} concurrent TTS tasks...")
    for i, sentence in enumerate(test_sentences):
        filename = f"test_output_concurrent_{i+1}.mp3"
        output_path = os.path.join(test_output_dir, filename)
        task = asyncio.create_task(
            tts_handler.process_sentence(sentence, output_path, force_regenerate=True),
            name=f"TTS_Task_{i+1}"
        )
        tasks.append(task)

    print("Waiting for tasks to complete...")
    # get current async safe time
    start_wait_time = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n--- Concurrent Test Results ---")
    successful_requests = 0
    total_duration = 0.0

    final_cost = Decimal("0.0")

    for i, result in enumerate(results):
        task_name = tasks[i].get_name()
        if isinstance(result, Exception):
            print(f"{task_name}: FAILED")
            logger.error(f"Error in {task_name}: {result}", exc_info=isinstance(result, TextToSpeechError))
        elif isinstance(result, dict):
            successful_requests += 1
            duration = result.get('duration', 0)
            cost = Decimal(str(result.get('estimated_cost', 0)))
            inference_time = result.get('inference_time', 0) # Get inference time
            total_duration += duration
            final_cost += cost
            print(f"{task_name}: SUCCESS")
            print(f"  - Audio: {os.path.basename(result.get('audio_path', 'N/A'))}")
            print(f"  - Duration: {duration:.3f}s")
            print(f"  - Est. Cost: ${cost:.4f}")
            print(f"  - Inference Wait: {inference_time:.3f}s") # Print inference time
            # Cleanup...
        else:
            print(f"{task_name}: UNEXPECTED RESULT TYPE: {type(result)}")

    print("\n--- Concurrent Test Summary ---")
    print(f"Successful Requests: {successful_requests} / {len(test_sentences)}")
    print(f"Total Audio Duration Generated: {total_duration:.3f} seconds")
    total_wait_time = time.monotonic() - start_wait_time
    print(f"Total Wait Time : {total_wait_time:.3f} seconds") # Report total wait time
    handler_total_cost = tts_handler.get_total_estimated_cost()
    print(f"Total Estimated Cost (Summed from results): ${final_cost:.4f}")
    print(f"Total Estimated Cost (From handler instance): ${handler_total_cost:.4f}")
    if abs(float(final_cost) - handler_total_cost) > 0.00001:
         logger.warning("Discrepancy noted between summed cost and handler total cost!")

    print("\n--- End of text2speech_.py Concurrent Test ---")

if __name__ == "__main__":
    try:
        asyncio.run(run_concurrent_test(num_concurrent=3))
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
