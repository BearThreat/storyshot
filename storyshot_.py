"""
StoryShot Orchestrator (storyshot_.py) - MVP Version 7 (Cascading Skips)

Main script for the StoryShot application. This version orchestrates:
1. Loading a transcript.
2. Breaking it into sentences using NLTK.
3. Managing a cache (JSON) for sentence-level processing results.
4. Orchestrating Text-to-Speech (TTS) using text2speech_.py.
5. Orchestrating Storyboard Prompt Generation using sentence2storyboard_.py.
    -> Can be skipped with --skip-prompting.
6. Orchestrating Prompt-to-Image Generation using prompt2image_openai_.py.
    -> Can be skipped with --skip-images or if prompting is skipped.
7. Orchestrating Image-to-Video Generation using image2video_.py.
    -> Can be skipped with --skip-videos or if images are skipped.
8. Orchestrating Final Video/Audio Assembly using ffmpeg_assembler_.py.
    -> Assembly is *always attempted* unless explicitly skipped with --skip-assembly.
       The assembler must handle missing/failed inputs for individual sentences.
9. Storing results (audio, prompts, images, videos, costs, times) in the cache.
10. Tracking overall costs and timings for the run across stages.

Dependencies:
- Python Packages: nltk, python-dotenv, fal-client, aiohttp, httpx, openai, tiktoken
- Project Modules: text2speech_.py, openrouter_engine_.py, sentence2storyboard_.py, prompt2image_openai_.py, image2video_.py, ffmpeg_assembler_.py
- Environment: .env file with FAL_KEY, OPENROUTER_KEY, OPENAI_KEY. Optional model overrides.
- External Tools: FFmpeg (REQUIRED for assembly stage).

Usage:
   python storyshot_.py <path_to_transcript.txt> [OPTIONS]

Options:
  --force-regenerate, -f    Force regeneration of all steps, ignoring cache.
  --skip-prompting          Skip storyboard prompt generation (implies skipping images & videos).
  --skip-images             Skip prompt-to-image generation (implies skipping videos).
  --skip-videos             Skip image-to-video generation.
  --skip-assembly           Skip the final video/audio assembly stage.

Design Principles:
- Class-based structure (`StoryShotOrchestrator`).
- Requires an initialized OpenRouterEngine instance.
- Sync-structured async execution (using asyncio.gather for concurrency).
- Pragmatic, focusing on functionality.
- Script-first testing via `if __name__ == "__main__"`.
- Caching implemented using a JSON file (`cache_data/cache.json`).
- Uses .env for configuration.
- Tracks cost and timing metrics per stage and overall.

Cache Structure (`cache_data/cache.json`): (Remains the same as V6)
{
  "normalized_sentence_key_1": {
    "original_sentence": "...",
    "tts_output": { ... },             # Result from text2speech_
    "storyboard_prompts": {            # Result from sentence2storyboard_
        "status": "success" | "failed",
        "prompts": ["prompt1", ...],
        "cost": float, ...,
        # --- Embedded Image/Video Results ---
        "image_outputs": [             # List matching prompts length
            {                          # Result from prompt2image_
                "status": "success" | "failed",
                "image_path": "...",
                "cost": float, ...,
                # --- Embedded Video Result ---
                "video_output": {      # Result from image2video_ (or null/pending)
                    "status": "success" | "failed",
                    "video_path": "...",
                    "cost": float, ...
                } | null
            },
            ... # More image/video slots
        ] | null
    },
    # The assembler reads from cache but doesn't typically write back to it directly
  }, ...
}
"""

import os
import sys
import json
import logging
import time
import asyncio
import hashlib
import argparse
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
from pathlib import Path
from dotenv import load_dotenv
from decimal import Decimal, ROUND_HALF_UP
import shutil

# --- Import other project modules ---
try:
    from text2speech_ import TextToSpeech, TextToSpeechError
    from openrouter_engine_ import OpenRouterEngine, OpenRouterError
    from sentence2storyboard_ import SentenceToStoryboard, SentenceToStoryboardError
    from prompt2image_openai_ import PromptToImageOpenAI, PromptToImageError
    from image2video_ import ImageToVideoFal, ImageToVideoError
    from ffmpeg_assembler_ import FFmpegAssembler, FFmpegAssemblyError # Assumes V5+ of assembler
except ImportError as e:
    print(f"Error: Could not import project modules. Ensure *.py files exist in the same directory or Python path. Details: {e}")
    sys.exit(1)

# --- Configuration ---
load_dotenv()

CACHE_DIR = Path("cache_data")
TEMP_DIR = Path("temp_files")
AUDIO_SUBDIR = TEMP_DIR / "audio"
IMAGES_SUBDIR = TEMP_DIR / "images"
VIDEOS_SUBDIR = TEMP_DIR / "videos"
FFMPEG_LIST_SUBDIR = TEMP_DIR / "ffmpeg_concat_lists" # Used by assembler
FINAL_CLIPS_SUBDIR = TEMP_DIR / "final_clips"       # Used by assembler
CACHE_FILE = CACHE_DIR / "cache.json"
FINAL_MOVIE_FILENAME = "storyshot_output.mp4" # Default name for the final movie

# Create necessary directories
CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
AUDIO_SUBDIR.mkdir(exist_ok=True)
IMAGES_SUBDIR.mkdir(exist_ok=True)
VIDEOS_SUBDIR.mkdir(exist_ok=True)
FFMPEG_LIST_SUBDIR.mkdir(exist_ok=True) # Ensure assembler dirs are created
FINAL_CLIPS_SUBDIR.mkdir(exist_ok=True)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StoryShotOrchestrator:
    """
    Orchestrates the transcript-to-movie pipeline steps (V7 - Cascading Skips).
    Manages caching and state. Requires an initialized OpenRouterEngine.
    Assembly stage is always attempted unless explicitly skipped.
    """

    def __init__(self, engine: OpenRouterEngine):
        """
        Initializes the orchestrator.

        Args:
            engine (OpenRouterEngine): An initialized and available OpenRouterEngine instance.
        """
        logger.info("Initializing StoryShotOrchestrator (V7 - Cascading Skips)...")
        if not isinstance(engine, OpenRouterEngine) or not engine.is_available():
           raise ValueError("An initialized and available OpenRouterEngine instance is required.")
        self.engine = engine

        self.cache_file_path = CACHE_FILE
        self.audio_output_dir = AUDIO_SUBDIR
        self.image_output_dir = IMAGES_SUBDIR
        self.video_output_dir = VIDEOS_SUBDIR
        self.final_movie_path = Path(FINAL_MOVIE_FILENAME) # Set final movie path
        self.cache = self._load_cache()

        # Initialize Handlers
        self.tts_handler = TextToSpeech()
        self.storyboard_handler = SentenceToStoryboard(self.engine)
        self.image_gen_handler = PromptToImageOpenAI()
        self.video_gen_handler = ImageToVideoFal()
        self.assembler = FFmpegAssembler(final_movie_path=self.final_movie_path) # Assumes V5+

        # Tracking metrics for the current run
        self.run_start_time = None
        self.run_end_time = None
        self._reset_run_metrics() # Initialize metrics


    def _reset_run_metrics(self):
        """Resets metrics for a new run."""
        # TTS Metrics
        self.current_run_cost_tts = Decimal("0.0"); self.current_run_tts_tasks = 0; self.current_run_tts_success = 0; self.current_run_tts_failed = 0; self.current_run_tts_cache_hits = 0; self.total_tts_inference_time = 0.0; self.total_tts_api_wait_time = 0.0
        # Storyboard Metrics (SB)
        self.current_run_cost_sb = Decimal("0.0"); self.current_run_sb_tasks = 0; self.current_run_sb_success = 0; self.current_run_sb_failed = 0; self.current_run_sb_cache_hits = 0; self.total_sb_llm_time = 0.0; self.total_sb_api_wait_time = 0.0; self.total_sb_prompt_tokens = 0; self.total_sb_completion_tokens = 0
        # Image Generation Metrics (IMG)
        self.current_run_cost_img = Decimal("0.0"); self.current_run_img_tasks = 0; self.current_run_img_success = 0; self.current_run_img_failed = 0; self.current_run_img_cache_hits = 0; self.total_img_request_time = 0.0; self.total_img_api_wait_time = 0.0; self.total_img_prompt_tokens = 0; self.total_img_output_tokens = 0
        # Video Generation Metrics (VID)
        self.current_run_cost_vid = Decimal("0.0"); self.current_run_vid_tasks = 0; self.current_run_vid_success = 0; self.current_run_vid_failed = 0; self.current_run_vid_cache_hits = 0; self.total_vid_inference_time = 0.0; self.total_vid_api_wait_time = 0.0;
        # Assembly Metrics (ASM)
        self.assembly_start_time = None; self.assembly_end_time = None; self.assembly_success = None; self.final_movie_output_path = None
        # Combined Metrics
        self.current_run_cost_total = Decimal("0.0")


    def _load_cache(self) -> dict:
        """Loads the cache data from the JSON file."""
        if self.cache_file_path.exists():
            try:
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                    logger.info(f"Cache loaded from {self.cache_file_path} ({len(loaded_cache)} entries).")
                    # Simple validation/migration for key fields
                    for norm_key, entry in loaded_cache.items():
                        entry.setdefault("original_sentence", "[Unknown - Loaded from Cache]")
                        entry.setdefault("tts_output", None)
                        entry.setdefault("storyboard_prompts", None)
                        if isinstance(entry.get("storyboard_prompts"), dict):
                            sb_data = entry["storyboard_prompts"]
                            sb_data.setdefault("image_outputs", None)
                            if isinstance(sb_data.get("image_outputs"), list):
                                for img_out in sb_data["image_outputs"]:
                                    if isinstance(img_out, dict):
                                        img_out.setdefault("video_output", None)
                                    else:
                                        logger.warning(f"Found non-dict item in image_outputs for key '{norm_key}'. Check cache integrity.")
                    return loaded_cache
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {self.cache_file_path}. Starting empty.", exc_info=True)
                return {}
            except Exception as e:
                logger.error(f"Failed to load cache: {e}. Starting empty.", exc_info=True)
                return {}
        else:
            logger.warning(f"Cache file {self.cache_file_path} not found. Starting empty.")
            return {}

    def _save_cache(self, cache_to_save=None):
        """Saves the current cache data (or provided dict) to the JSON file."""
        target_cache = cache_to_save if cache_to_save is not None else self.cache
        try:
            # Convert Decimal to float before saving to JSON
            def convert_decimals(obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert_decimals(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_decimals(i) for i in obj]
                return obj

            cache_for_json = convert_decimals(target_cache)

            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_for_json, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cache saved to {self.cache_file_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}", exc_info=True)

    def _normalize_sentence(self, sentence: str) -> str:
        """Normalizes a sentence for use as a cache key."""
        return sentence.lower().strip()

    def _get_sentence_hash(self, text_to_hash: str) -> str:
        """Generates a short hash for text, useful for filenames."""
        # Use the normalized key for consistent hashing
        normalized_text = self._normalize_sentence(text_to_hash)
        return hashlib.sha1(normalized_text.encode('utf-8')).hexdigest()[:10]

    def _get_audio_output_path(self, normalized_key: str) -> Path:
        """Generates a consistent output path for the audio file."""
        sentence_hash = self._get_sentence_hash(normalized_key)
        filename = f"audio_{sentence_hash}.mp3"
        return self.audio_output_dir / filename

    def _get_image_output_path(self, normalized_key: str, prompt_index: int) -> Path:
        """Generates a consistent output path for an image file."""
        sentence_hash = self._get_sentence_hash(normalized_key)
        filename = f"image_{sentence_hash}_p{prompt_index:02d}.png"
        return self.image_output_dir / filename

    def _get_video_output_path(self, normalized_key: str, prompt_index: int) -> Path:
        """Generates a consistent output path for a video file."""
        sentence_hash = self._get_sentence_hash(normalized_key)
        filename = f"video_{sentence_hash}_p{prompt_index:02d}.mp4" # Assuming mp4
        return self.video_output_dir / filename

    def load_transcript(self, filepath: str | Path) -> str:
        """Loads the transcript text from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                transcript = f.read()
                logger.info(f"Transcript loaded from {filepath}")
                return transcript
        except FileNotFoundError:
            logger.error(f"Transcript file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading transcript file {filepath}: {e}")
            raise

    def tokenize_sentences(self, text: str) -> list[str]:
        """Splits the transcript text into sentences using NLTK."""
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        logger.info(f"Transcript split into {len(sentences)} sentences.")
        return sentences


    async def process_transcript(self,
                                 transcript_text: str,
                                 force_regenerate: bool = False,
                                 skip_prompting: bool = False, # Effective skip flags passed in
                                 skip_images: bool = False,
                                 skip_videos: bool = False,
                                 skip_assembly: bool = False):
        """
        Main orchestration logic for processing the transcript through all stages. (V7)
        Handles caching and calls modules concurrently where appropriate.
        Uses the effective skip flags determined by command-line args & cascading logic.
        """
        self.run_start_time = time.monotonic()
        self._reset_run_metrics()
        # Log the *effective* skip status passed to this method
        logger.info(f"--- Starting Transcript Processing Run (ForceR: {force_regenerate}, SkipP: {skip_prompting}, SkipI: {skip_images}, SkipV: {skip_videos}, SkipA: {skip_assembly}) ---")

        original_sentences = self.tokenize_sentences(transcript_text)
        if not original_sentences:
            logger.warning("No sentences found in the transcript. Exiting processing.")
            self.run_end_time = time.monotonic(); return

        ordered_sentence_keys = [self._normalize_sentence(s) for s in original_sentences]

        # Initialize cache entries upfront
        for i, normalized_key in enumerate(ordered_sentence_keys):
            sentence = original_sentences[i]
            cache_entry = self.cache.setdefault(normalized_key, {"original_sentence": sentence})
            cache_entry.setdefault("tts_output", None)
            cache_entry.setdefault("storyboard_prompts", None)


        # ==============================
        # --- Step 1: Text-to-Speech ---
        # ==============================
        # TTS is independent of other skips
        logger.info("--- Stage 1: Text-to-Speech ---")
        tts_tasks = []; tts_task_info = {}
        for i, normalized_key in enumerate(ordered_sentence_keys):
            cache_entry = self.cache[normalized_key] # Should exist
            sentence = cache_entry["original_sentence"]
            tts_cache_data = cache_entry.get('tts_output')
            if not force_regenerate and tts_cache_data and tts_cache_data.get('status') == 'success':
                self.current_run_tts_cache_hits += 1; continue

            logger.debug(f"TTS Prep: Sentence {i+1}/{len(original_sentences)} ('{normalized_key[:10]}').")
            output_audio_path = self._get_audio_output_path(normalized_key)
            task = asyncio.create_task(
                self.tts_handler.process_sentence(sentence, str(output_audio_path), force_regenerate),
                name=f"TTS_{i+1}_{normalized_key[:10]}"
            )
            tts_tasks.append(task); tts_task_info[task.get_name()] = {"norm_key": normalized_key, "index": i}; self.current_run_tts_tasks += 1

        if tts_tasks:
            logger.info(f"Executing {len(tts_tasks)} TTS tasks concurrently...")
            gather_start_time = time.monotonic(); results = await asyncio.gather(*tts_tasks, return_exceptions=True); gather_end_time = time.monotonic()
            self.total_tts_api_wait_time = gather_end_time - gather_start_time; logger.info(f"Finished waiting for TTS tasks. Wait time: {self.total_tts_api_wait_time:.3f}s")

            for i, result in enumerate(results):
                task_name = tts_tasks[i].get_name(); task_meta = tts_task_info[task_name]; norm_key = task_meta["norm_key"]
                cache_entry = self.cache[norm_key] # Should exist

                if isinstance(result, Exception):
                    self.current_run_tts_failed += 1; logger.error(f"TTS task {task_name} FAILED: {result}", exc_info=False)
                    cache_entry['tts_output'] = {"status": "failed", "error_message": str(result)}
                elif isinstance(result, dict):
                     self.current_run_tts_success += 1
                     tts_cost = Decimal(str(result.get('estimated_cost', '0.0'))); tts_inf_time = result.get('inference_time', 0.0)
                     cache_entry['tts_output'] = {"status": "success", "audio_path": str(result.get('audio_path')), "duration": result.get('duration'), "estimated_cost": tts_cost, "inference_time": tts_inf_time, "error_message": None}
                     self.current_run_cost_tts += tts_cost; self.current_run_cost_total += tts_cost; self.total_tts_inference_time += tts_inf_time
                else:
                    self.current_run_tts_failed += 1; logger.error(f"TTS task {task_name} returned unexpected type: {type(result)}")
                    cache_entry['tts_output'] = {"status": "failed", "error_message": f"Unexpected result type: {type(result)}"}

            self._save_cache() # Save cache after processing stage results
        else:
            logger.info("No new TTS tasks needed for this run.")


        # =========================================
        # --- Step 2: Sentence-to-Storyboard    ---
        # =========================================
        # Use the effective skip_prompting flag
        if skip_prompting:
            logger.info("--- Stage 2: Sentence-to-Storyboard (Skipped via '--skip-prompting') ---")
        else:
            logger.info("--- Stage 2: Sentence-to-Storyboard ---")
            sb_tasks = []; sb_task_info = {}
            for i, normalized_key in enumerate(ordered_sentence_keys):
                cache_entry = self.cache[normalized_key] # Should exist
                sentence = cache_entry["original_sentence"]
                tts_results = cache_entry.get('tts_output')
                # Skip if TTS failed or missing duration
                if not tts_results or tts_results.get('status') != 'success': continue
                audio_duration = tts_results.get('duration')
                if audio_duration is None or audio_duration <= 0: continue

                # Initialize storyboard structure if missing
                if 'storyboard_prompts' not in cache_entry or cache_entry['storyboard_prompts'] is None:
                    cache_entry['storyboard_prompts'] = {"status": "pending", "prompts": [], "image_outputs": None}

                sb_cache_data = cache_entry['storyboard_prompts']
                # Cache check
                if not force_regenerate and sb_cache_data and sb_cache_data.get('status') == 'success':
                    self.current_run_sb_cache_hits += 1; continue

                logger.debug(f"SB Prep: Sentence {i+1}/{len(original_sentences)} ('{normalized_key[:10]}').")
                task = asyncio.create_task( self.storyboard_handler.generate_prompts_for_sentence(sentence, audio_duration), name=f"SB_{i+1}_{normalized_key[:10]}" )
                sb_tasks.append(task); sb_task_info[task.get_name()] = {"norm_key": normalized_key, "index": i}; self.current_run_sb_tasks += 1

            if sb_tasks:
                logger.info(f"Executing {len(sb_tasks)} Storyboard Generation tasks concurrently...")
                gather_start_time = time.monotonic(); results = await asyncio.gather(*sb_tasks, return_exceptions=True); gather_end_time = time.monotonic()
                self.total_sb_api_wait_time = gather_end_time - gather_start_time; logger.info(f"Finished waiting for Storyboard tasks. Wait time: {self.total_sb_api_wait_time:.3f}s")

                for i, result in enumerate(results):
                    task_name = sb_tasks[i].get_name(); task_meta = sb_task_info[task_name]; norm_key = task_meta["norm_key"]; cache_entry = self.cache[norm_key]
                    # Ensure storyboard_prompts key exists before updating
                    if 'storyboard_prompts' not in cache_entry or cache_entry['storyboard_prompts'] is None: cache_entry['storyboard_prompts'] = {}

                    if isinstance(result, Exception):
                        self.current_run_sb_failed += 1; logger.error(f"SB task {task_name} FAILED: {result}", exc_info=False)
                        cache_entry['storyboard_prompts'].update({"status": "failed", "error_message": str(result)})
                    elif isinstance(result, dict):
                        if result.get("status") == "success":
                            self.current_run_sb_success += 1
                            sb_cost = result.get('cost', Decimal("0.0")); sb_time = result.get('llm_request_time', 0.0)
                            p_tokens = result.get("prompt_tokens", 0); c_tokens = result.get("completion_tokens", 0)
                            # Ensure image_outputs is initialized correctly when prompts succeed
                            cache_entry['storyboard_prompts'].update({"status": "success", "prompts": result.get('prompts', []), "cost": sb_cost, "llm_request_time": sb_time, "prompt_tokens": p_tokens, "completion_tokens": c_tokens, "total_tokens": p_tokens + c_tokens, "error_message": None, "image_outputs": None })
                            self.current_run_cost_sb += sb_cost; self.current_run_cost_total += sb_cost; self.total_sb_llm_time += sb_time; self.total_sb_prompt_tokens += p_tokens; self.total_sb_completion_tokens += c_tokens
                        else:
                            self.current_run_sb_failed += 1; error_msg = result.get('error_message', 'Unknown SB failure'); logger.error(f"SB task {task_name} failed internally: {error_msg}")
                            cache_entry['storyboard_prompts'].update({"status": "failed", "error_message": error_msg})
                    else:
                        self.current_run_sb_failed += 1; logger.error(f"SB task {task_name} returned unexpected type: {type(result)}")
                        cache_entry['storyboard_prompts'].update({"status": "failed", "error_message": f"Unexpected result type: {type(result)}"})

                self._save_cache() # Save cache after processing stage results
            else:
                logger.info("No new Storyboard Generation tasks needed for this run.")

        # ===================================
        # --- Step 3: Prompt-to-Image     ---
        # ===================================
        # Use the effective skip_images flag (which might be true due to skip_prompting)
        if skip_images:
            reason = ("'--skip-prompting'" if skip_prompting else "'--skip-images'")
            logger.info(f"--- Stage 3: Prompt-to-Image (Skipped via {reason}) ---")
        else:
            logger.info("--- Stage 3: Prompt-to-Image ---")
            img_tasks = []; img_task_info = {}; img_gen_semaphore = asyncio.Semaphore(5) # Add semaphore for OpenAI rate limits
            # Use ordered_keys for iteration
            for i, normalized_key in enumerate(ordered_sentence_keys):
                cache_entry = self.cache.get(normalized_key)
                # Skip if SB failed or missing prompts
                if not cache_entry or 'storyboard_prompts' not in cache_entry or not cache_entry['storyboard_prompts']: continue
                sb_data = cache_entry['storyboard_prompts']
                if sb_data.get('status') != 'success' or not sb_data.get('prompts'): continue
                prompts = sb_data['prompts']

                # Initialize or validate image_outputs structure
                if 'image_outputs' not in sb_data or sb_data['image_outputs'] is None or not isinstance(sb_data['image_outputs'], list):
                    sb_data['image_outputs'] = [{"status": "pending", "video_output": None}] * len(prompts)
                elif len(sb_data['image_outputs']) != len(prompts):
                     logger.warning(f"Image cache slot mismatch for sentence {i+1} ('{normalized_key[:10]}'). Re-initializing image cache ({len(prompts)} slots).")
                     sb_data['image_outputs'] = [{"status": "pending", "video_output": None}] * len(prompts)
                else: # Ensure video_output key exists in existing slots
                    for slot in sb_data['image_outputs']:
                         if isinstance(slot, dict) and 'video_output' not in slot: slot['video_output'] = None

                # Iterate through prompts and create tasks
                for p_idx, prompt_text in enumerate(prompts):
                    if not prompt_text: logger.warning(f"Skipping empty prompt at index {p_idx} for sentence {i+1} ('{normalized_key[:10]}')."); continue

                    img_cache_slot = sb_data['image_outputs'][p_idx] # Should be a dict now

                    # Cache Check
                    if not force_regenerate and isinstance(img_cache_slot, dict) and img_cache_slot.get('status') == 'success':
                        self.current_run_img_cache_hits += 1; continue

                    logger.debug(f"IMG Prep: Sentence {i+1}, Prompt {p_idx+1} ('{normalized_key[:10]}').")
                    output_image_path = self._get_image_output_path(normalized_key, p_idx)
                    # Pass the semaphore to the image generation call
                    task = asyncio.create_task( self.image_gen_handler.generate_image(prompt_text, output_image_path, semaphore=img_gen_semaphore), name=f"IMG_s{i+1}_p{p_idx+1}_{normalized_key[:6]}" )
                    img_tasks.append(task); img_task_info[task.get_name()] = {"norm_key": normalized_key, "prompt_idx": p_idx}; self.current_run_img_tasks += 1

            if img_tasks:
                 logger.info(f"Executing {len(img_tasks)} Image Generation tasks concurrently (constrained by semaphore, max 5)...")
                 gather_start_time = time.monotonic(); results = await asyncio.gather(*img_tasks, return_exceptions=True); gather_end_time = time.monotonic()
                 self.total_img_api_wait_time = gather_end_time - gather_start_time; logger.info(f"Finished waiting for Image tasks. Wait time: {self.total_img_api_wait_time:.3f}s")

                 for i, result in enumerate(results):
                    task_name = img_tasks[i].get_name(); task_meta = img_task_info[task_name]; norm_key = task_meta["norm_key"]; p_idx = task_meta["prompt_idx"]; cache_entry = self.cache[norm_key]

                    # Safely access the image output slot
                    try:
                        img_output_list = cache_entry['storyboard_prompts']['image_outputs']
                        if not isinstance(img_output_list, list) or p_idx >= len(img_output_list):
                             raise IndexError("Prompt index out of bounds for image_outputs list.")
                    except (KeyError, TypeError, IndexError) as e:
                        logger.error(f"Cache structure error accessing results for IMG task {task_name}: {e}. Cannot store result."); continue

                    # Prepare base structure, preserving existing video output if generation fails/retries
                    base_img_output = {"status": "failed", "video_output": None}
                    if isinstance(img_output_list[p_idx], dict): # Preserve video output if present
                        base_img_output["video_output"] = img_output_list[p_idx].get("video_output")

                    if isinstance(result, Exception):
                        self.current_run_img_failed += 1; logger.error(f"IMG task {task_name} FAILED: {result}", exc_info=False)
                        base_img_output["error_message"] = str(result)
                    elif isinstance(result, dict):
                         img_cost = Decimal(str(result.get('cost', '0.0')))
                         if result.get("status") == "success":
                            self.current_run_img_success += 1; img_time = result.get('request_time', 0.0); p_tokens = result.get('prompt_tokens_est', 0); o_tokens = result.get('output_tokens', 0)
                            base_img_output.update({"status": "success", "image_path": result.get('image_path'), "cost": img_cost, "request_time": img_time, "revised_prompt": result.get('revised_prompt'), "prompt_tokens_est": p_tokens, "output_tokens": o_tokens, "error_message": None})
                            self.current_run_cost_img += img_cost; self.current_run_cost_total += img_cost; self.total_img_request_time += img_time; self.total_img_prompt_tokens += p_tokens; self.total_img_output_tokens += o_tokens
                         else:
                            # Handle internal failure reported by the handler
                            self.current_run_img_failed += 1; error_msg = result.get('error_message', 'Unknown IMG failure'); logger.error(f"IMG task {task_name} failed internally: {error_msg}")
                            base_img_output.update({"status": "failed", "cost": img_cost, "error_message": error_msg})
                            # Count cost even on failure if provided (e.g., partial attempt cost)
                            self.current_run_cost_img += img_cost; self.current_run_cost_total += img_cost
                    else:
                        # Handle unexpected result type
                        self.current_run_img_failed += 1; logger.error(f"IMG task {task_name} returned unexpected type: {type(result)}")
                        base_img_output["error_message"] = f"Unexpected result type: {type(result)}"

                    # Update the specific cache slot
                    img_output_list[p_idx] = base_img_output

                 self._save_cache() # Save cache after processing stage results
            else:
                logger.info("No new Image Generation tasks needed for this run.")


        # ===================================
        # --- Step 4: Image-to-Video      ---
        # ===================================
        # Use the effective skip_videos flag
        if skip_videos:
            reason = ("'--skip-prompting'" if skip_prompting else "'--skip-images'" if skip_images else "'--skip-videos'")
            logger.info(f"--- Stage 4: Image-to-Video (Skipped via {reason}) ---")
        else:
            logger.info("--- Stage 4: Image-to-Video ---")
            vid_tasks = []; vid_task_info = {}
            # Use ordered_keys for iteration
            for i, normalized_key in enumerate(ordered_sentence_keys):
                cache_entry = self.cache.get(normalized_key)
                # Skip if SB failed or image outputs missing/invalid
                if not cache_entry or 'storyboard_prompts' not in cache_entry or not cache_entry['storyboard_prompts']: continue
                sb_data = cache_entry['storyboard_prompts']
                if sb_data.get('status') != 'success' or not sb_data.get('image_outputs') or not isinstance(sb_data.get('image_outputs'), list): continue
                image_outputs = sb_data['image_outputs']
                prompts = sb_data.get('prompts', [])
                if len(prompts) != len(image_outputs):
                    logger.warning(f"Prompt/image count mismatch for sentence {i+1} ('{normalized_key[:10]}'). Skipping video gen."); continue

                for p_idx, img_output in enumerate(image_outputs):
                    # Skip if image generation failed or path missing
                    if not isinstance(img_output, dict) or img_output.get('status') != 'success': continue
                    image_path_str = img_output.get('image_path')
                    prompt_text = prompts[p_idx] if p_idx < len(prompts) else None

                    if not image_path_str or not Path(image_path_str).exists(): logger.warning(f"Image path missing/not found ({image_path_str}) for s{i+1}, p{p_idx+1}. Skip video."); continue
                    if not prompt_text: logger.warning(f"Original prompt missing for image s{i+1}, p{p_idx+1}. Skip video."); continue

                    # Cache Check for video slot
                    vid_cache_data = img_output.get('video_output')
                    if not force_regenerate and isinstance(vid_cache_data, dict) and vid_cache_data.get('status') == 'success':
                        self.current_run_vid_cache_hits += 1; continue

                    logger.debug(f"VID Prep: Sentence {i+1}, Prompt {p_idx+1} ('{normalized_key[:10]}').")
                    output_video_path = self._get_video_output_path(normalized_key, p_idx)
                    task = asyncio.create_task( self.video_gen_handler.generate_video(image_path_str, prompt_text, output_video_path), name=f"VID_s{i+1}_p{p_idx+1}_{normalized_key[:6]}" )
                    vid_tasks.append(task); vid_task_info[task.get_name()] = {"norm_key": normalized_key, "prompt_idx": p_idx}; self.current_run_vid_tasks += 1

            if vid_tasks:
                 logger.info(f"Executing {len(vid_tasks)} Video Generation tasks concurrently (this might take substantial time)...")
                 gather_start_time = time.monotonic(); results = await asyncio.gather(*vid_tasks, return_exceptions=True); gather_end_time = time.monotonic()
                 self.total_vid_api_wait_time = gather_end_time - gather_start_time; logger.info(f"Finished waiting for Video tasks. Wait time: {self.total_vid_api_wait_time:.3f}s")

                 for i, result in enumerate(results):
                     task_name = vid_tasks[i].get_name(); task_meta = vid_task_info[task_name]; norm_key = task_meta["norm_key"]; p_idx = task_meta["prompt_idx"]; cache_entry = self.cache[norm_key]

                     # Safely access the target image output slot to update its video_output
                     try:
                         img_output_slot = cache_entry['storyboard_prompts']['image_outputs'][p_idx]
                         if not isinstance(img_output_slot, dict): raise TypeError("Target image slot is not a dictionary.")
                     except (KeyError, TypeError, IndexError) as e:
                         logger.error(f"Cache structure error accessing results for VID task {task_name}: {e}. Cannot store result."); continue

                     # Process result and update the video_output key within the image slot
                     if isinstance(result, Exception):
                         self.current_run_vid_failed += 1; logger.error(f"VID task {task_name} FAILED: {result}", exc_info=False)
                         img_output_slot['video_output'] = {"status": "failed", "error_message": str(result), "cost": Decimal("0.0"), "inference_time": 0.0}
                     elif isinstance(result, dict):
                         vid_cost = Decimal(str(result.get('cost', '0.0'))) ; vid_inf_time = result.get('inference_time', 0.0)
                         if result.get("status") == "success":
                             self.current_run_vid_success += 1
                             img_output_slot['video_output'] = {"status": "success", "video_path": result.get('video_path'), "cost": vid_cost, "inference_time": vid_inf_time, "error_message": None}
                             self.current_run_cost_vid += vid_cost ; self.current_run_cost_total += vid_cost ; self.total_vid_inference_time += vid_inf_time
                         else:
                             # Handle internal failure from video handler
                             self.current_run_vid_failed += 1; error_msg = result.get('error_message', 'Unknown VID failure'); logger.error(f"VID task {task_name} failed internally: {error_msg}")
                             img_output_slot['video_output'] = {"status": "failed", "cost": vid_cost, "inference_time": vid_inf_time, "error_message": error_msg}
                             self.current_run_cost_vid += vid_cost ; self.current_run_cost_total += vid_cost # Include cost even if failed
                     else:
                          # Handle unexpected result type
                         self.current_run_vid_failed += 1; logger.error(f"VID task {task_name} returned unexpected type: {type(result)}")
                         img_output_slot['video_output'] = {"status": "failed", "error_message": f"Unexpected result type: {type(result)}", "cost": Decimal("0.0"), "inference_time": 0.0}

                 self._save_cache() # Save cache after processing stage results
            else:
                logger.info("No new Video Generation tasks needed for this run.")


        # ==================================
        # --- Step 5: Video/Audio Assembly ---
        # ==================================
        # Use the independent skip_assembly flag
        if skip_assembly:
            logger.info("--- Stage 5: Video/Audio Assembly (Skipped via '--skip-assembly') ---")
            self.assembly_success = None # None = Skipped via flag
        else:
            logger.info("--- Stage 5: Video/Audio Assembly (Attempting) ---")
            self.assembly_start_time = time.monotonic()
            try:
                # Pass the cache, the ordered keys, and the method to generate hashes
                # The assembler (V5+) will now internally check for valid inputs per sentence.
                final_movie_file = await self.assembler.assemble_movie(
                    self.cache,
                    ordered_sentence_keys,
                    self._get_sentence_hash # Pass the method itself
                )
                self.assembly_end_time = time.monotonic() # Record end time on success or failure attempt

                # Check the outcome of the assembly *attempt*
                if final_movie_file and final_movie_file.exists():
                    self.assembly_success = True # Mark as succeeded
                    self.final_movie_output_path = final_movie_file
                    logger.info(f"Assembly completed successfully. Final movie: {self.final_movie_output_path}")
                else:
                    self.assembly_success = False # Mark as failed
                    # The assembler should log details, but we add a summary error here.
                    logger.error("Assembly attempt finished, but no final movie file was generated or verified.")

            except FFmpegAssemblyError as e:
                 self.assembly_end_time = time.monotonic()
                 self.assembly_success = False # Mark as failed
                 logger.error(f"Assembly failed with FFmpeg error: {e}", exc_info=False)
            except Exception as e:
                 self.assembly_end_time = time.monotonic()
                 self.assembly_success = False # Mark as failed
                 logger.error(f"An unexpected error occurred during assembly stage: {e}", exc_info=True)


        # --- Finalize Run ---
        self.run_end_time = time.monotonic()
        logger.info(f"--- Transcript Processing Run Finished ---")
        self.log_run_summary() # Log summary regardless of assembly outcome


    def log_run_summary(self):
        """Logs a summary of the completed processing run."""
        if self.run_start_time is None or self.run_end_time is None:
            logger.warning("Run timing incomplete, cannot log summary."); return

        total_run_time = self.run_end_time - self.run_start_time
        total_sb_tokens = self.total_sb_prompt_tokens + self.total_sb_completion_tokens
        total_img_tokens = self.total_img_prompt_tokens + self.total_img_output_tokens

        # Format costs as floats for printing
        cost_total_str = f"{float(self.current_run_cost_total):.6f}"
        cost_tts_str = f"{float(self.current_run_cost_tts):.4f}"
        cost_sb_str = f"{float(self.current_run_cost_sb):.6f}"
        cost_img_str = f"{float(self.current_run_cost_img):.6f}"
        cost_vid_str = f"{float(self.current_run_cost_vid):.4f}"

        summary_width = 73
        print("\n" + "=" * ((summary_width - 13) // 2) + " Run Summary " + "=" * ((summary_width - 12) // 2))
        print(f"Total Processing Time: {total_run_time:.3f} seconds")
        print(f"Total Estimated Cost (API Calls): ${cost_total_str}")
        print("-" * summary_width)
        # TTS Summary
        print("[Text-to-Speech Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_tts_tasks} / {self.current_run_tts_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_tts_success} / {self.current_run_tts_failed}")
        print(f"  Async Gather Wait Time: {self.total_tts_api_wait_time:.3f}s")
        print(f"  Total API Inference Time (Sum): {self.total_tts_inference_time:.3f}s")
        print(f"  Estimated Cost (This Run): ${cost_tts_str}")
        print("-" * summary_width)
        # Storyboard Summary
        print("[Sentence-to-Storyboard Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_sb_tasks} / {self.current_run_sb_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_sb_success} / {self.current_run_sb_failed}")
        print(f"  Async Gather Wait Time: {self.total_sb_api_wait_time:.3f}s")
        print(f"  Total LLM Request Time (Sum): {self.total_sb_llm_time:.3f}s")
        print(f"  Total Tokens: {total_sb_tokens} ({self.total_sb_prompt_tokens}p + {self.total_sb_completion_tokens}c)")
        print(f"  Estimated Cost (This Run): ${cost_sb_str}")
        print("-" * summary_width)
        # Image Generation Summary
        print("[Prompt-to-Image Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_img_tasks} / {self.current_run_img_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_img_success} / {self.current_run_img_failed}")
        print(f"  Async Gather Wait Time: {self.total_img_api_wait_time:.3f}s")
        print(f"  Total API Request Time (Sum): {self.total_img_request_time:.3f}s")
        print(f"  Total Tokens Est: {total_img_tokens} ({self.total_img_prompt_tokens}p + {self.total_img_output_tokens}o)")
        print(f"  Estimated Cost (This Run): ${cost_img_str}")
        print("-" * summary_width)
        # Video Generation Summary
        print("[Image-to-Video Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_vid_tasks} / {self.current_run_vid_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_vid_success} / {self.current_run_vid_failed}")
        print(f"  Async Gather Wait Time: {self.total_vid_api_wait_time:.3f}s")
        print(f"  Total API Inference Time (Sum): {self.total_vid_inference_time:.3f}s")
        print(f"  Estimated Cost (This Run): ${cost_vid_str}")
        print("-" * summary_width)

        # --- Assembly Summary (Handles None/True/False for skipped/success/fail) ---
        print("[Video/Audio Assembly Stage]")
        if self.assembly_success is None: # Explicitly check for None, meaning skipped via flag
            print("  Status: Skipped (via flag)")
        elif self.assembly_start_time is not None: # Check if start time was set (i.e., stage was attempted)
            assembly_duration = (self.assembly_end_time - self.assembly_start_time) if self.assembly_end_time is not None and self.assembly_start_time is not None else 0.0
            # Use bool True/False for success/failure after attempt
            status_str = "Success" if self.assembly_success else "Failed"
            print(f"  Status: {status_str}")
            print(f"  Execution Time: {assembly_duration:.3f}s")
            if self.assembly_success and self.final_movie_output_path:
                print(f"  Final Output: {self.final_movie_output_path}")
            elif self.assembly_success is False:
                 print("  No final movie file generated or verified due to errors during assembly.")
        # Implicit else: means not skipped by flag, but start time wasn't set (shouldn't happen with current logic)
        # Can add an explicit else here for debugging if needed:
        # else: logger.debug("Assembly status reporting - condition not met for detailed status.")

        print("=" * summary_width + "\n")

        # Log lifetime costs from handlers
        try:
            handler_tts_total_cost = self.tts_handler.get_total_estimated_cost()
            handler_sb_stats = self.storyboard_handler.get_stats(); handler_sb_total_cost = float(handler_sb_stats.get('total_cost', Decimal('0.0')))
            handler_img_stats = self.image_gen_handler.get_stats(); handler_img_total_cost = float(handler_img_stats.get('total_estimated_cost', Decimal('0.0')))
            handler_vid_stats = self.video_gen_handler.get_stats(); handler_vid_total_cost = float(handler_vid_stats.get('total_estimated_cost', Decimal('0.0')))
            logger.info(f"Lifetime Handler Costs Est: TTS=${handler_tts_total_cost:.4f}, SB=${handler_sb_total_cost:.6f}, IMG=${handler_img_total_cost:.6f}, VID=${handler_vid_total_cost:.4f}")
        except Exception as e: logger.warning(f"Could not retrieve lifetime stats from handlers: {e}")


# --- Main Execution Block ---
async def run_orchestrator(args): # Pass the whole args namespace
    """Initializes engine and runs the orchestrator using effective skip flags."""
    or_engine = None
    try:
        logger.info("Initializing OpenRouter Engine...")
        or_engine = OpenRouterEngine(app_title="StoryShot_MainRun_V7", app_url="local://storyshot")
        await or_engine.wait_for_initialization()
        if not or_engine.is_available():
             logger.error("OpenRouter Engine initialization failed. Aborting."); return
        logger.info("OpenRouter Engine Initialized.")

        orchestrator = StoryShotOrchestrator(or_engine)
        transcript = orchestrator.load_transcript(args.transcript_file)

        # Call process_transcript with the *effective* skip flags from args
        await orchestrator.process_transcript(
            transcript,
            args.force_regenerate,
            args.skip_prompting, # Pass the effective skip flags
            args.skip_images,
            args.skip_videos,
            args.skip_assembly
        )

    except FileNotFoundError: sys.exit(1) # Already logged by load_transcript
    except (TextToSpeechError, SentenceToStoryboardError, OpenRouterError, PromptToImageError, ImageToVideoError, FFmpegAssemblyError) as e:
         logger.error(f"A critical processing error occurred in module {type(e).__name__}: {e}", exc_info=False); sys.exit(1)
    except Exception as e: logger.error(f"An unexpected error occurred during orchestration: {e}", exc_info=True); sys.exit(1)
    finally:
         if or_engine:
             logger.info("Closing OpenRouter Engine client..."); await or_engine.close(); logger.info("Engine client closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StoryShot (V7): Transcript to Movie pipeline with cascading skips.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # Use formatter to show defaults
    parser.add_argument("transcript_file", type=str, help="Path to the input transcript text file.")
    parser.add_argument( "-f", "--force-regenerate", action="store_true", help="Force regeneration of all steps, ignoring cache.")
    parser.add_argument("--skip-prompting", action="store_true", help="Skip storyboard prompt generation (implies --skip-images and --skip-videos).")
    parser.add_argument("--skip-images", action="store_true", help="Skip prompt-to-image generation (implies --skip-videos).")
    parser.add_argument("--skip-videos", action="store_true", help="Skip image-to-video generation.")
    parser.add_argument("--skip-assembly", action="store_true", help="Explicitly skip the final video/audio assembly stage.")
    args = parser.parse_args()

    # --- Apply Cascading Skip Logic ---
    # Start with the originally provided flags
    effective_skip_prompting = args.skip_prompting
    effective_skip_images = args.skip_images
    effective_skip_videos = args.skip_videos
    cascade_reason_images = ""
    cascade_reason_videos = ""

    if effective_skip_prompting:
        if not effective_skip_images:
             effective_skip_images = True
             cascade_reason_images = " (due to --skip-prompting)"
        if not effective_skip_videos:
             effective_skip_videos = True
             cascade_reason_videos = " (due to --skip-prompting)"

    if effective_skip_images:
        if not effective_skip_videos:
            effective_skip_videos = True
            # If videos weren't already skipped by prompting, set the reason to images
            if not cascade_reason_videos: cascade_reason_videos = " (due to --skip-images)"

    # Update the args namespace *copies* for clarity, though we could pass effective vars directly
    args.skip_prompting = effective_skip_prompting
    args.skip_images = effective_skip_images
    args.skip_videos = effective_skip_videos

    # --- Log Effective Skips ---
    logger.info("--- Effective Skip Flags for This Run ---")
    logger.info(f"Skip Prompting: {args.skip_prompting}")
    logger.info(f"Skip Images:    {args.skip_images}{cascade_reason_images}")
    logger.info(f"Skip Videos:    {args.skip_videos}{cascade_reason_videos}")
    logger.info(f"Skip Assembly:  {args.skip_assembly}")
    logger.info("---------------------------------------")

    # --- Pre-computation Checks ---
    logger.info("--- Running Pre-computation Checks ---")
    checks_passed = True
    # .env checks
    if not Path(".env").exists(): logger.error("Error: `.env` file missing."); checks_passed = False
    else:
        if not os.getenv("FAL_KEY") or ':' not in os.getenv("FAL_KEY", ""): logger.error("Error: `FAL_KEY` missing or format incorrect."); checks_passed = False
        if not os.getenv("OPENROUTER_KEY"): logger.error("Error: `OPENROUTER_KEY` missing."); checks_passed = False
        if not os.getenv("OPENAI_KEY"): logger.error("Error: `OPENAI_KEY` missing."); checks_passed = False
    # FFmpeg check
    logger.info("Info: Checking for FFmpeg...")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path: logger.info(f"Info: Found FFmpeg executable at: {ffmpeg_path}")
    else: logger.error("Error: FFmpeg executable not found in system PATH. Assembly stage will fail if attempted."); logger.error("       Please install FFmpeg and ensure its directory is in your system's PATH.");
    if not args.skip_assembly and not ffmpeg_path: # Only fail check if assembly is needed but ffmpeg missing
        checks_passed = False
    # Transcript check
    transcript_path = Path(args.transcript_file)
    if not transcript_path.is_file():
         logger.error(f"Error: Transcript file not found at '{args.transcript_file}'")
         # Attempt to create dummy only if it's a common default name
         if transcript_path.name.lower() in ["sample_transcript.txt", "transcript.txt", "input.txt"]:
             logger.warning("Attempting to create dummy transcript...")
             try:
                 with open(transcript_path.name, "w", encoding='utf-8') as f: f.write("This is the first sentence of a dummy transcript.\nThis is the second sentence.\nAnd a final short one.")
                 logger.info(f"Created dummy '{transcript_path.name}'. Please edit it with real content before the next run.")
             except Exception as e: logger.error(f"Failed to create dummy file: {e}"); checks_passed = False
         else: checks_passed = False # Fail if non-default name not found
    # Directory checks
    try:
        CACHE_DIR.mkdir(exist_ok=True); TEMP_DIR.mkdir(exist_ok=True); AUDIO_SUBDIR.mkdir(exist_ok=True); IMAGES_SUBDIR.mkdir(exist_ok=True); VIDEOS_SUBDIR.mkdir(exist_ok=True); FFMPEG_LIST_SUBDIR.mkdir(exist_ok=True); FINAL_CLIPS_SUBDIR.mkdir(exist_ok=True)
    except OSError as e: logger.error(f"Error creating required directories: {e}"); checks_passed = False

    if not checks_passed:
        logger.critical("--- Pre-computation checks failed. Please resolve the errors above before running again. ---"); sys.exit(1)
    else:
        logger.info("--- Pre-computation checks passed. Proceeding... ---")

    # --- Run Async Orchestrator ---
    if sys.platform == "win32" and sys.version_info >= (3, 8):
       asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        # Pass the modified args object to the runner
        asyncio.run(run_orchestrator(args))
    except KeyboardInterrupt:
        logger.info("\nOrchestration interrupted by user.")
    except RuntimeError as e:
        if "Event loop is closed" in str(e) or "cannot schedule new futures" in str(e):
            logger.warning("\nError during async shutdown (event loop likely closed).", exc_info=False)
        else:
            logger.error(f"RuntimeError during execution: {e}", exc_info=True)
    sys.exit(0) # Explicitly exit with success code if no fatal errors occurred