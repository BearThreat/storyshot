"""
StoryShot Orchestrator (storyshot_.py) - MVP Version 4 (Includes Video Gen)

Main script for the StoryShot application. This version orchestrates:
1. Loading a transcript.
2. Breaking it into sentences using NLTK.
3. Managing a cache (JSON) for sentence-level processing results.
4. Orchestrating Text-to-Speech (TTS) using text2speech_.py.
5. Orchestrating Storyboard Prompt Generation using sentence2storyboard_.py.
6. Orchestrating Prompt-to-Image Generation using prompt2image_openai_.py.
7. Orchestrating Image-to-Video Generation using image2video_.py.
8. Storing results (audio, prompts, images, videos, costs, times) in the cache.
9. Tracking overall costs and timings for the run across stages.
10. Final assembly step (FFmpeg) is still placeholder.

Dependencies:
- Python Packages: nltk, python-dotenv, fal-client, aiohttp, httpx, openai, tiktoken
- Project Modules: text2speech_.py, openrouter_engine_.py, sentence2storyboard_.py, prompt2image_openai_.py, image2video_.py
- Environment: .env file with FAL_KEY, OPENROUTER_KEY, OPENAI_KEY. Optional model overrides.
- External Tools: FFmpeg (required for future final assembly step).

Usage:
   python storyshot_.py <path_to_transcript.txt> [--force-regenerate] [--skip-images] [--skip-videos]

Design Principles:
- Class-based structure (`StoryShotOrchestrator`).
- Requires an initialized OpenRouterEngine instance.
- Sync-structured async execution (using asyncio.gather for concurrency).
- Pragmatic, focusing on functionality.
- Script-first testing via `if __name__ == "__main__"`.
- Caching implemented using a JSON file (`cache_data/cache.json`).
- Uses .env for configuration.
- Tracks cost and timing metrics per stage and overall (Video cost TBD).

Cache Structure (`cache_data/cache.json`):
{
  "normalized_sentence_key_1": {
    "original_sentence": "Original sentence text.",
    "tts_output": { ... },
    "storyboard_prompts": {
        "status": "success" | "failed",
        "prompts": ["prompt 1", ...],
        "cost": ..., "llm_request_time": ..., ...,
        "image_outputs": [ // List matching prompts
            { // Entry for prompt 1
                "status": "success" | "failed",
                "image_path": "...",
                "cost": ..., "request_time": ..., ...,
                "video_output": { // Populated by image2video_ stage
                    "status": "success" | "failed",
                    "video_path": "temp_files/videos/...", // path if success
                    "inference_time": 123.456,
                    "cost": 0.0, // Placeholder
                    "error_message": str | null
                } | null // Null if image failed or stage skipped/failed
            },
            // ... entry for prompt 2 ...
        ] | null
    },
    // "video_outputs" key is deprecated, now inside image_outputs
  },
  ...
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

# --- Import other project modules ---
try:
    from text2speech_ import TextToSpeech, TextToSpeechError
    from openrouter_engine_ import OpenRouterEngine, OpenRouterError
    from sentence2storyboard_ import SentenceToStoryboard, SentenceToStoryboardError
    from prompt2image_openai_ import PromptToImageOpenAI, PromptToImageError
    from image2video_ import ImageToVideoFal, ImageToVideoError # Added
except ImportError as e:
    print(f"Error: Could not import project modules. Ensure .py files exist in the same directory or Python path. Details: {e}")
    sys.exit(1)

# --- Configuration ---
load_dotenv()

CACHE_DIR = Path("cache_data")
TEMP_DIR = Path("temp_files")
AUDIO_SUBDIR = TEMP_DIR / "audio"
IMAGES_SUBDIR = TEMP_DIR / "images"
VIDEOS_SUBDIR = TEMP_DIR / "videos" # Added
CACHE_FILE = CACHE_DIR / "cache.json"

# Create necessary directories if they don't exist
CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
AUDIO_SUBDIR.mkdir(exist_ok=True)
IMAGES_SUBDIR.mkdir(exist_ok=True)
VIDEOS_SUBDIR.mkdir(exist_ok=True) # Added

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StoryShotOrchestrator:
    """
    Orchestrates the transcript-to-movie pipeline steps.
    Manages caching and state. Requires an initialized OpenRouterEngine.
    """

    def __init__(self, engine: OpenRouterEngine):
        """
        Initializes the orchestrator.

        Args:
            engine (OpenRouterEngine): An initialized and available OpenRouterEngine instance.
        """
        logger.info("Initializing StoryShotOrchestrator...")
        if not isinstance(engine, OpenRouterEngine) or not engine.is_available():
           raise ValueError("An initialized and available OpenRouterEngine instance is required.")
        self.engine = engine

        self.cache_file_path = CACHE_FILE
        self.audio_output_dir = AUDIO_SUBDIR
        self.image_output_dir = IMAGES_SUBDIR
        self.video_output_dir = VIDEOS_SUBDIR # Added
        self.cache = self._load_cache()

        # Initialize Handlers
        self.tts_handler = TextToSpeech()
        self.storyboard_handler = SentenceToStoryboard(self.engine)
        self.image_gen_handler = PromptToImageOpenAI()
        self.video_gen_handler = ImageToVideoFal() # Added

        # Tracking metrics for the current run
        self.run_start_time = None
        self.run_end_time = None
        self._reset_run_metrics() # Initialize metrics


    def _reset_run_metrics(self):
        """Resets metrics for a new run."""
        # TTS Metrics
        self.current_run_cost_tts = Decimal("0.0")
        self.current_run_tts_tasks = 0; self.current_run_tts_success = 0; self.current_run_tts_failed = 0; self.current_run_tts_cache_hits = 0
        self.total_tts_inference_time = 0.0; self.total_tts_api_wait_time = 0.0

        # Storyboard Metrics (SB)
        self.current_run_cost_sb = Decimal("0.0")
        self.current_run_sb_tasks = 0; self.current_run_sb_success = 0; self.current_run_sb_failed = 0; self.current_run_sb_cache_hits = 0
        self.total_sb_llm_time = 0.0; self.total_sb_api_wait_time = 0.0
        self.total_sb_prompt_tokens = 0; self.total_sb_completion_tokens = 0

        # Image Generation Metrics (IMG)
        self.current_run_cost_img = Decimal("0.0")
        self.current_run_img_tasks = 0; self.current_run_img_success = 0; self.current_run_img_failed = 0; self.current_run_img_cache_hits = 0
        self.total_img_request_time = 0.0; self.total_img_api_wait_time = 0.0
        self.total_img_prompt_tokens = 0; self.total_img_output_tokens = 0

        # Video Generation Metrics (VID) - Added
        self.current_run_cost_vid = Decimal("0.0") # Placeholder
        self.current_run_vid_tasks = 0; self.current_run_vid_success = 0; self.current_run_vid_failed = 0; self.current_run_vid_cache_hits = 0
        self.total_vid_inference_time = 0.0; self.total_vid_api_wait_time = 0.0;

        # Combined Metrics
        self.current_run_cost_total = Decimal("0.0")

    def _load_cache(self) -> dict:
        """Loads the cache data from the JSON file."""
        if self.cache_file_path.exists():
            try:
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                    logger.info(f"Cache loaded from {self.cache_file_path} ({len(loaded_cache)} entries).")
                    # --- Cache Structure Migration/Validation (Example) ---
                    migrated = False
                    for norm_key, entry in loaded_cache.items():
                        # Deprecated top-level 'video_outputs', move data if found (simple example)
                        if "video_outputs" in entry and entry["video_outputs"] is not None :
                             logger.warning(f"Found deprecated 'video_outputs' structure for key '{norm_key}'. Attempting migration (needs review if complex).")
                             # Simple move assuming it was a list matching images (likely not the case)
                             # More robust migration would map based on filenames or indices if possible
                             # For now, just log and remove the old key to avoid confusion
                             del entry["video_outputs"]
                             migrated = True
                             # Ensure image_outputs slots exist and have video_output initialized
                             if entry.get("storyboard_prompts") and entry["storyboard_prompts"].get("image_outputs"):
                                 for img_out in entry["storyboard_prompts"]["image_outputs"]:
                                      if isinstance(img_out, dict) and "video_output" not in img_out:
                                          img_out["video_output"] = None


                    if migrated:
                        logger.warning("Cache structure migration applied. Please review the cache file if needed.")
                        # Consider saving the migrated cache immediately
                        # self._save_cache(loaded_cache) # Pass cache ref if saving here
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

    def _save_cache(self, cache_to_save=None): # Added optional arg
        """Saves the current cache data (or provided dict) to the JSON file."""
        target_cache = cache_to_save if cache_to_save is not None else self.cache
        try:
            # Ensure costs are float before saving
            for key, entry in target_cache.items():
                # TTS Cost
                if entry.get("tts_output") and isinstance(entry["tts_output"].get("estimated_cost"), Decimal):
                    entry["tts_output"]["estimated_cost"] = float(entry["tts_output"]["estimated_cost"])
                # Storyboard Cost
                if entry.get("storyboard_prompts") and isinstance(entry["storyboard_prompts"].get("cost"), Decimal):
                     entry["storyboard_prompts"]["cost"] = float(entry["storyboard_prompts"]["cost"])
                # Image & Video Costs
                if entry.get("storyboard_prompts") and entry["storyboard_prompts"].get("image_outputs"):
                     for img_out in entry["storyboard_prompts"]["image_outputs"]:
                         if isinstance(img_out, dict): # Check if it's a dict before accessing keys
                             if isinstance(img_out.get("cost"), Decimal):
                                 img_out["cost"] = float(img_out["cost"])
                             if img_out.get("video_output") and isinstance(img_out["video_output"].get("cost"), Decimal):
                                 img_out["video_output"]["cost"] = float(img_out["video_output"]["cost"]) # Convert video cost

            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(target_cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cache saved to {self.cache_file_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}", exc_info=True)

    def _normalize_sentence(self, sentence: str) -> str:
        """Normalizes a sentence for use as a cache key."""
        return sentence.lower().strip()

    def _get_sentence_hash(self, text_to_hash: str) -> str:
        """Generates a short hash for text, useful for filenames."""
        return hashlib.sha1(text_to_hash.encode('utf-8')).hexdigest()[:10]

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

    def _get_video_output_path(self, normalized_key: str, prompt_index: int) -> Path: # Added
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

    async def process_transcript(self, transcript_text: str, force_regenerate: bool = False, skip_images: bool = False, skip_videos: bool = False): # Added skip_videos
        """
        Main orchestration logic for processing the transcript through all stages.
        Handles caching and calls modules concurrently where appropriate.
        """
        self.run_start_time = time.monotonic()
        self._reset_run_metrics() # Ensure metrics are fresh
        logger.info(f"--- Starting Transcript Processing Run (ForceR: {force_regenerate}, SkipI: {skip_images}, SkipV: {skip_videos}) ---") # Updated log

        sentences = self.tokenize_sentences(transcript_text)
        if not sentences:
            logger.warning("No sentences found in the transcript. Exiting processing.")
            self.run_end_time = time.monotonic()
            return

        # ==============================
        # --- Step 1: Text-to-Speech ---
        # ==============================
        logger.info("--- Stage: Text-to-Speech ---")
        # ... (TTS logic remains the same) ...
        tts_tasks = []
        tts_task_info = {}
        for i, sentence in enumerate(sentences):
            normalized_key = self._normalize_sentence(sentence)
            cache_entry = self.cache.setdefault(normalized_key, {"original_sentence": sentence})
            # Ensure nested structure
            cache_entry.setdefault("tts_output", None)
            cache_entry.setdefault("storyboard_prompts", None)
            # Deprecated: cache_entry.setdefault("video_outputs", None) - Now inside image_outputs

            tts_cache_data = cache_entry.get('tts_output')
            if not force_regenerate and tts_cache_data and tts_cache_data.get('status') == 'success':
                self.current_run_tts_cache_hits += 1
                continue

            logger.debug(f"TTS Prep: Sentence {i+1}/{len(sentences)}.")
            output_audio_path = self._get_audio_output_path(normalized_key)
            task = asyncio.create_task(
                self.tts_handler.process_sentence(sentence, str(output_audio_path), force_regenerate),
                name=f"TTS_{i+1}_{normalized_key[:10]}"
            )
            tts_tasks.append(task)
            tts_task_info[task.get_name()] = {"norm_key": normalized_key, "orig_sentence": sentence, "index": i}
            self.current_run_tts_tasks += 1

        if tts_tasks:
            logger.info(f"Executing {len(tts_tasks)} TTS tasks concurrently...")
            gather_start_time = time.monotonic()
            results = await asyncio.gather(*tts_tasks, return_exceptions=True)
            gather_end_time = time.monotonic()
            self.total_tts_api_wait_time = gather_end_time - gather_start_time
            logger.info(f"Finished waiting for TTS tasks. Wait time: {self.total_tts_api_wait_time:.3f}s")

            for i, result in enumerate(results):
                task_name = tts_tasks[i].get_name()
                task_meta = tts_task_info[task_name]
                norm_key = task_meta["norm_key"]
                cache_entry = self.cache[norm_key] # Should exist

                if isinstance(result, Exception):
                    self.current_run_tts_failed += 1; logger.error(f"TTS task {task_name} FAILED: {result}", exc_info=False)
                    cache_entry['tts_output'] = {"status": "failed", "error_message": str(result)}
                elif isinstance(result, dict):
                    self.current_run_tts_success += 1
                    tts_cost = Decimal(str(result.get('estimated_cost', '0.0'))); tts_inf_time = result.get('inference_time', 0.0)
                    cache_entry['tts_output'] = {"status": "success", "audio_path": str(result.get('audio_path')), "duration": result.get('duration'), "estimated_cost": float(tts_cost), "inference_time": tts_inf_time, "error_message": None}
                    self.current_run_cost_tts += tts_cost; self.current_run_cost_total += tts_cost; self.total_tts_inference_time += tts_inf_time
                else:
                    self.current_run_tts_failed += 1; logger.error(f"TTS task {task_name} returned unexpected type: {type(result)}")
                    cache_entry['tts_output'] = {"status": "failed", "error_message": f"Unexpected result type: {type(result)}"}
            self._save_cache()
        else: logger.info("No new TTS tasks needed for this run.")


        # =========================================
        # --- Step 2: Sentence-to-Storyboard    ---
        # =========================================
        logger.info("--- Stage: Sentence-to-Storyboard ---")
        # ... (Storyboard logic remains the same) ...
        sb_tasks = []; sb_task_info = {}
        for i, sentence in enumerate(sentences):
            normalized_key = self._normalize_sentence(sentence)
            cache_entry = self.cache[normalized_key]
            tts_results = cache_entry.get('tts_output')
            if not tts_results or tts_results.get('status') != 'success': continue
            audio_duration = tts_results.get('duration')
            if audio_duration is None or audio_duration <= 0: continue

            if 'storyboard_prompts' not in cache_entry or cache_entry['storyboard_prompts'] is None:
                 cache_entry['storyboard_prompts'] = {"status": "pending", "prompts": [], "image_outputs": None}

            sb_cache_data = cache_entry['storyboard_prompts']
            if not force_regenerate and sb_cache_data and sb_cache_data.get('status') == 'success':
                self.current_run_sb_cache_hits += 1; continue

            logger.debug(f"SB Prep: Sentence {i+1}/{len(sentences)}.")
            task = asyncio.create_task( self.storyboard_handler.generate_prompts_for_sentence(sentence, audio_duration), name=f"SB_{i+1}_{normalized_key[:10]}" )
            sb_tasks.append(task); sb_task_info[task.get_name()] = {"norm_key": normalized_key, "index": i}; self.current_run_sb_tasks += 1

        if sb_tasks:
            logger.info(f"Executing {len(sb_tasks)} Storyboard Generation tasks concurrently...")
            gather_start_time = time.monotonic(); results = await asyncio.gather(*sb_tasks, return_exceptions=True); gather_end_time = time.monotonic()
            self.total_sb_api_wait_time = gather_end_time - gather_start_time; logger.info(f"Finished waiting for Storyboard tasks. Wait time: {self.total_sb_api_wait_time:.3f}s")

            for i, result in enumerate(results):
                task_name = sb_tasks[i].get_name(); task_meta = sb_task_info[task_name]; norm_key = task_meta["norm_key"]; cache_entry = self.cache[norm_key]
                if 'storyboard_prompts' not in cache_entry or cache_entry['storyboard_prompts'] is None: cache_entry['storyboard_prompts'] = {}

                if isinstance(result, Exception):
                    self.current_run_sb_failed += 1; logger.error(f"SB task {task_name} FAILED: {result}", exc_info=False)
                    cache_entry['storyboard_prompts'].update({"status": "failed", "error_message": str(result)})
                elif isinstance(result, dict):
                    if result.get("status") == "success":
                        self.current_run_sb_success += 1
                        sb_cost = result.get('cost', Decimal("0.0")); sb_time = result.get('llm_request_time', 0.0)
                        p_tokens = result.get("prompt_tokens", 0); c_tokens = result.get("completion_tokens", 0)
                        cache_entry['storyboard_prompts'].update({"status": "success", "prompts": result.get('prompts', []), "cost": float(sb_cost), "llm_request_time": sb_time, "prompt_tokens": p_tokens, "completion_tokens": c_tokens, "total_tokens": p_tokens + c_tokens, "error_message": None, "image_outputs": None })
                        self.current_run_cost_sb += sb_cost; self.current_run_cost_total += sb_cost; self.total_sb_llm_time += sb_time; self.total_sb_prompt_tokens += p_tokens; self.total_sb_completion_tokens += c_tokens
                    else:
                        self.current_run_sb_failed += 1; error_msg = result.get('error_message', 'Unknown SB failure'); logger.error(f"SB task {task_name} failed internally: {error_msg}")
                        cache_entry['storyboard_prompts'].update({"status": "failed", "error_message": error_msg})
                else:
                    self.current_run_sb_failed += 1; logger.error(f"SB task {task_name} returned unexpected type: {type(result)}")
                    cache_entry['storyboard_prompts'].update({"status": "failed", "error_message": f"Unexpected result type: {type(result)}"})
            self._save_cache()
        else: logger.info("No new Storyboard Generation tasks needed for this run.")


        # ===================================
        # --- Step 3: Prompt-to-Image     ---
        # ===================================
        if skip_images: logger.info("--- Stage: Prompt-to-Image (Skipped via flag) ---")
        else:
            logger.info("--- Stage: Prompt-to-Image ---")
            # ... (Image gen logic remains mostly the same, but ensure image_outputs list structure handles video_output key) ...
            img_tasks = []; img_task_info = {}
            for i, sentence in enumerate(sentences):
                normalized_key = self._normalize_sentence(sentence); cache_entry = self.cache.get(normalized_key)
                if not cache_entry or 'storyboard_prompts' not in cache_entry or not cache_entry['storyboard_prompts']: continue
                sb_data = cache_entry['storyboard_prompts']
                if sb_data.get('status') != 'success' or not sb_data.get('prompts'): continue
                prompts = sb_data['prompts']

                if 'image_outputs' not in sb_data or sb_data['image_outputs'] is None:
                    # Initialize with dicts containing None for video_output
                    sb_data['image_outputs'] = [{"status": "pending", "video_output": None}] * len(prompts)
                elif len(sb_data['image_outputs']) != len(prompts):
                     logger.warning(f"Image cache mismatch for sentence {i+1}. Re-initializing image cache ({len(prompts)} slots).")
                     sb_data['image_outputs'] = [{"status": "pending", "video_output": None}] * len(prompts)
                # Ensure existing slots have the video_output key
                elif isinstance(sb_data['image_outputs'], list):
                    for slot in sb_data['image_outputs']:
                        if isinstance(slot, dict) and 'video_output' not in slot:
                            slot['video_output'] = None


                for p_idx, prompt_text in enumerate(prompts):
                    if not prompt_text: logger.warning(f"Skipping empty prompt at index {p_idx} for sentence {i+1}."); continue

                    # Safely access image_outputs slot
                    img_cache_slot = None
                    if isinstance(sb_data.get('image_outputs'), list) and p_idx < len(sb_data['image_outputs']):
                        img_cache_slot = sb_data['image_outputs'][p_idx]

                    # Cache Check
                    if not force_regenerate and isinstance(img_cache_slot, dict) and img_cache_slot.get('status') == 'success':
                        self.current_run_img_cache_hits += 1; continue

                    logger.debug(f"IMG Prep: Sentence {i+1}/{len(sentences)}, Prompt {p_idx+1}/{len(prompts)}.")
                    output_image_path = self._get_image_output_path(normalized_key, p_idx)
                    task = asyncio.create_task( self.image_gen_handler.generate_image(prompt_text, output_image_path), name=f"IMG_s{i+1}_p{p_idx+1}_{normalized_key[:6]}" )
                    img_tasks.append(task); img_task_info[task.get_name()] = {"norm_key": normalized_key, "sentence_idx": i, "prompt_idx": p_idx}; self.current_run_img_tasks += 1

            if img_tasks:
                logger.info(f"Executing {len(img_tasks)} Image Generation tasks concurrently (this might take time)...")
                gather_start_time = time.monotonic(); results = await asyncio.gather(*img_tasks, return_exceptions=True); gather_end_time = time.monotonic()
                self.total_img_api_wait_time = gather_end_time - gather_start_time; logger.info(f"Finished waiting for Image tasks. Wait time: {self.total_img_api_wait_time:.3f}s")

                for i, result in enumerate(results):
                    task_name = img_tasks[i].get_name(); task_meta = img_task_info[task_name]; norm_key = task_meta["norm_key"]; p_idx = task_meta["prompt_idx"]; cache_entry = self.cache[norm_key]
                    # Get reference to the specific slot in the image_outputs list
                    img_output_list = cache_entry['storyboard_prompts']['image_outputs']

                    if not isinstance(img_output_list, list) or p_idx >= len(img_output_list):
                        logger.error(f"Cache structure error for IMG task {task_name}. Cannot store result."); continue

                    base_img_output = {"status": "failed", "video_output": None} # Base structure

                    if isinstance(result, Exception):
                        self.current_run_img_failed += 1; logger.error(f"IMG task {task_name} FAILED: {result}", exc_info=False)
                        base_img_output["error_message"] = str(result)
                    elif isinstance(result, dict):
                         img_cost = Decimal(str(result.get('cost', '0.0')))
                         if result.get("status") == "success":
                            self.current_run_img_success += 1; img_time = result.get('request_time', 0.0); p_tokens = result.get('prompt_tokens_est', 0); o_tokens = result.get('output_tokens', 0)
                            base_img_output.update({"status": "success", "image_path": result.get('image_path'), "cost": float(img_cost), "request_time": img_time, "revised_prompt": result.get('revised_prompt'), "prompt_tokens_est": p_tokens, "output_tokens": o_tokens, "error_message": None})
                            self.current_run_cost_img += img_cost; self.current_run_cost_total += img_cost; self.total_img_request_time += img_time; self.total_img_prompt_tokens += p_tokens; self.total_img_output_tokens += o_tokens
                         else:
                            self.current_run_img_failed += 1; error_msg = result.get('error_message', 'Unknown IMG failure'); logger.error(f"IMG task {task_name} failed internally: {error_msg}")
                            base_img_output.update({"status": "failed", "cost": float(img_cost), "error_message": error_msg})
                            self.current_run_cost_img += img_cost; self.current_run_cost_total += img_cost # Accumulate cost even on failure
                    else:
                        self.current_run_img_failed += 1; logger.error(f"IMG task {task_name} returned unexpected type: {type(result)}")
                        base_img_output["error_message"] = f"Unexpected result type: {type(result)}"

                    # Ensure we preserve existing video_output if image is just regenerated
                    existing_video_output = img_output_list[p_idx].get("video_output") if isinstance(img_output_list[p_idx], dict) else None
                    if existing_video_output is not None:
                        base_img_output["video_output"] = existing_video_output

                    img_output_list[p_idx] = base_img_output # Update the slot

                self._save_cache()
            else: logger.info("No new Image Generation tasks needed for this run.")


        # ===================================
        # --- Step 4: Image-to-Video      ---    # ADDED STAGE
        # ===================================
        if skip_videos: logger.info("--- Stage: Image-to-Video (Skipped via flag) ---")
        elif skip_images: logger.info("--- Stage: Image-to-Video (Skipped because images were skipped) ---")
        else:
            logger.info("--- Stage: Image-to-Video ---")
            vid_tasks = []; vid_task_info = {}

            for i, sentence in enumerate(sentences):
                normalized_key = self._normalize_sentence(sentence)
                cache_entry = self.cache.get(normalized_key)

                # Pre-conditions check
                if not cache_entry or 'storyboard_prompts' not in cache_entry or not cache_entry['storyboard_prompts']: continue
                sb_data = cache_entry['storyboard_prompts']
                if sb_data.get('status') != 'success' or not sb_data.get('image_outputs'): continue # Need successful prompts AND image outputs list
                image_outputs = sb_data['image_outputs']
                prompts = sb_data.get('prompts', [])

                 # Ensure prompts and image_outputs lists match length
                if len(prompts) != len(image_outputs):
                    logger.warning(f"Prompt count ({len(prompts)}) and image output count ({len(image_outputs)}) mismatch for sentence {i+1}. Skipping video generation for this sentence.")
                    continue

                # Iterate through successful image outputs
                for p_idx, img_output in enumerate(image_outputs):
                    if not isinstance(img_output, dict) or img_output.get('status') != 'success': continue # Skip if image gen failed or slot isn't a dict
                    image_path = img_output.get('image_path')
                    prompt_text = prompts[p_idx] # Get corresponding prompt
                    if not image_path or not Path(image_path).exists():
                        logger.warning(f"Image path missing or file not found ({image_path}) for sentence {i+1}, prompt {p_idx+1}. Skipping video gen.")
                        continue
                    if not prompt_text:
                         logger.warning(f"Original prompt missing for sentence {i+1}, prompt {p_idx+1}. Skipping video gen.")
                         continue


                    # --- Check Video Cache ---
                    vid_cache_data = img_output.get('video_output')
                    if not force_regenerate and isinstance(vid_cache_data, dict) and vid_cache_data.get('status') == 'success':
                        self.current_run_vid_cache_hits += 1
                        continue

                    # --- Prepare Video Generation Task ---
                    logger.debug(f"VID Prep: Sentence {i+1}, Prompt {p_idx+1}.")
                    output_video_path = self._get_video_output_path(normalized_key, p_idx)
                    task = asyncio.create_task(
                        self.video_gen_handler.generate_video(image_path, prompt_text, output_video_path),
                        name=f"VID_s{i+1}_p{p_idx+1}_{normalized_key[:6]}"
                    )
                    vid_tasks.append(task)
                    vid_task_info[task.get_name()] = {"norm_key": normalized_key, "prompt_idx": p_idx}
                    self.current_run_vid_tasks += 1

            # --- Execute Video Generation tasks concurrently ---
            if vid_tasks:
                logger.info(f"Executing {len(vid_tasks)} Video Generation tasks concurrently (this might take substantial time)...")
                gather_start_time = time.monotonic()
                results = await asyncio.gather(*vid_tasks, return_exceptions=True)
                gather_end_time = time.monotonic()
                self.total_vid_api_wait_time = gather_end_time - gather_start_time
                logger.info(f"Finished waiting for Video tasks. Wait time: {self.total_vid_api_wait_time:.3f}s")

                # --- Process Video Generation Results and Update Cache ---
                for i, result in enumerate(results):
                    task_name = vid_tasks[i].get_name()
                    task_meta = vid_task_info[task_name]
                    norm_key = task_meta["norm_key"]; p_idx = task_meta["prompt_idx"]
                    cache_entry = self.cache[norm_key]

                    # Navigate to the correct image_output slot to store the video result
                    img_output_slot = None
                    try:
                         img_output_slot = cache_entry['storyboard_prompts']['image_outputs'][p_idx]
                         if not isinstance(img_output_slot, dict):
                             logger.error(f"Cache structure error for VID task {task_name}. Image slot is not a dict. Cannot store result.")
                             img_output_slot = None # Prevent attribute error below
                    except (IndexError, KeyError, TypeError) as e:
                         logger.error(f"Cache navigation error for VID task {task_name}: {e}. Cannot store result.")


                    if img_output_slot: # Only update if the slot exists and is a dict
                         if isinstance(result, Exception):
                            self.current_run_vid_failed += 1
                            logger.error(f"VID task {task_name} FAILED: {result}", exc_info=False)
                            img_output_slot['video_output'] = {"status": "failed", "error_message": str(result), "cost": 0.0, "inference_time": 0.0}
                         elif isinstance(result, dict):
                            vid_cost = Decimal(str(result.get('cost', '0.0'))) # Placeholder Cost
                            vid_inf_time = result.get('inference_time', 0.0)

                            if result.get("status") == "success":
                                self.current_run_vid_success += 1
                                img_output_slot['video_output'] = {
                                    "status": "success", "video_path": result.get('video_path'),
                                    "cost": float(vid_cost), # Store as float
                                    "inference_time": vid_inf_time,
                                    "error_message": None
                                }
                                self.current_run_cost_vid += vid_cost # Accumulate placeholder
                                self.current_run_cost_total += vid_cost # Accumulate placeholder
                                self.total_vid_inference_time += vid_inf_time
                            else: # Handled failure from video gen module
                                self.current_run_vid_failed += 1
                                error_msg = result.get('error_message', 'Unknown VID failure')
                                logger.error(f"VID task {task_name} failed internally: {error_msg}")
                                img_output_slot['video_output'] = {
                                    "status": "failed", "cost": float(vid_cost), # Store placeholder cost
                                    "inference_time": vid_inf_time, # Store time even on failure
                                    "error_message": error_msg
                                }
                                self.current_run_cost_vid += vid_cost # Accumulate placeholder
                                self.current_run_cost_total += vid_cost # Accumulate placeholder
                         else: # Unexpected type
                            self.current_run_vid_failed += 1
                            logger.error(f"VID task {task_name} returned unexpected type: {type(result)}")
                            img_output_slot['video_output'] = {"status": "failed", "error_message": f"Unexpected result type: {type(result)}", "cost": 0.0, "inference_time": 0.0}

                self._save_cache()
            else: logger.info("No new Video Generation tasks needed for this run.")


        # --- Future Stages ---
        logger.info("--- Stage: Video/Audio Concatenation (Not Implemented) ---")
        # TODO: Implement FFmpeg logic here
        # - For each sentence:
        #   - Find all successful video paths from cache[norm_key][sb][img_outputs][p_idx][vid_output]
        #   - Create ffmpeg input list file.
        #   - Use ffmpeg -f concat -safe 0 -i list.txt -c copy concatenated_video.mp4
        #   - Find the successful audio path from cache[norm_key][tts_output]
        #   - Use ffmpeg -i concatenated_video.mp4 -i audio.mp3 -c:v copy -c:a aac -shortest final_sentence_clip.mp4
        #   - Store path to final_sentence_clip.mp4 in cache?
        # - After all sentences:
        #   - Concatenate all final_sentence_clip.mp4 files into the final movie.

        # --- Finalize Run ---
        self.run_end_time = time.monotonic()
        logger.info(f"--- Transcript Processing Run Finished ---")
        self.log_run_summary()


    def log_run_summary(self):
        """Logs a summary of the completed processing run."""
        if self.run_start_time is None or self.run_end_time is None:
            logger.warning("Run timing incomplete, cannot log summary.")
            return

        total_run_time = self.run_end_time - self.run_start_time
        total_sb_tokens = self.total_sb_prompt_tokens + self.total_sb_completion_tokens
        total_img_tokens = self.total_img_prompt_tokens + self.total_img_output_tokens

        # Format costs for display
        cost_total_str = f"{self.current_run_cost_total:.6f}"
        cost_tts_str = f"{self.current_run_cost_tts:.4f}"
        cost_sb_str = f"{self.current_run_cost_sb:.6f}"
        cost_img_str = f"{self.current_run_cost_img:.6f}"
        cost_vid_str = f"{self.current_run_cost_vid:.4f} (Placeholder)" # Added Video Cost

        summary_width = 73
        print("\n" + "=" * ((summary_width - 13) // 2) + " Run Summary " + "=" * ((summary_width - 12) // 2))
        print(f"Total Processing Time: {total_run_time:.3f} seconds")
        print(f"Total Estimated Cost (This Run): ${cost_total_str}")
        print("-" * summary_width)
        # --- TTS Summary ---
        print("[Text-to-Speech Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_tts_tasks} / {self.current_run_tts_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_tts_success} / {self.current_run_tts_failed}")
        print(f"  Async Gather Wait Time: {self.total_tts_api_wait_time:.3f}s")
        print(f"  Total API Inference Time (Sum): {self.total_tts_inference_time:.3f}s")
        print(f"  Estimated Cost (This Run): ${cost_tts_str}")
        print("-" * summary_width)
        # --- Storyboard Summary ---
        print("[Sentence-to-Storyboard Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_sb_tasks} / {self.current_run_sb_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_sb_success} / {self.current_run_sb_failed}")
        print(f"  Async Gather Wait Time: {self.total_sb_api_wait_time:.3f}s")
        print(f"  Total LLM Request Time (Sum): {self.total_sb_llm_time:.3f}s")
        print(f"  Total Tokens: {total_sb_tokens} ({self.total_sb_prompt_tokens}p + {self.total_sb_completion_tokens}c)")
        print(f"  Estimated Cost (This Run): ${cost_sb_str}")
        print("-" * summary_width)
        # --- Image Generation Summary ---
        print("[Prompt-to-Image Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_img_tasks} / {self.current_run_img_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_img_success} / {self.current_run_img_failed}")
        print(f"  Async Gather Wait Time: {self.total_img_api_wait_time:.3f}s")
        print(f"  Total API Request Time (Sum): {self.total_img_request_time:.3f}s")
        print(f"  Total Tokens Est: {total_img_tokens} ({self.total_img_prompt_tokens}p + {self.total_img_output_tokens}o)")
        print(f"  Estimated Cost (This Run): ${cost_img_str}")
        print("-" * summary_width)
        # --- Video Generation Summary --- # ADDED
        print("[Image-to-Video Stage]")
        print(f"  Tasks Executed / Cache Hits: {self.current_run_vid_tasks} / {self.current_run_vid_cache_hits}")
        print(f"  Successful / Failed: {self.current_run_vid_success} / {self.current_run_vid_failed}")
        print(f"  Async Gather Wait Time: {self.total_vid_api_wait_time:.3f}s")
        print(f"  Total API Inference Time (Sum): {self.total_vid_inference_time:.3f}s")
        print(f"  Estimated Cost (This Run): ${cost_vid_str}")
        print("-" * summary_width)
        # --- Other Stages ---
        print("[Other Stages]")
        print("  Video/Audio Assembly: Not Yet Implemented")
        print("=" * summary_width + "\n")

        # Log lifetime costs from handlers
        try:
            handler_tts_total_cost = self.tts_handler.get_total_estimated_cost()
            handler_sb_stats = self.storyboard_handler.get_stats(); handler_sb_total_cost = handler_sb_stats.get('total_cost', Decimal('0.0'))
            handler_img_stats = self.image_gen_handler.get_stats(); handler_img_total_cost = handler_img_stats.get('total_estimated_cost', Decimal('0.0'))
            handler_vid_stats = self.video_gen_handler.get_stats(); handler_vid_total_cost = handler_vid_stats.get('total_estimated_cost', Decimal('0.0')) # Added Video

            logger.info(f"Lifetime Handler Costs Est: TTS=${handler_tts_total_cost:.4f}, SB=${handler_sb_total_cost:.6f}, IMG=${handler_img_total_cost:.6f}, VID=${handler_vid_total_cost:.4f} (Placeholder)")
        except Exception as e: logger.warning(f"Could not retrieve lifetime stats from handlers: {e}")


# --- Main Execution Block ---
async def run_orchestrator(args):
    """Initializes engine and runs the orchestrator."""
    or_engine = None
    try:
        print("Initializing OpenRouter Engine...")
        or_engine = OpenRouterEngine(app_title="StoryShot_MainRun", app_url="local://storyshot")
        await or_engine.wait_for_initialization()
        if not or_engine.is_available():
             print("OpenRouter Engine initialization failed. Aborting.", file=sys.stderr); return
        print("OpenRouter Engine Initialized.")

        orchestrator = StoryShotOrchestrator(or_engine)
        transcript = orchestrator.load_transcript(args.transcript_file)
        await orchestrator.process_transcript(transcript, args.force_regenerate, args.skip_images, args.skip_videos) # Pass skip_videos

    except FileNotFoundError: sys.exit(1)
    except (TextToSpeechError, SentenceToStoryboardError, OpenRouterError, PromptToImageError, ImageToVideoError) as e: # Added ImageToVideoError
         logger.error(f"A critical processing error occurred in {type(e).__name__}: {e}", exc_info=False); sys.exit(1)
    except Exception as e: logger.error(f"An unexpected error occurred during orchestration: {e}", exc_info=True); sys.exit(1)
    finally:
         if or_engine:
             print("Closing OpenRouter Engine client..."); await or_engine.close(); print("Engine client closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StoryShot: Transcript to Storyboard/Images/Videos MVP")
    parser.add_argument("transcript_file", type=str, help="Path to the input transcript text file.")
    parser.add_argument( "-f", "--force-regenerate", action="store_true", help="Force regeneration of all steps, ignoring cache.")
    parser.add_argument("--skip-images", action="store_true", help="Skip the prompt-to-image generation stage.")
    parser.add_argument("--skip-videos", action="store_true", help="Skip the image-to-video generation stage.") # Added
    args = parser.parse_args()

    # --- Pre-computation Checks ---
    print("--- Running Pre-computation Checks ---")
    checks_passed = True
    if not Path(".env").exists(): print("Error: `.env` file missing."); checks_passed = False
    else:
         if not os.getenv("FAL_KEY") or ':' not in os.getenv("FAL_KEY", ""): print("Error: `FAL_KEY` missing or format incorrect (key_id:key_secret)."); checks_passed = False
         if not os.getenv("OPENROUTER_KEY"): print("Error: `OPENROUTER_KEY` missing."); checks_passed = False
         if not os.getenv("OPENAI_KEY"): print("Error: `OPENAI_KEY` missing."); checks_passed = False

    transcript_path = Path(args.transcript_file)
    if not transcript_path.is_file():
         print(f"Error: Transcript file not found at '{args.transcript_file}'")
         if transcript_path.name.lower() in ["sample_transcript.txt", "transcript.txt"]:
             print("Attempting to create dummy transcript...")
             try:
                 with open(transcript_path.name, "w", encoding='utf-8') as f:
                     f.write("This is the first sentence for the dummy transcript.\nThis is the second sentence, designed to be slightly longer.\nAnd a final, third sentence!")
                 print(f"Created dummy '{transcript_path.name}'. Please edit it with real content.")
                 checks_passed = True # Allow continuing with dummy file
             except Exception as e: print(f"Failed to create dummy file: {e}"); checks_passed = False
         else: checks_passed = False

    try: # Check directories
        CACHE_DIR.mkdir(exist_ok=True); TEMP_DIR.mkdir(exist_ok=True)
        AUDIO_SUBDIR.mkdir(exist_ok=True); IMAGES_SUBDIR.mkdir(exist_ok=True); VIDEOS_SUBDIR.mkdir(exist_ok=True) # Added Videos dir check
    except OSError as e: print(f"Error creating directories: {e}"); checks_passed = False

    if not checks_passed:
        print("--- Pre-computation checks failed. Please resolve errors. ---"); sys.exit(1)
    else: print("--- Pre-computation checks passed. ---")

    # --- Run Async ---
    if sys.platform == "win32" and sys.version_info >= (3, 8):
       asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(run_orchestrator(args))
    except KeyboardInterrupt: print("\nOrchestration interrupted by user.")
    except RuntimeError as e:
        if "Event loop is closed" in str(e) or "cannot schedule new futures" in str(e): print("\nError during async shutdown.", file=sys.stderr)
        else: logger.error(f"RuntimeError: {e}", exc_info=True)