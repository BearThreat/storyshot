# File: image2video_.py
"""
Image-to-Video Module using Fal.ai (image2video_.py) - Multi-Model Support V6

Handles generating short video clips from a starting image and a text prompt using
different Fal.ai image-to-video APIs, currently supporting:
- fal-ai/kling-video/v1.6/standard/image-to-video (Default)
- fal-ai/wan-i2v

Includes:
- Basic retry logic for image upload and API calls.
- Aspect ratio detection for local images when using Kling.
- Log fetching during generation using iter_events.
- Uses handler.get() after iter_events to reliably fetch final results.
- Estimated cost calculation.
- **Improved download logic using aiohttp.iter_chunked.**
- **Added final file size check after download.**
- **Corrected SyntaxError and removed check for non-existent fal_client.Failed.**

... [Rest of the header comments remain largely the same] ...

Key Changes in V6:
- Fixed SyntaxError in file deletion try/except block.
- Removed check for non-existent fal_client.Failed event.
- Switched download loop to use response.content.iter_chunked() for robustness.
- Added logging of final file size after download completes.
- Added logging if Content-Length header is missing during download check.
"""

import os
import sys
import logging
import time
import asyncio
import math
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional

from dotenv import load_dotenv
import fal_client
import aiohttp
import httpx
try:
    from PIL import Image # For aspect ratio detection
except ImportError:
    print("Pillow not found. Please install it: pip install Pillow")
    Image = None # Set to None if import fails

# --- Configuration ---
load_dotenv()
FAL_KEY = os.getenv("FAL_KEY")

# --- Model IDs ---
KLING_MODEL_ID = "fal-ai/kling-video/v1.6/standard/image-to-video"
WAN_I2V_MODEL_ID = "fal-ai/wan-i2v"
DEFAULT_VIDEO_MODEL = KLING_MODEL_ID
FALAI_VIDEO_MODEL = os.getenv("FALAI_VIDEO_MODEL", DEFAULT_VIDEO_MODEL)

DEFAULT_API_RETRY_COUNT = 1
DEFAULT_UPLOAD_RETRY_COUNT = 1
DEFAULT_UPLOAD_RETRY_DELAY = 2
DEFAULT_TIMEOUT_SECONDS = 300

# --- Model Specific Default Settings ---
KLING_DEFAULT_DURATION_SEC = 5; KLING_DEFAULT_ASPECT_RATIO = "16:9"; KLING_ALLOWED_ASPECT_RATIOS = ["16:9", "1:1", "9:16"]; KLING_DEFAULT_CFG = 0.5
WAN_TARGET_VIDEO_DURATION_DEFAULT = 2.0; WAN_MIN_FRAMES = 81; WAN_MAX_FRAMES = 100; WAN_DEFAULT_FPS = 16; WAN_DEFAULT_RESOLUTION = "480p"; WAN_DEFAULT_GUIDE_SCALE = 5.0

# --- Estimated Cost Configuration ---
WAN_COST_BASE_480P = Decimal("0.20"); WAN_COST_BASE_720P = Decimal("0.40"); WAN_FRAME_MULTIPLIER_THRESHOLD = 81; WAN_FRAME_MULTIPLIER_FACTOR = Decimal("1.25")
KLING_COST_PER_SECOND = Decimal("0.045")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fal_client").setLevel(logging.WARNING)

class ImageToVideoError(Exception):
    """Custom exception for ImageToVideoFal errors."""
    pass

class ImageToVideoFal:
    """
    Handles image-to-video generation using Fal.ai (Async, Multi-Model). V6
    Tracks timings and estimated costs. Includes upload retry & aspect ratio detection.
    Uses iter_events for logs and handler.get() for final results. Improves download handling.
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = FALAI_VIDEO_MODEL,
                 api_retries: int = DEFAULT_API_RETRY_COUNT,
                 upload_retries: int = DEFAULT_UPLOAD_RETRY_COUNT):
        # (init remains the same)
        self.api_key = api_key or FAL_KEY
        if not self.api_key or ':' not in self.api_key: raise ValueError("Fal API key error (missing or bad format 'key_id:key_secret')")
        self.model = model; self.api_retries = api_retries; self.upload_retries = upload_retries
        self.total_requests = 0; self.total_successful_requests = 0; self.total_failed_requests = 0
        self.total_inference_time_sum = 0.0; self.total_estimated_cost = Decimal("0.0")
        logger.info(f"ImageToVideoFal initialized. Model: {self.model}. Upload Retries: {self.upload_retries}. Async.")
        logger.warning(f"Using ESTIMATED pricing. Actual costs may differ.")
        if self.model == WAN_I2V_MODEL_ID: logger.info(f"Wan Settings: Min Frames={WAN_MIN_FRAMES}, Default Res={WAN_DEFAULT_RESOLUTION}")
        elif self.model == KLING_MODEL_ID: logger.info(f"Kling Settings: Default Duration={KLING_DEFAULT_DURATION_SEC}s, Aspect Ratio Detection (Local Files): {'Enabled' if Image else 'Disabled (Pillow not found)'}")

    def _calculate_cost(self, model_id: str, **kwargs) -> Decimal:
        # (remains the same)
        cost = Decimal("0.0")
        try:
            if model_id == WAN_I2V_MODEL_ID: num_frames = kwargs.get('num_frames'); resolution = kwargs.get('resolution').lower(); base_cost = WAN_COST_BASE_720P if resolution == "720p" else WAN_COST_BASE_480P; multiplier = WAN_FRAME_MULTIPLIER_FACTOR if num_frames > WAN_FRAME_MULTIPLIER_THRESHOLD else Decimal("1.0"); cost = base_cost * multiplier
            elif model_id == KLING_MODEL_ID: duration_str = kwargs.get('duration'); duration_sec = int(duration_str); cost = Decimal(str(duration_sec)) * KLING_COST_PER_SECOND
            else: logger.warning(f"Cost calculation not implemented for model: {model_id}")
        except Exception as e: logger.error(f"Cost calculation error: {e}", exc_info=True); cost = Decimal("0.0")
        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    async def _upload_image_if_needed(self, image_path_or_url: str | Path) -> str:
        # (remains the same)
        path = Path(image_path_or_url)
        if path.is_file():
            upload_attempts = 0
            while upload_attempts <= self.upload_retries:
                try:
                    logger.debug(f"Uploading local image: {path} (Attempt {upload_attempts + 1}/{self.upload_retries + 1})")
                    upload_start_time = time.monotonic(); uploaded_url = await fal_client.upload_file_async(str(path)); upload_duration = time.monotonic() - upload_start_time
                    logger.info(f"Image uploaded successfully to {uploaded_url} ({upload_duration:.3f}s)")
                    return uploaded_url
                except (httpx.TransportError, fal_client.RequestError, Exception) as e: # Added fal_client.RequestError explicitly
                    logger.warning(f"Upload attempt {upload_attempts + 1} failed: {type(e).__name__}: {e}")
                    upload_attempts += 1
                    if upload_attempts <= self.upload_retries: logger.info(f"Retrying upload in {DEFAULT_UPLOAD_RETRY_DELAY}s..."); await asyncio.sleep(DEFAULT_UPLOAD_RETRY_DELAY)
                    else: logger.error(f"Upload failed after {self.upload_retries + 1} attempts."); raise ImageToVideoError(f"Failed to upload image after retries: {e}") from e
        elif isinstance(image_path_or_url, str) and image_path_or_url.startswith(('http://', 'https://')): return image_path_or_url
        else: raise ImageToVideoError(f"Invalid image input: '{image_path_or_url}'. Needs path or URL.")

    def _detect_aspect_ratio(self, image_path: Path) -> Optional[str]:
        # (remains the same)
        if not Image: return None
        try:
            with Image.open(image_path) as img: width, height = img.size
            if width <= 0 or height <= 0: return None
            ratio = width / height; target_ratios = { "16:9": 16/9, "1:1": 1.0, "9:16": 9/16 }
            closest_ratio_str = min(target_ratios, key=lambda k: abs(target_ratios[k] - ratio))
            logger.info(f"Detected aspect ratio {ratio:.3f} for {image_path.name}, closest standard: {closest_ratio_str}")
            return closest_ratio_str # Always return the closest one detected
        except Exception as e: logger.warning(f"Could not detect aspect ratio for {image_path}: {e}", exc_info=False); return None


    async def generate_video(
        self,
        image_path_or_url: str | Path,
        prompt: str,
        output_video_path: str | Path,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        vid_len: Optional[float] = None,
        fps: int = WAN_DEFAULT_FPS,
        resolution: str = WAN_DEFAULT_RESOLUTION,
        guide_scale: float = WAN_DEFAULT_GUIDE_SCALE,
        kling_duration_sec: int = KLING_DEFAULT_DURATION_SEC,
        cfg_scale: float = KLING_DEFAULT_CFG,
    ) -> Dict[str, Any]:
        """
        Generates a video using the configured model, with aspect ratio detection for Kling. V6
        Uses iter_events for logs and handler.get() for final results. Improved download.
        """
        self.total_requests += 1
        start_time = time.monotonic()
        output_path = Path(output_video_path)
        log_prefix = f"Video Gen ({self.model.split('/')[-1]} -> {output_path.name})"
        logger.info(f"{log_prefix}: Requesting video for prompt: \"{prompt[:60]}...\"")

        result_dict = {"status": "failed", "video_path": None, "inference_time": 0.0, "cost": 0.0, "error_message": "Unknown failure."}
        estimated_cost = Decimal("0.0"); arguments = {}; cost_params = {}

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image_url = await self._upload_image_if_needed(image_path_or_url)

            arguments = { "prompt": prompt, "image_url": image_url }
            detected_aspect_ratio = None
            is_local_file = isinstance(image_path_or_url, Path) and image_path_or_url.is_file()

            # --- Build Arguments based on Model ---
            # (Arg building logic remains same)
            if self.model == WAN_I2V_MODEL_ID:
                target_duration = vid_len if vid_len is not None and vid_len > 0 else WAN_TARGET_VIDEO_DURATION_DEFAULT
                num_frames = math.ceil(target_duration * fps); num_frames = max(WAN_MIN_FRAMES, min(num_frames, WAN_MAX_FRAMES))
                arguments.update({"num_frames": num_frames, "frames_per_second": fps, "resolution": resolution, "guide_scale": guide_scale})
                if negative_prompt: arguments["negative_prompt"] = negative_prompt 
                else: arguments["negative_prompt"] = "worst quality, low quality, blur, text, words"
                cost_params = {"num_frames": num_frames, "resolution": resolution}; logger.info(f"{log_prefix}: Wan Params: frames={num_frames}, res={resolution}")
            elif self.model == KLING_MODEL_ID:
                logger.debug(f"{log_prefix}: Preparing args for Kling")
                duration_str = str(kling_duration_sec)
                if duration_str not in ["5", "10"]: logger.warning(f"{log_prefix}: Invalid duration {kling_duration_sec}, using default {KLING_DEFAULT_DURATION_SEC}s."); duration_str = str(KLING_DEFAULT_DURATION_SEC)
                aspect_ratio_to_use = KLING_DEFAULT_ASPECT_RATIO
                if is_local_file and Image:
                     detected_aspect_ratio = self._detect_aspect_ratio(Path(image_path_or_url))
                     if detected_aspect_ratio in KLING_ALLOWED_ASPECT_RATIOS: aspect_ratio_to_use = detected_aspect_ratio
                     else: logger.warning(f"{log_prefix}: Could not determine valid aspect ratio for {Path(image_path_or_url).name}, using default: {aspect_ratio_to_use}")
                elif not is_local_file: logger.warning(f"{log_prefix}: Input is URL, cannot detect aspect ratio. Using default: {aspect_ratio_to_use}")
                arguments.update({"duration": duration_str, "aspect_ratio": aspect_ratio_to_use, "cfg_scale": cfg_scale})
                if negative_prompt: arguments["negative_prompt"] = negative_prompt 
                else: arguments["negative_prompt"] = "blur, distortion, noise, low quality, text"
                cost_params = {"duration": duration_str}; logger.info(f"{log_prefix}: Kling Params: duration={duration_str}s, aspect_ratio={aspect_ratio_to_use}")
            else:
                raise ImageToVideoError(f"Unsupported model ID: {self.model}")
            if seed is not None: arguments["seed"] = seed

            estimated_cost = self._calculate_cost(self.model, **cost_params)
            logger.info(f"{log_prefix}: Estimated cost: ${estimated_cost:.6f}"); result_dict["cost"] = float(estimated_cost)

            # --- API Call with Retries ---
            attempts = 0; last_exception = None; handler = None
            while attempts <= self.api_retries:
                inference_duration = 0.0; response_data = None; fal_logs = []
                get_result_start_time = time.monotonic() # Start timing before submit/events

                try:
                    logger.debug(f"{log_prefix}: Fal Submit/Check attempt {attempts + 1}/{self.api_retries + 1}.")
                    if not handler: # Only submit on the first attempt
                         submit_start_time = time.monotonic()
                         handler = await fal_client.submit_async(self.model, arguments=arguments)
                         submit_duration = time.monotonic() - submit_start_time
                         logger.debug(f"{log_prefix}: Submitted in {submit_duration:.3f}s. Waiting for result...")
                    else:
                         logger.debug(f"{log_prefix}: Using existing handler for retry attempt {attempts + 1}.")

                    # --- Use iter_events ONLY for logs and intermediate status ---
                    logger.debug(f"{log_prefix}: Monitoring events (for logs)...")
                    try:
                        async for event in handler.iter_events(with_logs=True, logs_interval=5): # Check logs every 5s
                            if isinstance(event, fal_client.Queued): logger.debug(f"{log_prefix}: Request Queued (Position: {event.position})")
                            elif isinstance(event, fal_client.InProgress):
                                if event.logs:
                                     for log in event.logs: logger.debug(f"{log_prefix} [FAL LOG]: {log.get('message', '')}")
                                     fal_logs.extend(event.logs)
                            elif isinstance(event, fal_client.Completed): # Removed check for non-existent Failed
                                logger.debug(f"{log_prefix}: Received completion event indicator.")
                                break # Stop iterating events once complete status is seen
                    except Exception as stream_exc:
                         logger.warning(f"{log_prefix}: Error during Fal event streaming: {type(stream_exc).__name__}: {stream_exc}. Will still attempt handler.get().", exc_info=False)
                         last_exception = stream_exc

                    logger.debug(f"{log_prefix}: Event monitoring finished. Attempting handler.get() for final result.")

                    # --- Use handler.get() to fetch the final result / check for errors ---
                    try:
                        response_data = await handler.get() # Blocks until completion
                        get_result_end_time = time.monotonic()
                        inference_duration = get_result_end_time - get_result_start_time
                        logger.debug(f"{log_prefix}: handler.get() successful. Total wait approx: {inference_duration:.3f}s")
                        logger.debug(f"{log_prefix}: Raw response data from handler.get(): {response_data}")
                    except fal_client.client.FalClientError as get_exc: #### Changed from generic RequestError
                        get_result_end_time = time.monotonic()
                        inference_duration = get_result_end_time - get_result_start_time
                        logger.error(f"{log_prefix}: Fal job failed (reported by handler.get()): {get_exc}. Duration: {inference_duration:.3f}s")
                        result_dict["error_message"] = f"Fal job failed: {get_exc}"; result_dict["inference_time"] = inference_duration; self.total_failed_requests += 1
                        if output_path.exists():
                            try:
                                output_path.unlink()
                                logger.debug(f"{log_prefix}: Removed potentially incomplete file {output_path}")
                            except OSError as e:
                                logger.warning(f"{log_prefix}: Failed to remove incomplete file {output_path}: {e}") # Fixed SyntaxError here
                        return result_dict # Exit on definitive job failure

                    # --- Validate Response Data ---
                    if not isinstance(response_data, dict) or "video" not in response_data or not isinstance(response_data["video"], dict) or "url" not in response_data["video"]:
                        logger.error(f"{log_prefix}: Unexpected API response format after handler.get(). Response: {response_data}.")
                        raise ImageToVideoError(f"Unexpected API response format (missing video URL).")

                    video_url = response_data["video"]["url"]

                    # --- Download with iter_chunked ---
                    download_start_time = time.monotonic()
                    logger.debug(f"{log_prefix}: Downloading video from {video_url}...")
                    bytes_written = 0
                    try:
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                            async with session.get(video_url) as dl_response:
                                dl_response.raise_for_status()
                                content_length = dl_response.headers.get('Content-Length')
                                if content_length:
                                     logger.debug(f"{log_prefix}: Video Content-Length: {content_length} bytes")
                                     if int(content_length) < 2000 :
                                          logger.error(f"{log_prefix}: Downloaded file size header ({content_length} bytes) is suspiciously small. Assuming generation failure.")
                                          raise ImageToVideoError(f"Downloaded file size header ({content_length} bytes) indicates generation failure.")
                                else:
                                     logger.warning(f"{log_prefix}: Content-Length header missing from download response.")

                                with open(output_path, 'wb') as f:
                                     # Use iter_chunked for reliable download streaming
                                     async for chunk in dl_response.content.iter_chunked(1024 * 1024): # 1MB chunks
                                         f.write(chunk)
                                         bytes_written += len(chunk)

                        # --- Final File Size Check ---
                        if output_path.exists():
                            final_size = output_path.stat().st_size
                            logger.info(f"{log_prefix}: Final file size on disk: {final_size} bytes (wrote {bytes_written}).")
                            if final_size < 2000: # Apply heuristic again to actual size
                                 logger.error(f"{log_prefix}: Final saved file size ({final_size} bytes) is suspiciously small. Raising error.")
                                 raise ImageToVideoError(f"Saved file size ({final_size} bytes) indicates generation failure.")
                        else:
                             logger.error(f"{log_prefix}: File not found after download completed: {output_path}")
                             raise ImageToVideoError("File missing after seemingly successful download.")

                    except asyncio.TimeoutError:
                        raise ImageToVideoError("Timeout during video download.")
                    except aiohttp.ClientError as dl_exc:
                        raise ImageToVideoError(f"HTTP error during download: {dl_exc}") from dl_exc

                    download_duration = time.monotonic() - download_start_time
                    logger.info(f"{log_prefix}: Video successfully downloaded ({download_duration:.3f}s) and saved to {output_path}")

                    # Success!
                    self.total_successful_requests += 1; self.total_inference_time_sum += inference_duration; self.total_estimated_cost += estimated_cost
                    result_dict.update({"status": "success", "video_path": str(output_path), "inference_time": inference_duration, "error_message": None})
                    return result_dict # Exit retry loop on success

                # --- Error Handling within the retry loop ---
                except fal_client.client.FalClientError as e: # If submit fails etc.
                     error_message = str(e); logger.warning(f"{log_prefix}: Fal Client Error (Attempt {attempts + 1}): {error_message}.")
                     is_non_retryable = "value_error" in error_message.lower() or "validation_error" in error_message.lower() or ("status_code" in dir(e) and 400 <= e.status_code < 500)
                     if is_non_retryable:
                         logger.error(f"{log_prefix}: Non-retryable client error."); result_dict["error_message"] = f"Fal Client Error: {error_message}"; self.total_failed_requests += 1
                         if output_path.exists(): 
                             try: output_path.unlink() 
                             except OSError: pass # Ensure cleanup attempt
                         return result_dict
                     else: last_exception = e; logger.info(f"{log_prefix}: Potentially retryable Fal client error. Will retry."); handler = None # Reset handler only if submit likely failed

                except (aiohttp.ClientError, httpx.TransportError, asyncio.TimeoutError) as e:
                    last_exception = e; logger.warning(f"{log_prefix}: Network/Timeout Error (Attempt {attempts + 1}): {type(e).__name__}. Retrying...")

                except ImageToVideoError as e: # Our specific non-retryable errors
                     last_exception = e; logger.error(f"{log_prefix}: Processing error (Attempt {attempts + 1}): {e}")
                     result_dict["error_message"] = str(e); result_dict["inference_time"] = inference_duration; self.total_failed_requests += 1
                     if output_path.exists(): 
                         try: output_path.unlink() 
                         except OSError: pass # Ensure cleanup attempt
                     return result_dict # Don't retry

                except Exception as e: # Catch-all
                      last_exception = e; error_type_name = type(e).__name__; logger.error(f"{log_prefix}: Unexpected error (Attempt {attempts + 1}) [{error_type_name}]: {e}", exc_info=True); logger.warning(f"{log_prefix}: Retrying unexpected...")

                # -- Retry --
                attempts += 1
                if attempts <= self.api_retries:
                    wait_time = 2**(attempts); logger.info(f"{log_prefix}: Retrying in {wait_time} seconds..."); await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{log_prefix}: Max API retries reached."); self.total_failed_requests += 1
                    if last_exception: result_dict["error_message"] = f"Failed after {self.api_retries+1} attempts. Last: {type(last_exception).__name__}: {last_exception}"
                    else: result_dict["error_message"] = f"Failed after {self.api_retries+1} attempts."
                    result_dict["inference_time"] = time.monotonic() - get_result_start_time
                    if output_path.exists(): 
                        try: output_path.unlink() 
                        except OSError: pass # Ensure cleanup attempt
                    return result_dict

        # --- Error Handling outside the retry loop (Config, Upload etc) ---
        except (ImageToVideoError, ValueError) as e:
            logger.error(f"{log_prefix}: Pre-API phase error: {e}", exc_info=False)
            result_dict["error_message"] = str(e); self.total_failed_requests += 1
            return result_dict
        except Exception as e:
            logger.error(f"{log_prefix}: Unexpected Pre-API phase error: {e}", exc_info=True)
            result_dict["error_message"] = f"Unexpected Pre-API error: {e}"; self.total_failed_requests += 1
            return result_dict


    def get_stats(self) -> Dict[str, Any]:
        # (remain the same)
        avg_inf_time = (self.total_inference_time_sum / self.total_successful_requests) if self.total_successful_requests > 0 else 0.0
        return { "total_requests": self.total_requests, "successful": self.total_successful_requests, "failed": self.total_failed_requests, "total_estimated_cost": float(self.total_estimated_cost), "total_inference_time_sum": self.total_inference_time_sum, "average_inference_time_success": avg_inf_time }

# --- Main Execution Block (for testing) ---
async def run_video_gen_task(handler, image_input, prompt, output_path, task_id, **kwargs):
    # (remains the same)
    model_name = handler.model.split('/')[-1]
    logger.info(f"--- Task {task_id} ({model_name}): Starting generation ---")
    result = await handler.generate_video(image_input, prompt, output_path, seed=12345 + task_id, **kwargs)
    logger.info(f"--- Task {task_id}: Finished '{result.get('status', 'unknown')}' ---") # Add status to finish log
    return result

async def main_test():
    print("--- Running ImageToVideoFal Multi-Model Test V6 ---") # Updated name
    if not FAL_KEY or ':' not in FAL_KEY: print("Error: FAL_KEY invalid.", file=sys.stderr); return
    if not Image: print("Error: Pillow is not installed. Aspect ratio detection disabled.", file=sys.stderr)

    # --- Test Inputs ---
    local_image_path_panda = Path("temp_files/image_test_concurrent/test_image_concurrent_1.png")
    kling_doc_image_url = "https://storage.googleapis.com/falserverless/kling/kling_input.jpeg"
    kling_doc_prompt = "Snowflakes fall as a car moves forward along the road."
    if local_image_path_panda.exists():
        print(f"Using local image: {local_image_path_panda}")
        test_image_input_1 = local_image_path_panda
        test_prompt_1 = "The red panda looks directly at the camera, subtle steam rises from its cup."
    else:
        print(f"Warning: Local image {local_image_path_panda} not found. Using Kling doc example URL for Task 1.")
        test_image_input_1 = kling_doc_image_url
        test_prompt_1 = kling_doc_prompt
    test_image_input_2 = kling_doc_image_url
    test_prompt_2 = kling_doc_prompt
    test_cases = [{"image": test_image_input_1, "prompt": test_prompt_1}, {"image": test_image_input_2, "prompt": test_prompt_2}]
    num_tasks = len(test_cases)
    test_output_dir = Path("temp_files") / "video_test_multi_v6" # Changed dir name
    test_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output videos will be saved to: {test_output_dir}")

    # --- Test Configuration ---
    model_to_test = KLING_MODEL_ID

    video_handler = None
    try:
        video_handler = ImageToVideoFal(model=model_to_test, api_retries=1)
        print(f"\n--- Testing Model: {video_handler.model} ---")
        model_specific_args = {}
        if video_handler.model == KLING_MODEL_ID: model_specific_args = {"kling_duration_sec": 5}; print(f"Kling Test Args: duration={model_specific_args['kling_duration_sec']}s")
        elif video_handler.model == WAN_I2V_MODEL_ID: target_len = 2.0; model_specific_args = {"vid_len": target_len, "resolution": "480p"}; print(f"Wan Test Args: target_len={model_specific_args['vid_len']}s, res={model_specific_args['resolution']}")

        # --- Create Tasks ---
        tasks = []; print(f"\nCreating {num_tasks} video generation tasks...")
        for i, case in enumerate(test_cases):
            filename = f"test_{video_handler.model.split('/')[-1]}_task{i+1}.mp4"
            output_path = test_output_dir / filename
            task = asyncio.create_task( run_video_gen_task(handler=video_handler, image_input=case["image"], prompt=case["prompt"], output_path=output_path, task_id=i+1, **model_specific_args), name=f"{video_handler.model.split('/')[-1]}_Task_{i+1}")
            tasks.append(task)

        print(f"Waiting for {num_tasks} tasks..."); concurrent_start_time = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True); concurrent_end_time = time.monotonic()
        total_concurrent_duration = concurrent_end_time - concurrent_start_time; print(f"--- Concurrent execution finished in {total_concurrent_duration:.3f} seconds ---")

        # --- Process Results ---
        print("\n--- Individual Task Results ---"); successful_tasks=0; total_inference_time_sum_success=0.0; total_cost_sum_success = Decimal("0.0"); failed_task_details = []
        for i, res in enumerate(results):
            print("-" * 20); task_name = tasks[i].get_name()
            if isinstance(res, Exception): logger.error(f"Task {task_name} failed catastrophically: {res}", exc_info=res); print(f"Task {i+1} ({task_name}): CATASTROPHIC FAILURE\n Error: {res}"); failed_task_details.append(f"Task {i+1}: Catastrophic Failure - {res}")
            elif isinstance(res, dict):
                print(f"Task {i+1} ({task_name}) - Input: {os.path.basename(str(test_cases[i]['image']))}"); print(f"  Status:   {res['status']}")
                inf_time = res['inference_time']; cost = Decimal(str(res['cost']))
                print(f"  Inference Wait: {inf_time:.3f}s"); print(f"  Cost Est: ${cost:.6f}")
                if res['status'] == "success":
                    successful_tasks += 1; total_inference_time_sum_success += inf_time; total_cost_sum_success += cost
                    video_path = res.get('video_path')
                    print(f"  Video:    {video_path}")
                    # Add file size check in results display
                    try: final_size = Path(video_path).stat().st_size if video_path else -1; print(f"  File Size:{final_size} bytes")
                    except FileNotFoundError: print(f"  File Size: Not Found!")
                    except Exception as e: print(f"  File Size: Error checking size - {e}")
                else:
                    print(f"  Error:    {res['error_message']}")
                    failed_task_details.append(f"Task {i+1}: Status='failed' - {res.get('error_message', 'Unknown')}")
            else: print(f"Task ID {i+1}: UNEXPECTED RESULT TYPE: {type(res)}"); failed_task_details.append(f"Task {i+1}: Unexpected result type {type(res)}")

        # --- Final Summary ---
        print("\n" + "="*15 + f" Video Gen Test Summary V6 ({video_handler.model}) " + "="*15) # V6
        print(f"Total Wall-Clock Time (Batch): {total_concurrent_duration:.3f} seconds"); print(f"Tasks Attempted / Successful: {len(tasks)} / {successful_tasks}")
        if failed_task_details: print("Failed Tasks Details:"); [print(f"  - {d}") for d in failed_task_details[:5]]; print("  ...") if len(failed_task_details)>5 else None
        avg_inference_time = (total_inference_time_sum_success / successful_tasks) if successful_tasks > 0 else 0.0
        print(f"Average Inference Wait Time (Successful): {avg_inference_time:.3f}s"); print(f"Total Estimated Cost (Successful Sum): ${total_cost_sum_success:.6f}")
        print("-" * 76); print("Handler Internal Metrics:"); handler_stats = video_handler.get_stats()
        print(f" Handler Total Req: {handler_stats['total_requests']} (S:{handler_stats['successful']}, F:{handler_stats['failed']}), Total Cost Est: ${handler_stats['total_estimated_cost']:.6f}, Avg Inf Time (S): {handler_stats['average_inference_time_success']:.3f}s")
        print("=" * 76)

    except ImageToVideoError as e: print(f"\nImageToVideo Error: {e}", file=sys.stderr)
    except ValueError as e: print(f"\nConfiguration Error: {e}", file=sys.stderr)
    except Exception as e: logger.error(f"Unexpected test error: {e}", exc_info=True)
    finally: pass

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8): asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main_test())
    except KeyboardInterrupt: print("\nTest interrupted.")

