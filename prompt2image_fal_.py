# File: prompt2image_fal_.py
"""
Prompt-to-Image Module using Fal.ai (prompt2image_fal_.py) - V2 (Async Fix)

Handles generating images from text prompts using a Fal.ai endpoint,
hypothetically "fal-ai/gpt-image-1/text-to-image". This implies Fal is
providing access to OpenAI's gpt-image-1 technology.

*** RATE LIMIT NOTE ***
Fal.ai will have its own rate limits for any given model endpoint.
The orchestrator may need to implement throttling (e.g., using asyncio.Semaphore
or delays between batches) to avoid hitting this limit. A placeholder rate limit
is used in the test section.

API Info (Fal.ai - assuming gpt-image-1 endpoint):
- Docs: Check Fal.ai documentation for the specific model ID.
- Authentication: FAL_KEY environment variable.
- Client Library: `fal-ai` (pip install fal-ai)
- Model: Specified by `FALAI_IMAGE_MODEL` env var (e.g., "fal-ai/gpt-image-1/text-to-image").
- Endpoint: Managed by the `fal-ai` library via `fal.submit_async()` and `handler.get()`.
- Key Input Parameters (Assumed based on gpt-image-1):
    - `prompt`: The text description for the image.
    - `image_size`: Image dimensions (e.g., "1024x1024", "auto").
    - `quality`: "low", "medium", "high", "auto".
    - `background`: "transparent", "opaque", "auto".
    - `num_images`: Defaults to 1 (fixed at 1 in this script).
- Key Output (from Fal.ai API via library):
    - Typically a dictionary containing `images`: a list of objects.
    - Each image object might have `url` and/or `b64_json`.
    - May include `revised_prompt`.
    - Expected to include `timings: {'inference_time': ...}` for cost calculation.
- Cost: Assumed to be based on inference time * price_per_second for the specific Fal model.
  This script uses a placeholder `FAL_GPT_IMAGE_1_PRICE_PER_SECOND`.

Workflow:
- Input: Text prompt, output path, image generation parameters.
- Action:
    1. Call Fal.ai API (`fal.submit_async` then `handler.get`) with the prompt and parameters.
    2. If successful, retrieve image data (b64 or download from URL).
    3. Save image to the specified path.
    4. Calculate cost based on `inference_time` from the response.
- Output: Dictionary with status, image path, cost, request time, etc.

Key Features & Outputs:
- Async Ready: Uses `async`/`await` and the `fal-ai` async capabilities.
- Cost Tracking: Estimates cost per request based on inference time. Maintains a running total.
- Timing Tracking: Measures *individual successful API call* duration.
- Result Dictionary: The `generate_image` method returns a dictionary:
    {
        "status": "success" | "failed",
        "image_path": str | None,
        "cost": float,
        "request_time": float, # Duration of this specific successful API call (handler.get wait)
        "revised_prompt": str | None,
        "inference_time_sec": float, # Inference time reported by Fal
        "error_message": str | None
    }
- Retries: Implements basic retry logic for transient errors.

Usage Notes / Integration with storyshot_.py:
- Environment Variables: Requires `FAL_KEY`. Optional: `FALAI_IMAGE_MODEL`, `FAL_GPT_IMAGE_1_PRICE_PER_SECOND`.
- Dependencies: Requires `fal-ai`, `python-dotenv`, `aiohttp` (for URL downloads).
- Caching: Calling code manages caching.
- Concurrency: Designed for concurrent execution, be mindful of Fal.ai rate limits.
- Error Handling: Raises `PromptToImageFalError` on failure after retries or returns dict with status.
"""

import os
import sys
import logging
import time
import asyncio
import base64
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional

from dotenv import load_dotenv
import fal_client as fal # Using `fal` alias for `fal_client`
import aiohttp # For downloading images if URL is provided

# --- Configuration ---
load_dotenv()
FAL_API_KEY = os.getenv("FAL_KEY")
FALAI_IMAGE_MODEL_DEFAULT = "fal-ai/gpt-image-1/text-to-image"
FALAI_IMAGE_MODEL = os.getenv("FALAI_IMAGE_MODEL", FALAI_IMAGE_MODEL_DEFAULT)

FAL_GPT_IMAGE_1_PRICE_PER_SECOND_DEFAULT = "0.0001"
FAL_GPT_IMAGE_1_PRICE_PER_SECOND = Decimal(os.getenv("FAL_GPT_IMAGE_1_PRICE_PER_SECOND", FAL_GPT_IMAGE_1_PRICE_PER_SECOND_DEFAULT))

PER_MIN_RATE_LIMIT_FAL_TEST = 20 # For testing rate limits (Semaphore setting)
DEFAULT_API_RETRY_COUNT_FAL = 1
DEFAULT_TIMEOUT_SECONDS_FAL_TEST = 120 # For aiohttp and potentially Fal calls (if configurable)

# Test parameters for 'low' quality and smaller size
DEFAULT_IMAGE_SIZE_FAL_TEST = "1024x1024"
DEFAULT_IMAGE_QUALITY_FAL_TEST = "low"
DEFAULT_IMAGE_BACKGROUND_FAL_TEST = "opaque"

# Original defaults (can be used by orchestrator if not testing)
DEFAULT_IMAGE_SIZE_FAL = "1024x1024"
DEFAULT_IMAGE_QUALITY_FAL = "medium"
DEFAULT_IMAGE_BACKGROUND_FAL = "opaque"


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("fal_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)


class PromptToImageFalError(Exception):
    """Custom exception for PromptToImageFal errors."""
    pass

class PromptToImageFal:
    """
    Handles image generation using a Fal.ai endpoint (Async, V2 - Async Fix).
    Tracks estimated costs and timings.
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 model_id: str = FALAI_IMAGE_MODEL,
                 price_per_second: Decimal = FAL_GPT_IMAGE_1_PRICE_PER_SECOND,
                 api_retries: int = DEFAULT_API_RETRY_COUNT_FAL,
                 timeout: int = DEFAULT_TIMEOUT_SECONDS_FAL_TEST): # Using test default timeout here
        self.api_key = api_key or FAL_API_KEY
        if not self.api_key or ':' not in self.api_key:
            raise ValueError("Fal API key not found or incorrect format ('key_id:key_secret').")
        
        self.model_id = model_id
        self.price_per_second = price_per_second
        self.api_retries = api_retries
        self.timeout = timeout # For aiohttp and potentially Fal timeout if handler supports

        self.total_requests = 0; self.total_successful_requests = 0; self.total_failed_requests = 0
        self.total_estimated_cost = Decimal("0.0")
        self.total_successful_request_duration_sum = 0.0
        self.total_inference_time_sec = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"PromptToImageFal V2 initialized. Model: {self.model_id}. Price: ${self.price_per_second}/sec.")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout_config)
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close(); self._session = None
            logger.info("Aiohttp session closed for PromptToImageFal.")
            
    def _calculate_cost(self, inference_time_sec: float) -> Decimal:
        if inference_time_sec <= 0: return Decimal("0.0")
        return (Decimal(str(inference_time_sec)) * self.price_per_second).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    async def generate_image(
        self,
        prompt: str,
        output_image_path: str | Path,
        size: str = DEFAULT_IMAGE_SIZE_FAL,      # Using original defaults
        quality: str = DEFAULT_IMAGE_QUALITY_FAL,  # Using original defaults
        background: str = DEFAULT_IMAGE_BACKGROUND_FAL, # Using original defaults
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> Dict[str, Any]:
        if semaphore: await semaphore.acquire()

        try:
            self.total_requests += 1
            overall_start_time = time.monotonic()
            output_path = Path(output_image_path)
            log_prefix = f"Fal Img ({self.model_id.split('/')[-1]}, {size}, {quality} -> {output_path.name})"
            logger.info(f"{log_prefix}: Requesting image for prompt: \"{prompt[:60]}...\"")

            try: output_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                 logger.error(f"{log_prefix}: Failed to create output dir {output_path.parent}: {e}")
                 self.total_failed_requests += 1
                 return {"status": "failed", "image_path": None, "cost": 0.0, "request_time": time.monotonic() - overall_start_time, "revised_prompt": None, "inference_time_sec": 0.0, "error_message": f"Failed to create output dir: {e}"}

            result_dict = {"status": "failed", "image_path": None, "cost": 0.0, "request_time": 0.0, "revised_prompt": None, "inference_time_sec": 0.0, "error_message": "Unknown."}
            arguments = {"prompt": prompt, "image_size": size, "quality": quality, "background": background, "num_images": 1}
            arguments = {k: v for k, v in arguments.items() if v is not None}
            logger.debug(f"{log_prefix}: Calling Fal with args: {arguments}")

            attempts = 0; last_exception = None; response_data = None
            actual_api_request_duration = 0.0

            while attempts <= self.api_retries:
                # --- Timing for API submission and result retrieval ---
                submit_and_get_start_time = time.monotonic() 
                handler = None # Define handler outside try for potential access in error logging
                try:
                    logger.debug(f"{log_prefix}: Fal API call attempt {attempts + 1}/{self.api_retries + 1}")
                    
                    # Async submission
                    handler = await fal.submit_async(self.model_id, arguments=arguments)
                    # Wait for the result
                    response_data = await handler.get() # This is where most of the waiting happens

                    actual_api_request_duration = time.monotonic() - submit_and_get_start_time
                    logger.debug(f"{log_prefix}: Fal API call successful (Attempt {attempts + 1}). Total wait for result: {actual_api_request_duration:.3f}s")

                    if not response_data or not isinstance(response_data, dict):
                        raise PromptToImageFalError(f"Fal API response invalid. Type: {type(response_data)}")

                    images_data = response_data.get("images")
                    if not images_data or not isinstance(images_data, list) or not images_data[0]:
                        raise PromptToImageFalError("API response missing 'images' or empty list.")

                    image_info = images_data[0]
                    image_b64 = image_info.get("b64_json"); image_url = image_info.get("url")
                    revised_prompt = response_data.get("revised_prompt") 
                    
                    timings = response_data.get("timings", {})
                    inference_time_sec = float(timings.get("inference_time", 0.0))
                    if inference_time_sec == 0.0 and actual_api_request_duration > 0:
                        logger.warning(f"{log_prefix}: Fal 'inference_time' 0.0. Using API wait ({actual_api_request_duration:.3f}s) for cost.")
                        inference_time_sec = actual_api_request_duration

                    estimated_cost = self._calculate_cost(inference_time_sec)
                    logger.debug(f"{log_prefix}: Inference reported: {inference_time_sec:.3f}s. Cost: ${estimated_cost:.6f}")

                    image_bytes = None
                    if image_b64:
                        try: image_bytes = base64.b64decode(image_b64)
                        except Exception as e: raise PromptToImageFalError(f"Base64 decode error: {e}")
                    elif image_url:
                        logger.info(f"{log_prefix}: Downloading from URL: {image_url}")
                        session = await self._get_session()
                        try:
                            async with session.get(image_url) as resp:
                                if resp.status == 200: image_bytes = await resp.read()
                                else:
                                    error_text = await resp.text()
                                    raise PromptToImageFalError(f"Download fail ({resp.status}): {error_text[:200]}")
                        except aiohttp.ClientError as e: raise PromptToImageFalError(f"Download net error: {e}")
                    else: raise PromptToImageFalError("No b64_json or URL in response.")

                    if not image_bytes: raise PromptToImageFalError("Image_bytes empty.")

                    try:
                        with open(output_path, "wb") as f: f.write(image_bytes)
                        logger.info(f"{log_prefix}: Image saved to {output_path}")
                    except Exception as e: raise PromptToImageFalError(f"Save image error: {e}")

                    self.total_successful_requests += 1; self.total_estimated_cost += estimated_cost
                    self.total_successful_request_duration_sum += actual_api_request_duration
                    self.total_inference_time_sec += inference_time_sec
                    result_dict.update({"status": "success", "image_path": str(output_path), "cost": float(estimated_cost), "request_time": actual_api_request_duration, "revised_prompt": revised_prompt, "inference_time_sec": inference_time_sec, "error_message": None})
                    return result_dict

                # --- Error Handling (Corrected Fal client errors) ---
                except (fal.api.FalServerException, fal.api.HTTPError, fal_client.RequestError, fal_client.client.FalClientError) as e:
                    last_exception = e; actual_api_request_duration = time.monotonic() - submit_and_get_start_time
                    error_details = str(e)
                    if hasattr(e, 'request_id') and e.request_id: error_details += f" (Request ID: {e.request_id})"
                    if hasattr(handler, 'request_id') and handler.request_id and 'Request ID' not in error_details: error_details += f" (Handler Request ID: {handler.request_id})"

                    logger.warning(f"{log_prefix}: Fal Client/API Error (Attempt {attempts + 1}): {type(e).__name__} - {error_details} (Duration: {actual_api_request_duration:.3f}s)")
                except PromptToImageFalError as e: 
                    last_exception = e; actual_api_request_duration = time.monotonic() - submit_and_get_start_time
                    logger.error(f"{log_prefix}: Processing error (Attempt {attempts + 1}): {e} (Duration: {actual_api_request_duration:.3f}s)")
                    result_dict["error_message"] = str(e); result_dict["request_time"] = actual_api_request_duration
                    self.total_failed_requests += 1
                    return result_dict 
                except aiohttp.ClientError as e: 
                    last_exception = e; actual_api_request_duration = time.monotonic() - submit_and_get_start_time
                    logger.warning(f"{log_prefix}: Image download net error (Attempt {attempts + 1}): {e} (Duration: {actual_api_request_duration:.3f}s)")
                except Exception as e: 
                    last_exception = e; actual_api_request_duration = time.monotonic() - submit_and_get_start_time
                    logger.error(f"{log_prefix}: Unexpected error (Attempt {attempts + 1}) [{type(e).__name__}]: {e} (Duration: {actual_api_request_duration:.3f}s)", exc_info=True)
                    result_dict["error_message"] = f"Unexpected: {type(e).__name__}: {e}"; result_dict["request_time"] = actual_api_request_duration
                    self.total_failed_requests +=1
                    return result_dict

                attempts += 1
                if attempts <= self.api_retries:
                    wait_time = 2 ** (attempts - 1) 
                    logger.info(f"{log_prefix}: Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{log_prefix}: Max retries reached. Failed.")
                    self.total_failed_requests += 1
                    final_duration = time.monotonic() - overall_start_time
                    if last_exception: result_dict["error_message"] = f"Failed after {self.api_retries+1} attempts. Last: {type(last_exception).__name__}: {last_exception}"
                    else: result_dict["error_message"] = f"Failed after {self.api_retries+1} attempts."
                    result_dict["request_time"] = final_duration 
                    if output_path.exists() and result_dict["status"] == "failed":
                        try: output_path.unlink()
                        except OSError as rm_err: logger.warning(f"{log_prefix}: Failed to remove {output_path}: {rm_err}")
                    return result_dict
        finally:
            if semaphore: semaphore.release()
        
        result_dict["request_time"] = time.monotonic() - overall_start_time
        return result_dict


    def get_stats(self) -> Dict[str, Any]:
        return {"total_requests": self.total_requests, "successful": self.total_successful_requests, "failed": self.total_failed_requests, "total_estimated_cost": self.total_estimated_cost, "total_successful_request_duration_sum": self.total_successful_request_duration_sum, "total_inference_time_sec_all_successful": self.total_inference_time_sec}

# --- Main Execution Block (for testing) ---
async def run_fal_image_gen_task(handler: PromptToImageFal, prompt: str, output_path: Path, task_id: int, semaphore: asyncio.Semaphore, size: str, quality:str, background: str):
    logger.info(f"--- Fal Task {task_id}: Waiting for semaphore ---")
    result = await handler.generate_image(prompt=prompt, output_image_path=output_path, size=size, quality=quality, background=background, semaphore=semaphore)
    logger.info(f"--- Fal Task {task_id}: Finished ---")
    return result

async def main_test_fal():
    print("--- Running PromptToImageFal Test (V2 Async Fix) ---")
    if not FAL_API_KEY or ':' not in FAL_API_KEY:
        print("Error: FAL_KEY not found or incorrect format.", file=sys.stderr); return

    print(f"Using Fal Model ID: {FALAI_IMAGE_MODEL}")
    print(f"Test Params: Size='{DEFAULT_IMAGE_SIZE_FAL_TEST}', Quality='{DEFAULT_IMAGE_QUALITY_FAL_TEST}', Timeout='{DEFAULT_TIMEOUT_SECONDS_FAL_TEST}s'")
    print(f"Semaphore limit for test: {PER_MIN_RATE_LIMIT_FAL_TEST}")

    image_handler = None
    try:
        # Pass test-specific timeout to handler
        image_handler = PromptToImageFal(api_retries=1, timeout=DEFAULT_TIMEOUT_SECONDS_FAL_TEST)

        test_prompts = [
            "Ultra-HD, photorealistic image: A cunning red fox, fur detailed to the whisker, stealthily navigates a dew-kissed forest floor at dawn. Soft golden light filters through the dense canopy, illuminating patches of moss and ferns. One paw is delicately lifted mid-stride. Its amber eyes are sharply focused on something unseen. gpt-image-1 style.",
            "Whimsical digital painting: A tiny, glowing astronaut cat, wearing a comically oversized helmet, joyfully bounces on a giant, cratered cheese moon. Earth hangs like a distant blue marble in a swirling nebula of pink and purple. Stars twinkle brightly. gpt-image-1 style.",
            "Extreme close-up, macro photography: A single, perfectly formed snowflake rests on a dark, textured piece of wool. The intricate, symmetrical patterns of the ice crystal are sharply defined. Tiny air bubbles are trapped within its structure. Refracted light creates rainbow micro-glints. gpt-image-1 style.",
            "Surreal matte painting: A colossal, antique pocket watch, its gears exposed and rusted, melts dramatically over a vast, sun-scorched desert. The sky above is a maelstrom of dark, turbulent storm clouds, with a single ray of light piercing through to illuminate the watch. Time itself seems to be dissolving. gpt-image-1 style.",
            "Concept art sketch, detailed: A friendly, bipedal robot companion, primarily white with orange accents, stands in a bright, modern workshop. It has large, expressive eyes and a slightly curious head tilt. Tools and holographic schematics are visible in the background. gpt-image-1 style.",
            "Vibrant abstract digital art: Flowing ribbons of electric blue, fiery orange, and deep magenta intertwine and swirl across a dark canvas, representing the boundless energy of 'creativity'. Textured brushstrokes and particle effects add depth and dynamism. gpt-image_1 style.",
            "Charming illustration: A fluffy corgi dog with a joyful expression zooms through a star-dusted galaxy, riding a giant slice of pepperoni pizza as if it were a surfboard. A colorful nebula forms a vibrant backdrop. Its tongue is lolling out in a happy pant. gpt-image-1 style."
        ]
        num_tasks = 7 # Test 7 images

        test_output_dir = Path("temp_files") / "image_test_fal_concurrent_v3" # New dir for new test
        test_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output images will be saved to: {test_output_dir}")

        semaphore = asyncio.Semaphore(PER_MIN_RATE_LIMIT_FAL_TEST)
        print(f"\nConcurrency limited to {PER_MIN_RATE_LIMIT_FAL_TEST} tasks using Semaphore.")

        tasks = []
        print(f"Creating {num_tasks} Fal image generation tasks...")
        for i, prompt_text in enumerate(test_prompts[:num_tasks]): # Slice to ensure 7 tasks
            filename = f"test_image_fal_v3_{i+1}.png"
            output_path = test_output_dir / filename
            task = asyncio.create_task(
                run_fal_image_gen_task(
                    image_handler, prompt_text, output_path, task_id=i+1, semaphore=semaphore,
                    size=DEFAULT_IMAGE_SIZE_FAL_TEST, # Use test parameters
                    quality=DEFAULT_IMAGE_QUALITY_FAL_TEST,
                    background=DEFAULT_IMAGE_BACKGROUND_FAL_TEST
                ),
                name=f"FalImageTask_v3_{i+1}"
            )
            tasks.append(task)

        print(f"Waiting for {num_tasks} tasks to complete (max {PER_MIN_RATE_LIMIT_FAL_TEST} concurrent)...")
        concurrent_start_time = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_concurrent_duration = time.monotonic() - concurrent_start_time
        print(f"--- Fal Concurrent execution finished in {total_concurrent_duration:.3f}s ---")

        print("\n--- Individual Fal Task Results ---")
        successful_tasks = 0; total_cost_all_tasks = Decimal("0.0"); failed_task_details = []
        for i, res in enumerate(results):
            print("-" * 20); task_name = tasks[i].get_name()
            if isinstance(res, Exception):
                logger.error(f"Task {task_name} failed catastrophically: {res}", exc_info=res)
                print(f"Task {i+1} ({task_name}): CATASTROPHIC FAILURE\n  Error: {res}")
                failed_task_details.append(f"Task {i+1} ({task_name}): Catastrophic - {res}")
            elif isinstance(res, dict):
                 print(f"Task {i+1} ({task_name}) - Prompt: \"{test_prompts[i][:30]}...\"")
                 print(f"  Status:   {res['status']}")
                 print(f"  API Wait: {res['request_time']:.3f}s") # This is handler.get() wait
                 cost = Decimal(str(res.get('cost','0.0')))
                 
                 if res['status'] == "success":
                     successful_tasks += 1; total_cost_all_tasks += cost 
                     print(f"  Image:    {res.get('image_path')}")
                     print(f"  Cost Est: ${cost:.6f} (Inference: {res.get('inference_time_sec',0.0):.3f}s)")
                     if res.get('revised_prompt'): print(f"  Revised:  {res['revised_prompt']}")
                 else:
                    if res.get('cost') is not None: total_cost_all_tasks += cost
                    print(f"  Cost Est: ${cost:.6f} (Attempted)")
                    if res.get('inference_time_sec', 0.0) > 0: print(f"  Inference Attempt: {res['inference_time_sec']:.3f}s")
                    print(f"  Error:    {res.get('error_message')}")
                    failed_task_details.append(f"Task {i+1} ({task_name}): Status='{res['status']}' - {res.get('error_message')}")
            else:
                print(f"Task ID {i+1} ({task_name}): UNEXPECTED RESULT: {type(res)}")
                failed_task_details.append(f"Task {i+1} ({task_name}): Unexpected type {type(res)}")

        print("\n" + "="*25 + " Fal Image Gen Test Summary (V2) " + "="*25)
        print(f"Total Wall-Clock (Batch): {total_concurrent_duration:.3f}s")
        print(f"Attempted / Successful: {len(tasks)} / {successful_tasks}")
        if failed_task_details:
             print("Failed Tasks:"); [print(f"  - {d}") for d in failed_task_details[:5]]; print("  ..." if len(failed_task_details)>5 else "")
        print(f"Total Cost (Results/Attempts): ${total_cost_all_tasks:.6f}")
        print("-" * 76)
        if image_handler:
            stats = image_handler.get_stats()
            print(f"Handler Metrics: Req:{stats['total_requests']} (S:{stats['successful']},F:{stats['failed']}), Cost:${stats['total_estimated_cost']:.6f}, API Wait Sum:{stats['total_successful_request_duration_sum']:.3f}s, Fal Inf Sum:{stats['total_inference_time_sec_all_successful']:.3f}s")
        print("=" * 76)

    except Exception as e: logger.error(f"Unexpected test error: {e}", exc_info=True)
    finally:
        if image_handler: await image_handler.close_session()

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_test_fal())
    except KeyboardInterrupt: print("\nFal Test interrupted.")
