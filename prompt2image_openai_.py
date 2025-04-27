# File: prompt2image_openai_.py
"""
Prompt-to-Image Module using OpenAI's gpt-image-1 (prompt2image_openai_.py)

Handles generating images from text prompts using the OpenAI Images API, specifically
targeting the 'gpt-image-1' model (GPT-4o's image generation capability).

This module is part of the 'storyshot' project and is called by `storyshot_.py`
after storyboard prompts have been generated.

*** RATE LIMIT NOTE ***
The gpt-image-1 model currently has a rate limit of approximately 5 images per minute.
When processing many images concurrently, the orchestrator may need to implement
throttling (e.g., using asyncio.Semaphore or delays between batches) to avoid
hitting this limit and receiving RateLimitError exceptions.

API Info (OpenAI Images API - gpt-image-1):
- Docs: https://platform.openai.com/docs/guides/images/image-generation-api
- Authentication: Bearer token via `Authorization` header. Requires OPENAI_KEY environment variable.
- Client Library: `openai` (pip install openai)
- Model: `gpt-image-1` (configurable via OPENAI_IMAGE_MODEL env var).
- Endpoint: `v1/images/generations` (managed by the library).
- Key Input Parameters:
    - `model`: The specific model ID (e.g., "gpt-image-1").
    - `prompt`: The text description for the image.
    - `n`: Number of images to generate (currently fixed at 1).
    - `size`: Image dimensions (e.g., "1024x1024", "1024x1536", "1536x1024", "auto").
    - `quality`: "low", "medium", "high", "auto". Affects cost and detail.
    - `background`: "transparent" or "opaque" (or "auto"). Used as "opaque".
- Key Output (from API via library): An object containing a `data` list. Each item has:
    - `b64_json`: Base64 encoded string of the PNG image data.
    - `revised_prompt`: Potentially a modified version of the input prompt.
- Cost: Token-based. Depends on prompt length and output image size/quality.

Workflow: (Same as before)

Key Features & Outputs:
- Async Ready: Uses `async`/`await` and the `openai` async client.
- Cost Tracking: Estimates cost per request. Maintains a running total within the instance.
- Timing Tracking: Measures *individual successful API call* duration.
- Result Dictionary: The `generate_image` method returns a dictionary:
    {
        "status": "success" | "failed",
        "image_path": str | None,
        "cost": float,
        "request_time": float, # Duration of this specific successful API call
        "revised_prompt": str | None,
        "prompt_tokens_est": int,
        "output_tokens": int,
        "error_message": str | None
    }
- Retries: Implements basic retry logic for transient errors.

Usage Notes / Integration with storyshot_.py:
- Environment Variables: Requires `OPENAI_KEY`. Optional: `OPENAI_IMAGE_MODEL`.
- Dependencies: Requires `openai`, `python-dotenv`, `tiktoken`.
- Caching: Calling code manages caching.
- Concurrency: Designed for concurrent execution, but be mindful of rate limits (see note above).
- Error Handling: Raises `PromptToImageError` on failure after retries.
"""

import os
import sys
import logging
import time
import asyncio
import base64
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv
import openai
import tiktoken

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
OPENAI_IMAGE_MODEL_DEFAULT = "gpt-image-1"
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", OPENAI_IMAGE_MODEL_DEFAULT)
PER_MIN_RATE_LIMIT = 5  # Images per minute
DEFAULT_API_RETRY_COUNT = 1
DEFAULT_TIMEOUT_SECONDS = 180

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_IMAGE_QUALITY = "low"
DEFAULT_IMAGE_BACKGROUND = "opaque"
DEFAULT_IMAGE_MODERATION = "low" # or "auto"

PRICE_PER_MILLION_INPUT_TOKENS = Decimal("5.00")
PRICE_PER_MILLION_OUTPUT_TOKENS = Decimal("15.00")
TOKENS_PER_MILLION = Decimal("1000000")

OUTPUT_TOKENS_MAP = {
    "low": {"1024x1024": 272, "1024x1536": 408, "1536x1024": 400},
    "medium": {"1024x1024": 1056, "1024x1536": 1584, "1536x1024": 1568},
    "high": {"1024x1024": 4160, "1024x1536": 6240, "1536x1024": 6208},
    "auto": {"1024x1024": 1056, "1024x1536": 1584, "1536x1024": 1568},
}


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class PromptToImageError(Exception):
    """Custom exception for PromptToImageOpenAI errors."""
    pass

class PromptToImageOpenAI:
    """
    Handles image generation using OpenAI's gpt-image-1 model (Async).
    Tracks estimated costs and timings.
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = OPENAI_IMAGE_MODEL,
                 api_retries: int = DEFAULT_API_RETRY_COUNT,
                 timeout: int = DEFAULT_TIMEOUT_SECONDS):
        """
        Initializes the PromptToImageOpenAI handler.

        Args:
            api_key (Optional[str]): OpenAI API key. Defaults to OPENAI_KEY env var.
            model (str): The specific OpenAI image model ID to use.
            api_retries (int): Number of retries for transient API errors.
            timeout (int): Timeout in seconds for API requests.
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables (OPENAI_KEY) or arguments.")

        self.model = model
        self.api_retries = api_retries
        self.timeout = timeout

        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}")
            raise PromptToImageError(f"Failed to initialize AsyncOpenAI client: {e}") from e

        # Internal cumulative tracking
        self.total_requests = 0
        self.total_successful_requests = 0
        self.total_failed_requests = 0
        self.total_estimated_cost = Decimal("0.0")
        # *** CLARIFIED METRIC ***
        self.total_successful_request_duration_sum = 0.0 # Sum of INDIVIDUAL successful request durations
        self.total_prompt_tokens_est = 0
        self.total_output_tokens = 0

        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoder 'cl100k_base'. Falling back to simple word count for prompt token estimation. Error: {e}")
            self._tokenizer = None

        logger.info(f"PromptToImageOpenAI initialized. Model: {self.model}. Async ready.")

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimates the number of tokens in the input prompt."""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(prompt))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed for prompt, falling back to word count. Error: {e}")
                return int(len(prompt.split()) * 1.3)
        else:
            return int(len(prompt.split()) * 1.3)

    def _get_output_tokens(self, size: str, quality: str) -> int:
        """Gets the number of output tokens based on size and quality."""
        quality_key = quality if quality in OUTPUT_TOKENS_MAP else "medium"
        size_key = size

        if quality_key not in OUTPUT_TOKENS_MAP or size_key not in OUTPUT_TOKENS_MAP[quality_key]:
            default_size = DEFAULT_IMAGE_SIZE
            logger.warning(f"Could not find output tokens for size='{size}' quality='{quality}'. Falling back to quality='{quality_key}' size='{default_size}'.")
            if default_size not in OUTPUT_TOKENS_MAP[quality_key]:
                 logger.error(f"Fallback size '{default_size}' also not found in token map for quality '{quality_key}'. Returning 0 tokens.")
                 return 0
            return OUTPUT_TOKENS_MAP[quality_key].get(default_size, 0)
        return OUTPUT_TOKENS_MAP[quality_key].get(size_key, 0)

    def _calculate_cost(self, prompt_tokens: int, output_tokens: int) -> Decimal:
        """Calculates the estimated cost based on input and output tokens."""
        if prompt_tokens < 0 or output_tokens < 0: return Decimal("0.0")
        input_cost = (Decimal(str(prompt_tokens)) / TOKENS_PER_MILLION) * PRICE_PER_MILLION_INPUT_TOKENS
        output_cost = (Decimal(str(output_tokens)) / TOKENS_PER_MILLION) * PRICE_PER_MILLION_OUTPUT_TOKENS
        total_cost = input_cost + output_cost
        return total_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    async def generate_image(
        self,
        prompt: str,
        output_image_path: str | Path,
        size: str = DEFAULT_IMAGE_SIZE,
        quality: str = DEFAULT_IMAGE_QUALITY,
        moderation: str = DEFAULT_IMAGE_MODERATION,
        semaphore: Optional[asyncio.Semaphore] = None # Added semaphore argument
    ) -> Dict[str, Any]:
        """
        Generates an image from a prompt asynchronously, saves it, estimates cost.
        Uses an optional semaphore to limit concurrency.

        Args:
            prompt (str): The text prompt for image generation.
            output_image_path (str | Path): Full path to save the generated image.
            size (str): Image dimensions (e.g., "1024x1024").
            quality (str): Rendering quality ("low", "medium", "high", "auto").
            semaphore (Optional[asyncio.Semaphore]): Semaphore to limit concurrent API calls.

        Returns:
            Dict[str, Any]: Result dictionary.
        """
        # Acquire semaphore if provided, making this call wait if limit is reached
        if semaphore:
            await semaphore.acquire()

        try: # Use try/finally to ensure semaphore is released
            self.total_requests += 1
            start_time = time.monotonic() # Overall start time for this specific call attempt
            output_path = Path(output_image_path)
            log_prefix = f"Image Gen ({self.model}, {size}, {quality} -> {output_path.name})"
            logger.info(f"{log_prefix}: Requesting image for prompt: \"{prompt[:60]}...\"")

            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                 logger.error(f"{log_prefix}: Failed to create output directory {output_path.parent}: {e}")
                 self.total_failed_requests += 1
                 return {
                     "status": "failed", "image_path": None, "cost": 0.0, "request_time": time.monotonic() - start_time,
                     "revised_prompt": None, "prompt_tokens_est": 0, "output_tokens": 0,
                     "error_message": f"Failed to create output directory: {e}"
                 }

            prompt_tokens_est = self._estimate_prompt_tokens(prompt)
            output_tokens = self._get_output_tokens(size, quality)
            estimated_cost = self._calculate_cost(prompt_tokens_est, output_tokens)
            logger.debug(f"{log_prefix}: Tokens Est: {prompt_tokens_est} prompt + {output_tokens} output. Cost Est: ${estimated_cost:.6f}")

            result_dict = {
                "status": "failed", "image_path": None, "cost": float(estimated_cost),
                "request_time": 0.0, # Duration of the successful API call attempt
                "revised_prompt": None,
                "prompt_tokens_est": prompt_tokens_est, "output_tokens": output_tokens,
                "error_message": "An unknown error occurred."
            }

            attempts = 0
            last_exception = None
            while attempts <= self.api_retries:
                attempt_start_time = time.monotonic()
                request_duration = 0.0
                try:
                    logger.debug(f"{log_prefix}: API call attempt {attempts + 1}/{self.api_retries + 1}")
                    response = await self.client.images.generate(
                        model=self.model,
                        prompt=prompt,
                        n=1,
                        size=size,
                        quality=quality,
                        moderation=moderation,
                    )

                    request_duration = time.monotonic() - attempt_start_time
                    logger.debug(f"{log_prefix}: API call successful (Attempt {attempts + 1}). Time: {request_duration:.3f}s")

                    if not response.data or not response.data[0].b64_json:
                         raise PromptToImageError("API response missing image data (b64_json).")

                    image_b64 = response.data[0].b64_json
                    revised_prompt = response.data[0].revised_prompt

                    try:
                        image_bytes = base64.b64decode(image_b64)
                        # Use async file writing for better event loop usage (though disk I/O can still block)
                        # Needs 'aiofiles' library: pip install aiofiles
                        # import aiofiles
                        # async with aiofiles.open(output_path, "wb") as f:
                        #     await f.write(image_bytes)
                        # Using sync write for simplicity for now, as it's usually fast
                        with open(output_path, "wb") as f:
                            f.write(image_bytes)

                        logger.info(f"{log_prefix}: Image successfully generated and saved to {output_path}")
                    except (base64.binascii.Error, IOError, OSError) as e:
                         raise PromptToImageError(f"Failed to decode or save image: {e}") from e

                    # Success!
                    self.total_successful_requests += 1
                    self.total_estimated_cost += estimated_cost
                    self.total_successful_request_duration_sum += request_duration # Add duration of successful attempt
                    self.total_prompt_tokens_est += prompt_tokens_est
                    self.total_output_tokens += output_tokens

                    result_dict.update({
                        "status": "success",
                        "image_path": str(output_path),
                        "cost": float(estimated_cost),
                        "request_time": request_duration, # Duration of this specific call
                        "revised_prompt": revised_prompt,
                        "error_message": None
                    })
                    return result_dict # Exit loop on success

                # --- Error Handling & Retry Logic ---
                except openai.RateLimitError as e:
                    last_exception = e
                    logger.warning(f"{log_prefix}: Rate limit exceeded (Attempt {attempts + 1}): {e}. Retrying...")
                     # Add extra wait specifically for rate limits
                    wait_time = 5 + (2 ** attempts) # Base wait + exponential
                    logger.info(f"{log_prefix}: Waiting {wait_time}s due to rate limit...")
                    await asyncio.sleep(wait_time)
                    # Don't increment attempts here, let the loop do it, but continue to next iteration
                    continue

                except openai.APIConnectionError as e: last_exception = e; logger.warning(f"{log_prefix}: Network error (Attempt {attempts + 1}): {e}. Retrying...")
                except openai.APITimeoutError as e: last_exception = e; logger.warning(f"{log_prefix}: Request timed out (Attempt {attempts + 1}): {e}. Retrying...")
                except openai.APIStatusError as e:
                    last_exception = e
                    request_duration = time.monotonic() - attempt_start_time
                    logger.error(f"{log_prefix}: OpenAI API Status Error (Attempt {attempts + 1}): {e.status_code} - {e.response.text} (Duration: {request_duration:.3f}s)")
                    if 400 <= e.status_code < 500 and e.status_code != 429:
                        result_dict["error_message"] = f"OpenAI Client Error {e.status_code}: {e.message}"
                        result_dict["request_time"] = request_duration
                        self.total_failed_requests += 1
                        return result_dict # No retry
                except PromptToImageError as e:
                    last_exception = e
                    request_duration = time.monotonic() - attempt_start_time
                    logger.error(f"{log_prefix}: Processing error (Attempt {attempts + 1}): {e} (Duration: {request_duration:.3f}s)")
                    result_dict["error_message"] = str(e)
                    result_dict["request_time"] = request_duration
                    self.total_failed_requests += 1
                    return result_dict # No retry
                except Exception as e:
                    last_exception = e
                    request_duration = time.monotonic() - attempt_start_time
                    error_type_name = type(e).__name__
                    logger.error(f"{log_prefix}: Unexpected error (Attempt {attempts + 1}) [{error_type_name}]: {e} (Duration: {request_duration:.3f}s)", exc_info=True)
                    result_dict["error_message"] = f"Unexpected error: {error_type_name}"

                # --- Retry Logic ---
                attempts += 1
                if attempts <= self.api_retries:
                    # General backoff for errors other than rate limit
                    wait_time = 2 ** (attempts - 1)
                    logger.info(f"{log_prefix}: Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{log_prefix}: Max retries ({self.api_retries}) reached. Failed.")
                    self.total_failed_requests += 1
                    final_duration = time.monotonic() - start_time # Total time for this call including waits/retries
                    if last_exception: result_dict["error_message"] = f"Failed after {self.api_retries+1} attempts. Last error: {type(last_exception).__name__}: {last_exception}"
                    else: result_dict["error_message"] = f"Failed after {self.api_retries+1} attempts. Unknown error."
                    result_dict["request_time"] = final_duration
                    if output_path.exists():
                        try: output_path.unlink()
                        except OSError as rm_err: logger.warning(f"{log_prefix}: Could not remove incomplete file {output_path}: {rm_err}")
                    return result_dict

        finally:
             # Release semaphore if it was provided
             if semaphore:
                 semaphore.release()

        # Fallback return (should not be reached ideally)
        result_dict["request_time"] = time.monotonic() - start_time
        return result_dict


    def get_stats(self) -> Dict[str, Any]:
        """ Returns cumulative statistics for this instance. """
        return {
            "total_requests": self.total_requests,
            "successful": self.total_successful_requests,
            "failed": self.total_failed_requests,
            "total_estimated_cost": self.total_estimated_cost,
            "total_successful_request_duration_sum": self.total_successful_request_duration_sum, # Clarified name
            "total_prompt_tokens_est": self.total_prompt_tokens_est,
            "total_output_tokens": self.total_output_tokens,
        }

# --- Main Execution Block (for testing) ---
async def run_image_gen_task(handler, prompt, output_path, task_id, semaphore):
    """Helper to run a single generation task with semaphore."""
    logger.info(f"--- Task {task_id}: Waiting for semaphore ---")
    # Pass the semaphore to the generate_image method
    result = await handler.generate_image(prompt, output_path, semaphore=semaphore)
    logger.info(f"--- Task {task_id}: Finished (Semaphore released implicitly) ---")
    return result

async def main_test():
    """Tests the PromptToImageOpenAI module with concurrency limit."""
    print("--- Running PromptToImageOpenAI Test ---")
    if not OPENAI_API_KEY:
        print("Error: OPENAI_KEY not found in environment variables.", file=sys.stderr)
        return

    print(f"Using OpenAI Model: {OPENAI_IMAGE_MODEL}")

    image_handler = None
    # *** Define Concurrency Limit ***
    MAX_CONCURRENT_REQUESTS =  PER_MIN_RATE_LIMIT # Set below the 5/min limit to be safe

    try:
        image_handler = PromptToImageOpenAI(api_retries=1)

        test_prompts = [
            "A high-resolution photorealistic image of a red panda wearing tiny glasses, programming intently on a glowing vintage computer in a dark room illuminated by neon signs reflecting off rain-slicked windows.",
            "A beautiful watercolor painting capturing the serene atmosphere of a crystal-clear mountain lake at sunrise. Mist gently rises from the water surface and weaves through the dense pine forest surrounding the lake. Soft pastel colors dominate the sky.",
            "A highly detailed close-up vector illustration, flat design style with clean lines, showcasing a complex mechanical steampunk heart with intricate gears, copper pipes, glowing vacuum tubes, and subtle wisps of escaping steam.",
            "Draw a cute fluffy Corgi dog flying through space on a slice of pizza, vibrant nebula background, cartoon style.",
            "An abstract representation of digital data streams using flowing lines and particles of light, dark background, minimalist.",
            "A macro photograph of a dewdrop on a blade of grass reflecting the morning sun.",
        ]
        num_tasks = len(test_prompts)

        test_output_dir = Path("temp_files") / "image_test_concurrent"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {test_output_dir}")

        # *** Create Semaphore ***
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        print(f"\nConcurrency limited to {MAX_CONCURRENT_REQUESTS} tasks using Semaphore.")

        tasks = []
        print(f"Creating {num_tasks} image generation tasks...")
        for i, prompt in enumerate(test_prompts):
            filename = f"test_image_concurrent_{i+1}.png"
            output_path = test_output_dir / filename
            # Pass semaphore to the task runner
            task = asyncio.create_task(
                run_image_gen_task(image_handler, prompt, output_path, task_id=i+1, semaphore=semaphore),
                name=f"ImageTask_{i+1}"
            )
            tasks.append(task)

        print(f"Waiting for {num_tasks} tasks to complete (with max {MAX_CONCURRENT_REQUESTS} running at a time)...")
        concurrent_start_time = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_end_time = time.monotonic()
        total_concurrent_duration = concurrent_end_time - concurrent_start_time
        print(f"--- Concurrent execution finished in {total_concurrent_duration:.3f} seconds ---")


        print("\n--- Individual Task Results ---")
        successful_tasks = 0
        total_cost_all_tasks = Decimal("0.0")
        failed_task_details = []
        for i, res in enumerate(results):
            print("-" * 20)
            task_name = tasks[i].get_name()
            if isinstance(res, Exception):
                logger.error(f"Task {task_name} failed catastrophically: {res}", exc_info=res)
                print(f"Task {i+1} ({task_name}): CATASTROPHIC FAILURE\n  Error: {res}")
                failed_task_details.append(f"Task {i+1}: Catastrophic Failure - {res}")
            elif isinstance(res, dict):
                 print(f"Task {i+1} ({task_name}) - Prompt: \"{test_prompts[i][:50]}...\"")
                 print(f"  Status:   {res['status']}")
                 # request_time now reflects individual call duration
                 print(f"  Call Duration: {res['request_time']:.3f}s")
                 cost = Decimal(str(res['cost']))
                 total_cost_all_tasks += cost
                 if res['status'] == "success":
                     successful_tasks += 1
                     print(f"  Image:    {res['image_path']}")
                     print(f"  Cost Est: ${cost:.6f}")
                     print(f"  Tokens:   {res['prompt_tokens_est']}p + {res['output_tokens']}o")
                     print(f"  Revised:  {res['revised_prompt']}")
                 else:
                    print(f"  Cost Est: ${cost:.6f} (Attempted)")
                    print(f"  Tokens:   {res['prompt_tokens_est']}p + {res['output_tokens']}o (Attempted)")
                    print(f"  Error:    {res['error_message']}")
                    failed_task_details.append(f"Task {i+1}: Status='failed' - {res['error_message']}")
            else:
                print(f"Task ID {i+1}: UNEXPECTED RESULT TYPE: {type(res)}")
                failed_task_details.append(f"Task {i+1}: Unexpected result type {type(res)}")


        print("\n" + "=" * 30 + " Image Gen Test Summary " + "=" * 30)
        print(f"Total Wall-Clock Time (Concurrent Batch): {total_concurrent_duration:.3f} seconds") # Corrected metric
        print(f"Tasks Attempted / Successful: {len(tasks)} / {successful_tasks}")
        if failed_task_details:
             print("Failed Tasks Details:")
             for detail in failed_task_details[:5]: # Print first 5 failures
                  print(f"  - {detail}")
             if len(failed_task_details) > 5:
                  print(f"  ... and {len(failed_task_details) - 5} more failures.")

        print(f"Total Cost (Summed from results/attempts): ${total_cost_all_tasks:.6f}")
        print("-" * 76)
        print("Handler Internal Metrics:")
        handler_stats = image_handler.get_stats()
        print(f"  Handler Total Requests Attempted: {handler_stats['total_requests']}")
        print(f"  Handler Successful / Failed: {handler_stats['successful']} / {handler_stats['failed']}")
        print(f"  Handler Total Cost Est (Successful Only): ${handler_stats['total_estimated_cost']:.6f}")
        # *** RENAMED METRIC ***
        print(f"  Handler Total Successful Request Duration (Sum): {handler_stats['total_successful_request_duration_sum']:.3f}s")
        handler_total_tokens_success = handler_stats['total_prompt_tokens_est'] + handler_stats['total_output_tokens']
        print(f"  Handler Total Tokens (Successful Only): {handler_total_tokens_success} ({handler_stats['total_prompt_tokens_est']}p + {handler_stats['total_output_tokens']}o)")
        print("=" * 76)

    except PromptToImageError as e: print(f"\nA PromptToImage specific error occurred: {e}", file=sys.stderr)
    except openai.AuthenticationError: print("\nAuthenticationError: Check your OPENAI_KEY.", file=sys.stderr)
    except Exception as e: logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
    finally:
        # Client closure handled internally by openai library context (if used) or GC
        pass

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_test())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

