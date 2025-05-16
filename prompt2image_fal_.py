# File: prompt2image_fal_.py
"""
Prompt-to-Image Module using Fal.ai (prompt2image_fal_.py)
Version: 4.0.0 (Based on V3.0.6 REST API logic)
Date: 2025-05-16

Handles generating images from text prompts using Fal.ai's queue system directly
via REST API calls. This module was specifically refactored to use the REST API
because an official 'cost' or precise 'inference_time' (for billing calculation)
was not reliably exposed by the high-level `fal-ai` Python library's abstractions
(e.g., `fal.submit_async()` and `handler.get()`) for the target image generation model
(e.g., "fal-ai/gpt-image-1/text-to-image") at the time of development.

By interacting directly with the Fal.ai REST queue API, this module can:
1. Submit an image generation request.
2. Poll the job's status URL.
3. Upon job completion, retrieve the final response which, according to Fal.ai's
   queue API documentation, should include a `metrics` object containing
   `inference_time`, and/or potentially a direct `cost` field.

This allows for more accurate cost tracking and billing duration recording.

*** Fal.ai REST Queue API Workflow Utilized ***
1.  **Submit Job (POST):**
    -   URL: `https://queue.fal.run/{FALAI_IMAGE_MODEL_ID}`
    -   Method: POST
    -   Authentication: `Authorization: Key YOUR_FAL_KEY_ID:YOUR_FAL_KEY_SECRET` header.
    -   Body: JSON payload with `prompt`, `image_size`, `quality`, `background`, `num_images`.
    -   Expected Response: JSON containing `request_id`, `status_url`, `response_url`, `cancel_url`.

2.  **Poll Status (GET):**
    -   URL: The `status_url` received from the submission step.
    -   Method: GET
    -   Loop: Periodically poll this URL until the job status is terminal (e.g., "COMPLETED", "FAILED").
    -   Expected status values: "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "ERROR".
    -   Upon "COMPLETED", the status response itself might contain `metrics` (like `inference_time`) and/or `cost`.

3.  **Fetch Result (GET):**
    -   URL: The `response_url` received from the submission step (or potentially from status updates).
    -   Method: GET
    -   This is fetched once the status is "COMPLETED".
    -   Expected Response: JSON containing the primary output of the function (e.g., an `images` array)
      and often repeats or includes definitive `metrics` and/or `cost`.
      Fal.ai functions typically wrap their output in a "response" key, so the image data might be
      at `final_result.response.images`.

*** Cost Calculation Logic ***
The module attempts to determine cost in the following order of preference:
1.  **Direct `cost` field:** Looks for a `cost` key in the final API response (either from the
    last status poll if it contained the full result upon completion, or from fetching the dedicated response URL).
2.  **`metrics.inference_time`:** If a direct cost is not found, it looks for `metrics.inference_time`
    (in seconds) in the API response. This duration is then multiplied by the configured
    `FAL_GPT_IMAGE_1_PRICE_PER_SECOND` to estimate the cost.
3.  **Total Job Wall-Clock Time (Fallback):** If neither direct cost nor `inference_time` is available from the API,
    the module falls back to using the total measured wall-clock time from job submission to result retrieval
    for cost estimation. This is the least accurate method and is used as a last resort.

*** Key Module Features ***
-   Direct Fal.ai REST API interaction using `aiohttp`.
-   Robust polling mechanism for asynchronous job completion.
-   Prioritized extraction of official cost/billing metrics from API responses.
-   Async/await for non-blocking operations.
-   Configurable retries for job submission.
-   Detailed logging of the generation process.

*** Environment Variables ***
-   `FAL_KEY`: Required. Fal.ai API key in `key_id:key_secret` format.
-   `FALAI_IMAGE_MODEL_ID`: Optional. The Fal.ai application/model ID.
    Defaults to "fal-ai/gpt-image-1/text-to-image".
-   `FAL_GPT_IMAGE_1_PRICE_PER_SECOND`: Optional. The estimated price per second of inference time
    for the target model, used if only duration is available from the API.

*** Dependencies ***
-   `aiohttp`: For making asynchronous HTTP requests.
-   `python-dotenv`: For loading environment variables from a .env file.
"""

import os
import sys
import logging
import time
import asyncio
import base64
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, Any, Optional
import json

from dotenv import load_dotenv
import aiohttp

# --- Configuration ---
load_dotenv()
FAL_API_KEY = os.getenv("FAL_KEY")
FALAI_IMAGE_MODEL_ID = os.getenv("FALAI_IMAGE_MODEL", "fal-ai/gpt-image-1/text-to-image")

FAL_GPT_IMAGE_1_PRICE_PER_SECOND_DEFAULT = "0.0016584" # Estimate derived from Fal.ai playground observations
FAL_GPT_IMAGE_1_PRICE_PER_SECOND = Decimal(os.getenv("FAL_GPT_IMAGE_1_PRICE_PER_SECOND", FAL_GPT_IMAGE_1_PRICE_PER_SECOND_DEFAULT))

FAL_QUEUE_BASE_URL = "https://queue.fal.run"
DEFAULT_SUBMIT_TIMEOUT_SEC = 30         # Timeout for the initial job submission POST request.
DEFAULT_POLLING_INTERVAL_SEC = 2        # How often to poll the status URL.
DEFAULT_POLLING_TIMEOUT_SEC = 360       # Max time to wait for job completion via polling (increased from 300).
DEFAULT_RESULT_TIMEOUT_SEC = 60         # Timeout for fetching the final result after COMPLETED status.
DEFAULT_API_RETRY_COUNT_FAL = 1         # Retries for initial job submission only.

# Default image parameters for testing or direct use if not overridden by caller
DEFAULT_IMAGE_SIZE_FAL_TEST = "1024x1024"
DEFAULT_IMAGE_QUALITY_FAL_TEST = "low"
DEFAULT_IMAGE_BACKGROUND_FAL_TEST = "opaque"

# Default image parameters for general use (can be overridden by caller)
DEFAULT_IMAGE_SIZE_FAL = "1024x1024"    # Standard OpenAI DALL-E 3 size
DEFAULT_IMAGE_QUALITY_FAL = "medium"    # Corresponds to standard "dall-e-3" if Fal maps quality
DEFAULT_IMAGE_BACKGROUND_FAL = "opaque"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("aiohttp").setLevel(logging.WARNING) # Keep aiohttp's own logs quieter

class FalApiError(Exception):
    """Custom exception for Fal REST API interaction errors."""
    def __init__(self, message, status_code=None, error_body=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_body = str(error_body) # Ensure error_body is string for safe printing

    def __str__(self):
        # Provides a concise string representation for logging.
        return (f"{super().__str__()} "
                f"(Status: {self.status_code if self.status_code else 'N/A'}, "
                f"Body: {self.error_body[:200]}{'...' if len(self.error_body) > 200 else ''})")

class PromptToImageFalRest:
    """
    Generates images from prompts using Fal.ai's REST queue API for accurate cost/metric tracking.
    """
    def __init__(self, api_key: Optional[str] = None, model_id: str = FALAI_IMAGE_MODEL_ID,
                 price_per_second: Decimal = FAL_GPT_IMAGE_1_PRICE_PER_SECOND,
                 submit_retries: int = DEFAULT_API_RETRY_COUNT_FAL, polling_timeout: int = DEFAULT_POLLING_TIMEOUT_SEC):
        self.api_key = api_key or FAL_API_KEY
        if not self.api_key or ':' not in self.api_key: # Fal keys are typically key_id:key_secret
            raise ValueError("Fal API key not found or in incorrect format ('key_id:key_secret'). Check FAL_KEY environment variable.")
        
        self.model_id = model_id
        self.price_per_second = price_per_second
        self.submit_retries = submit_retries
        self.polling_timeout = polling_timeout
        self.submit_url = f"{FAL_QUEUE_BASE_URL}/{self.model_id}"
        # Base headers for authenticated Fal.ai API calls. Content-Type added by _make_request for POST.
        self._base_headers = {"Authorization": f"Key {self.api_key}", "Accept": "application/json"}

        # Tracking metrics
        self.total_requests = 0; self.total_successful_requests = 0; self.total_failed_requests = 0
        self.total_accumulated_cost = Decimal("0.0")
        self.total_api_call_duration_sum_successful = 0.0 # Sum of wall-clock time for successful generate_image calls
        self.total_billing_duration_sec_sum_successful = 0.0 # Sum of API-reported billing durations for successful calls
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"PromptToImageFalRest (v4.0.0) initialized. Model ID: {self.model_id}. Price/Sec (for calc): ${self.price_per_second:.8f}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Ensures an active aiohttp.ClientSession is available."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        """Closes the aiohttp.ClientSession if it's open."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("Aiohttp session closed for PromptToImageFalRest.")

    def _calculate_cost_from_duration(self, duration_sec: float) -> Decimal:
        """Calculates cost based on duration and configured price per second."""
        if duration_sec <= 0: return Decimal("0.0")
        try:
            return (Decimal(str(duration_sec)) * self.price_per_second).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        except InvalidOperation: # Handles non-finite float values if they somehow occur
            logger.error(f"Invalid duration value for cost calculation: {duration_sec}")
            return Decimal("0.0")

    async def _make_request(self, method: str, url: str, timeout_sec: int, 
                            is_json_response: bool = True, data: Optional[Any] = None, 
                            custom_headers: Optional[Dict[str,str]] = None) -> Any:
        """
        Makes an asynchronous HTTP request.
        Handles common errors, timeouts, and response parsing.
        """
        session = await self._get_session()
        
        headers_to_use = self._base_headers.copy()
        if custom_headers is not None: # If an empty dict {} is passed, it effectively clears auth for public URLs
            headers_to_use = custom_headers 
        
        if method.upper() == "POST" and data is not None:
            if 'Content-Type' not in headers_to_use: # Only add if not already specified in custom_headers
                 headers_to_use['Content-Type'] = 'application/json'
        
        try:
            async with session.request(method, url, timeout=aiohttp.ClientTimeout(total=timeout_sec), data=data, headers=headers_to_use) as response:
                if not (200 <= response.status < 300): # Not a successful HTTP status
                    response_text = await response.text()
                    logger.error(f"Fal API Error: {method} {url} - Status {response.status} - Body: {response_text[:500]}")
                    raise FalApiError(f"API request to {url} failed with status {response.status}", 
                                      status_code=response.status, error_body=response_text)
                
                if not is_json_response: # Expecting binary data (e.g., image)
                    return await response.read() 

                # Expecting JSON response
                response_text = await response.text() 
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Fal API {method} {url} - Status {response.status} returned non-JSON text that failed to parse: {response_text[:200]}. Error: {json_err}")
                    raise FalApiError(f"Failed to decode JSON response from {url}. Status: {response.status}", 
                                      status_code=response.status, error_body=response_text)
        
        except asyncio.TimeoutError:
            logger.error(f"Fal API Timeout: {method} {url} after {timeout_sec}s")
            raise FalApiError(f"API request to {url} timed out after {timeout_sec}s", status_code=408) # Using 408 for timeout
        except aiohttp.ClientError as e: # Covers connection errors, etc.
            logger.error(f"Fal API ClientError: {method} {url} - {type(e).__name__}: {e}")
            raise FalApiError(f"API client error ({type(e).__name__}) for {url}: {e}")


    async def generate_image(self, prompt: str, output_image_path: Path, 
                             size: str = DEFAULT_IMAGE_SIZE_FAL, quality: str = DEFAULT_IMAGE_QUALITY_FAL, 
                             background: str = DEFAULT_IMAGE_BACKGROUND_FAL, 
                             semaphore: Optional[asyncio.Semaphore] = None) -> Dict[str, Any]:
        """
        Orchestrates the image generation process: submit, poll, get result, process.
        """
        if semaphore: await semaphore.acquire()

        overall_start_time = time.monotonic()
        self.total_requests += 1
        log_prefix = f"FalImgRST ({self.model_id.split('/')[-1]} -> {output_image_path.name})"
        
        result_dict: Dict[str, Any] = {
            "status": "failed", "image_path": None, "cost": 0.0, "api_call_duration_sec": 0.0, 
            "billing_duration_sec": 0.0, "revised_prompt": None, "error_message": "Process initiated.",
            "fal_request_id": None, "raw_final_response_snippet": None
        }

        try: 
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k:v for k,v in {"prompt":prompt,"image_size":size,"quality":quality,"background":background,"num_images":1}.items() if v is not None}

            # --- 1. Submit Job ---
            fal_request_id, status_url, response_url_from_submit = None, None, None
            submit_attempts = 0
            while submit_attempts <= self.submit_retries:
                try:
                    logger.debug(f"{log_prefix}: Submitting job (Attempt {submit_attempts + 1})")
                    submit_data = await self._make_request("POST", self.submit_url, DEFAULT_SUBMIT_TIMEOUT_SEC, data=json.dumps(payload))
                    fal_request_id = submit_data.get("request_id")
                    status_url = submit_data.get("status_url")
                    response_url_from_submit = submit_data.get("response_url")
                    if not all([fal_request_id, status_url, response_url_from_submit]):
                        raise FalApiError("Submission response missing critical fields (request_id, status_url, or response_url).", error_body=submit_data)
                    logger.info(f"{log_prefix}: Job submitted. Request ID: {fal_request_id}.")
                    result_dict["fal_request_id"] = fal_request_id
                    break # Successful submission
                except FalApiError as e:
                    submit_attempts += 1
                    logger.warning(f"{log_prefix}: Submission failed (Attempt {submit_attempts}): {e}")
                    if submit_attempts > self.submit_retries:
                        result_dict["error_message"] = f"Failed to submit job after {submit_attempts} attempts: {e}"
                        raise # Propagate to outer try-finally, which will then populate result_dict before returning
                    await asyncio.sleep(2**(submit_attempts-1)) # Exponential backoff
            
            # --- 2. Poll Status ---
            polling_start_time = time.monotonic()
            final_status_data = None # Store data from the COMPLETED status poll
            while True: 
                if time.monotonic() - polling_start_time > self.polling_timeout:
                    raise FalApiError(f"Polling timed out for request {fal_request_id} after {self.polling_timeout}s.", status_code=408) # Poll timeout
                
                try:
                    # Use "?logs=0" to keep status poll response lean unless debugging Fal function itself.
                    status_data = await self._make_request("GET", status_url + "?logs=0", DEFAULT_SUBMIT_TIMEOUT_SEC) 
                    current_status = status_data.get("status")
                    logger.info(f"{log_prefix}: Request {fal_request_id} status: {current_status} (Queue Pos: {status_data.get('queue_position','N/A')})")

                    if current_status == "COMPLETED":
                        final_status_data = status_data # This data might contain metrics/cost.
                        break # Exit polling loop
                    elif current_status in ["IN_QUEUE", "IN_PROGRESS"]:
                        await asyncio.sleep(DEFAULT_POLLING_INTERVAL_SEC)
                    elif current_status in ["FAILED", "CANCELLED", "ERROR"]: # Terminal failure states from Fal
                        raise FalApiError(f"Job {fal_request_id} ended with Fal status: {current_status}", 
                                          status_code=500, # Generic server-side type error from Fal's perspective
                                          error_body=str(status_data)[:500]) # Keep error concise
                    else: # Unknown or unexpected status from Fal
                        raise FalApiError(f"Job {fal_request_id} returned unknown status from Fal: {current_status}", 
                                          error_body=str(status_data)[:500])
                except FalApiError as e_poll: # Error during a poll GET request itself
                    logger.warning(f"{log_prefix}: Error during status poll for {fal_request_id}: {e_poll}. Retrying poll after delay.")
                    await asyncio.sleep(DEFAULT_POLLING_INTERVAL_SEC * 2) # Wait a bit longer if poll GET fails
            
            # --- 3. Fetch Final Result ---
            logger.info(f"{log_prefix}: Job {fal_request_id} COMPLETED. Fetching final result from {response_url_from_submit}")
            final_result_data = await self._make_request("GET", response_url_from_submit, DEFAULT_RESULT_TIMEOUT_SEC)
            result_dict["raw_final_response_snippet"] = str(final_result_data)[:1000]
            
            # --- 4. Extract Data, Cost, and Save Image ---
            extracted_cost, billing_duration_sec, cost_source = Decimal("0.0"), 0.0, "calculated"
            
            # Consolidate data for extraction: prefer final_result_data, fallback to final_status_data
            data_to_parse_metrics_from = final_result_data if isinstance(final_result_data, dict) else {}
            
            api_cost_value = data_to_parse_metrics_from.get('cost')
            if api_cost_value is None and isinstance(final_status_data, dict): # Check status data if not in final result
                api_cost_value = final_status_data.get('cost')

            api_metrics_object = data_to_parse_metrics_from.get('metrics')
            if api_metrics_object is None and isinstance(final_status_data, dict): # Check status data
                api_metrics_object = final_status_data.get('metrics')

            if api_cost_value is not None: # Direct cost found
                try: extracted_cost = Decimal(str(api_cost_value)); cost_source = "direct_api_cost"
                except InvalidOperation: logger.warning(f"{log_prefix}: Invalid 'cost' value from API: {api_cost_value}")
            
            if isinstance(api_metrics_object, dict) and "inference_time" in api_metrics_object:
                try: billing_duration_sec = float(api_metrics_object["inference_time"])
                except (ValueError, TypeError): logger.warning(f"{log_prefix}: Invalid 'inference_time' value: {api_metrics_object.get('inference_time')}")
            
            # Finalize cost calculation
            if cost_source == "direct_api_cost":
                logger.info(f"{log_prefix}: Cost directly from API: ${extracted_cost:.6f}. API Billing Duration (if provided): {billing_duration_sec:.3f}s")
            elif billing_duration_sec > 0: # Calculate cost from API-reported billing duration
                extracted_cost = self._calculate_cost_from_duration(billing_duration_sec)
                logger.info(f"{log_prefix}: Cost calculated from API billing duration: ${extracted_cost:.6f} (Duration: {billing_duration_sec:.3f}s)")
            else: # Fallback: no direct cost, no API billing duration. Use total job wall-clock time.
                current_job_duration_wc = time.monotonic() - overall_start_time 
                extracted_cost = self._calculate_cost_from_duration(current_job_duration_wc)
                billing_duration_sec = current_job_duration_wc # Record this wall-clock time as the billing_duration
                cost_source = "calculated_total_job_wallclock_time"
                logger.warning(f"{log_prefix}: No direct cost or API inference_time. Cost estimated using total job wall-clock time ({current_job_duration_wc:.3f}s): ${extracted_cost:.6f}")
            
            result_dict.update({"cost": float(extracted_cost), "billing_duration_sec": billing_duration_sec})

            # Extract image data: Fal often wraps actual output in a "response" key
            actual_response_content = data_to_parse_metrics_from.get("response", data_to_parse_metrics_from) # Default to data_to_parse anw
            if not isinstance(actual_response_content, dict): 
                raise FalApiError(f"Expected dictionary for actual response content, got {type(actual_response_content)}", 
                                  error_body=str(actual_response_content)[:200])
            
            images_list = actual_response_content.get("images")
            result_dict["revised_prompt"] = actual_response_content.get("revised_prompt") # Common for DALL-E type models
            if not images_list or not isinstance(images_list, list) or not images_list[0]: 
                raise FalApiError("No 'images' array found in the final response content.", 
                                  error_body=str(actual_response_content)[:200])

            image_info = images_list[0]
            image_b64_data, image_url_data = image_info.get("b64_json"), image_info.get("url")
            image_bytes_content = None

            if image_b64_data:
                try: image_bytes_content = base64.b64decode(image_b64_data)
                except Exception as e_b64: logger.warning(f"{log_prefix}: Base64 decoding error: {e_b64}. Will attempt URL download if available."); image_bytes_content=None
            
            if not image_bytes_content and image_url_data: # If b64 failed or not present, try URL
                logger.info(f"{log_prefix}: Downloading image from URL: {image_url_data}")
                # For public image URLs (like from Fal CDN), no auth headers are typically needed or wanted.
                image_bytes_content = await self._make_request("GET", image_url_data, DEFAULT_RESULT_TIMEOUT_SEC, 
                                                             is_json_response=False, custom_headers={}) # Pass empty dict for no auth
            
            if not image_bytes_content: raise FalApiError("Failed to obtain image bytes from b64 or URL.")
            
            with open(output_image_path, "wb") as f: f.write(image_bytes_content)
            logger.info(f"{log_prefix}: Image successfully saved to {output_image_path}")
            
            # Mark as overall success
            result_dict.update({"status": "success", "image_path": str(output_image_path), "error_message": None})
            self.total_successful_requests += 1
            self.total_accumulated_cost += extracted_cost
            self.total_billing_duration_sec_sum_successful += billing_duration_sec # Add API reported/calculated billing time
        
        except FalApiError as e_api: # Catch specific API errors from our logic or _make_request
            logger.error(f"{log_prefix}: Fal API processing chain error: {e_api}")
            # Ensure error_message in result_dict is updated if it was a more generic one before
            if not result_dict.get("error_message") or result_dict.get("error_message") == "Process initiated.":
                 result_dict["error_message"] = str(e_api)
        except Exception as e_general: # Catch any other unexpected errors during the process
            logger.error(f"{log_prefix}: Unexpected error during image generation: {type(e_general).__name__} - {e_general}", exc_info=True)
            if not result_dict.get("error_message") or result_dict.get("error_message") == "Process initiated.":
                result_dict["error_message"] = f"Unexpected error: {type(e_general).__name__} - {e_general}"
        finally:
            current_wall_clock_duration = time.monotonic() - overall_start_time
            result_dict["api_call_duration_sec"] = current_wall_clock_duration # Record total wall-clock time
            
            # If successful, add wall-clock time to its sum
            if result_dict.get("status") == "success": 
                self.total_api_call_duration_sum_successful += current_wall_clock_duration
            
            # If request failed for any reason AND it hasn't been counted as FAILED yet
            if result_dict.get("status") == "failed" and \
               self.total_requests > (self.total_successful_requests + self.total_failed_requests):
                self.total_failed_requests += 1
                
            if semaphore: semaphore.release()
            
        return result_dict

    def get_stats(self) -> Dict[str, Any]:
        """Returns a dictionary of accumulated operational statistics."""
        return {
            "total_requests_attempted": self.total_requests, 
            "total_requests_successful": self.total_successful_requests, 
            "total_requests_failed": self.total_failed_requests, 
            "total_accumulated_cost_successful": self.total_accumulated_cost, 
            "total_api_call_duration_sum_successful_sec": self.total_api_call_duration_sum_successful, 
            "total_billing_duration_sum_successful_sec": self.total_billing_duration_sec_sum_successful
        }

# --- Task Runner Helper ---
async def run_fal_image_gen_task_rest(handler: PromptToImageFalRest, prompt_str: str, 
                                      out_path: Path, task_id: int, sem: asyncio.Semaphore, 
                                      size_str: str, quality_str: str, background_str: str) -> Dict[str, Any]:
    """Helper coroutine to run a single image generation task."""
    logger.info(f"--- Fal REST Task {task_id}: Waiting for semaphore for prompt '{prompt_str[:30]}...' ---")
    result = await handler.generate_image(
        prompt=prompt_str, output_image_path=out_path, size=size_str, 
        quality=quality_str, background=background_str, semaphore=sem
    )
    
    error_msg_val = result.get('error_message')
    error_msg_display = 'N/A'
    if error_msg_val is not None:
        error_msg_str = str(error_msg_val)
        error_msg_display = (error_msg_str[:100] + '...' if len(error_msg_str) > 100 else error_msg_str)
    
    logger.info(f"--- Fal REST Task {task_id}: Finished with Status: {result.get('status')}. Error: {error_msg_display} ---")
    return result
    
# --- Main Execution Block (for testing this module standalone) ---
async def main_test_fal_rest():
    """Main function to test the PromptToImageFalRest class."""
    print("--- Running PromptToImageFalRest Test (V4.0.0 - REST API) ---")
    if not FAL_API_KEY or ':' not in FAL_API_KEY:
        print("ERROR: FAL_KEY not found or in incorrect format in .env file.", file=sys.stderr)
        return
    
    print(f"Target Fal Model ID (appId): {FALAI_IMAGE_MODEL_ID}")
    print(f"Configured Price/Sec (for fallback calculation): ${FAL_GPT_IMAGE_1_PRICE_PER_SECOND:.8f}")
    
    image_handler_rest = None
    try:
        # Initialize with 0 submission retries for faster test feedback on critical errors.
        image_handler_rest = PromptToImageFalRest(submit_retries=0, polling_timeout=DEFAULT_POLLING_TIMEOUT_SEC)
        
        test_prompts = [
            "V4.0 Test: A photorealistic image of a majestic wolf howling at a luminous full moon in a snowy forest, gpt-image-1 style.",
            "V4.0 Test: An oil painting of a quaint Parisian street cafe scene at dusk, with warm lights and cobblestone streets, gpt-image-1 style."
        ]
        num_tasks = len(test_prompts)

        test_output_dir = Path("temp_files") / "image_test_fal_rest_v4_0_0"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output images will be saved to: {test_output_dir.resolve()}")
        
        semaphore = asyncio.Semaphore(2) # Limit concurrency for testing
        tasks = []
        print(f"\nCreating {num_tasks} Fal REST image generation task(s) with concurrency limit of {semaphore._value}...")
        for i, prompt_text in enumerate(test_prompts):
            output_path_obj = test_output_dir / f"test_image_v400_{i+1}.png"
            task = asyncio.create_task(
                run_fal_image_gen_task_rest(
                    image_handler_rest, prompt_text, output_path_obj, 
                    task_id = i+1, sem = semaphore,
                    size_str = DEFAULT_IMAGE_SIZE_FAL_TEST, 
                    quality_str = DEFAULT_IMAGE_QUALITY_FAL_TEST, 
                    background_str = DEFAULT_IMAGE_BACKGROUND_FAL_TEST
                ), 
                name=f"FalRestImageTask_v400_{i+1}" # Name the asyncio task
            )
            tasks.append(task)
            
        batch_start_time = time.monotonic()
        print(f"Waiting for {num_tasks} task(s) to complete...")
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
        batch_total_duration = time.monotonic() - batch_start_time
        print(f"--- Batch processing finished in {batch_total_duration:.3f}s ---")

        print("\n--- Individual Task Results (V4.0.0) ---")
        successful_task_count = 0
        batch_total_cost_this_run = Decimal("0.0")

        for i, res_or_exc in enumerate(results_or_exceptions):
            task_name = tasks[i].get_name() if i < len(tasks) else f"Task {i+1}"
            current_prompt = test_prompts[i] if i < len(test_prompts) else "N/A"
            print(f"\n--- Result for {task_name} (Prompt: \"{current_prompt[:40]}...\") ---")
            # For robust debugging, print the raw result or exception
            # print(f"  DEBUG: Raw result/exception ({type(res_or_exc)}): {str(res_or_exc)[:300]}{'...' if len(str(res_or_exc)) > 300 else ''}")

            if res_or_exc is None: # Should ideally not happen with return_exceptions=True
                print(f"  Status: CRITICAL_ERROR - Task returned None.")
            elif isinstance(res_or_exc, Exception): 
                print(f"  Status: FAILED_WITH_EXCEPTION")
                print(f"  Error Type: {type(res_or_exc).__name__}")
                print(f"  Error Details: {str(res_or_exc)}")
            elif isinstance(res_or_exc, dict):
                status = res_or_exc.get('status', 'Status N/A')
                api_time = res_or_exc.get('api_call_duration_sec', 0.0)
                print(f"  Status: {status.upper()}")
                print(f"  Wall-Clock API Call Time: {api_time:.3f}s")
                
                cost_val = Decimal(str(res_or_exc.get('cost', '0.0')))
                billing_time_val = res_or_exc.get('billing_duration_sec', 0.0)
                print(f"  Reported Cost: ${cost_val:.6f}")
                print(f"  Reported Billing Duration: {billing_time_val:.3f}s")
                
                if status == "success": 
                    successful_task_count += 1
                    batch_total_cost_this_run += cost_val
                    print(f"  Image Path: {res_or_exc.get('image_path', 'N/A')}")
                    if res_or_exc.get('revised_prompt'): 
                        print(f"  Revised Prompt: {res_or_exc.get('revised_prompt')}")
                else: 
                    print(f"  Error Message: {res_or_exc.get('error_message', 'No specific error message.')}")
                
                print(f"  Fal Request ID: {res_or_exc.get('fal_request_id', 'N/A')}")
                # print(f"  Raw Final Resp Snippet: {res_or_exc.get('raw_final_response_snippet', 'N/A')[:70]}...") # Optional detailed log
            else: 
                print(f"  Status: UNEXPECTED_RESULT_TYPE ({type(res_or_exc)})")
        
        print(f"\n--- Test Run Summary (V4.0.0) ---")
        print(f"Total Wall-Clock Time for Batch: {batch_total_duration:.3f}s")
        print(f"Tasks Attempted: {len(tasks)}, Successful: {successful_task_count}")
        print(f"Total Cost for Successful Tasks in this Batch: ${batch_total_cost_this_run:.6f}")
        
        if image_handler_rest: 
            stats = image_handler_rest.get_stats()
            print("\n--- Handler Cumulative Statistics ---")
            for key, value in stats.items():
                if isinstance(value, Decimal): print(f"  {key.replace('_', ' ').title()}: ${value:.6f}")
                elif isinstance(value, float): print(f"  {key.replace('_', ' ').title()}: {value:.3f}s")
                else: print(f"  {key.replace('_', ' ').title()}: {value}")
            
    except Exception as e_main_test: # Catch errors from main_test_fal_rest setup or teardown
        logger.error(f"Error in main_test_fal_rest: {e_main_test}", exc_info=True)
        print(f"CRITICAL ERROR in test harness: {e_main_test}")
    finally:
        if image_handler_rest: 
            await image_handler_rest.close_session()

if __name__ == "__main__":
    # Standard asyncio setup for Windows compatibility
    if sys.platform == "win32" and sys.version_info >= (3,8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_test_fal_rest())
    except KeyboardInterrupt: 
        print("\nTest run interrupted by user (KeyboardInterrupt).")
    except Exception as e_asyncio_run: # Catch any other very top-level errors
        print(f"Critical error during asyncio.run execution: {type(e_asyncio_run).__name__}: {e_asyncio_run}")
        logger.error("Critical error during asyncio.run", exc_info=True)

