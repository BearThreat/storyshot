# File: prompt2image_fal_.py
"""
Prompt-to-Image Module using Fal.ai (prompt2image_fal_.py)
Version: 4.5.0 (Final - Documented Insights, QoL improvements)
Date: 2025-05-18

This module provides a Python interface for generating and editing images using
Fal.ai's "GPT-Image-1" model (and its associated sub-applications). It encapsulates
the interaction logic, handling both direct REST API calls for some functionalities
and leveraging the `fal-client` library for others, based on hard-won insights
into Fal.ai's API behavior.

*** Key Functionalities & Design Choices ***

1.  **Hybrid API Interaction Model:**
    *   **Text-to-Image (`generate_image`):** Interacts via direct REST API calls
        to the Fal.ai queue system (e.g., `https://queue.fal.run/{app_id}`).
        This method allows for direct parsing of queue responses, including metrics
        and cost information when available.
    *   **Image Editing & Masked Inpainting (`edit_image`):** Utilizes the
        `fal-client` Python library. This is crucial as certain Fal.ai
        endpoints, specifically `fal-ai/gpt-image-1/edit-image`, do not
        reliably support result retrieval through the generic REST queue polling
        mechanism used for text-to-image. The `fal-client` handles the
        specifics of communication for these serverless function-like endpoints.

2.  **Application ID (`app_id`) Management:**
    *   The base application ID (e.g., "fal-ai/gpt-image-1") is configured via
        the `FALAI_IMAGE_MODEL_ID` environment variable or defaults.
    *   For text-to-image, this `app_id` is used directly.
    *   For image editing, the script derives the specific edit application ID
        (e.g., "fal-ai/gpt-image-1/edit-image") by appending "/edit-image" to
        a cleaned base ID, ensuring compatibility with `fal-client`'s expectations.

3.  **Image URL Handling for Editing:**
    *   A crucial insight gained: For `edit_image` operations using the
        `fal-ai/gpt-image-1/edit-image` endpoint, providing image URLs hosted
        on Fal.ai's own storage (e.g., `https://v3.fal.media/...`, obtained via
        `fal_client.upload_file_async`) is significantly more reliable than
        using arbitrary external URLs. External URLs sometimes resulted in
        "File download error" from Fal's backend during processing, even if the
        job initially showed as `COMPLETED`.

4.  **Cost Calculation and Metrics:**
    *   For text-to-image (direct REST), the script attempts to extract `cost`
        and `metrics.inference_time` directly from the API response.
    *   For image editing (via `fal-client`), such detailed metrics are often
        not available in the `fal-client` result. Cost estimation for these
        operations primarily relies on the total wall-clock time of the API call,
        which is less precise but provides a reasonable estimate.
    *   TODO: Clearly differentiate when a cost is an estimate or precise.

5.  **Asynchronous Operations and Event Loop Management:**
    *   All API interactions are asynchronous (`async/await`).
    *   The test suite is structured to run all test functions under a single
        `asyncio.run()` call. This was a key fix for "Event loop is closed"
        errors that could occur when `fal_client` (using `httpx` internally)
        was used across multiple, separate `asyncio.run()` invocations.

6.  **Robust Error Handling and Retries:**
    *   Includes retries for initial job submission (both REST and `fal-client`).
    *   Handles various `fal_client` status objects (e.g., `Completed`,
        `InProgress`, `Queued`, base `FalError`) during polling.
    *   Logs detailed error messages and relevant information.

*** Hard-Won Insights (Do Not Forget!) ***

*   **Not all Fal "applications" behave the same via REST Queue API:** Some sub-paths or
    specialized functions (like `.../edit-image`) require `fal-client` for reliable
    status/result handling, as their queue interaction is abstracted. Direct
    REST polling on `status_url` / `response_url` from these might fail (e.g., 404).
*   **`fal-client` is the "official" way for certain endpoints:** The docs for
    `fal-ai/gpt-image-1/edit-image` emphasize `fal-client`.
*   **External URL reliability for `image_urls` in `edit-image`:** Fal's backend may
    struggle with some external URLs. Using Fal-hosted URLs (uploaded via
    `fal_client.upload_file_async`) is much safer for `edit-image` inputs.
    The "File download error" (422 Unprocessable Entity) received from `fal-client`
    when trying to get *results* was a misleading symptom of this input issue.
*   **`asyncio.run()` scope:** For multiple async test suites using libraries like
    `fal-client` (which uses `httpx`), run them under a single `asyncio.run()`
    to avoid "Event loop is closed" issues related to shared/global client resources.
*   **`fal-client` status objects:** Need to be checked using `isinstance` (e.g.,
    `isinstance(status_obj, fal_client.Completed)`), not by assuming a `.status.value`
    attribute that might not exist on all status types.
*   **Logs from `fal_client.InProgress`:** The `logs` attribute can be `None`. Always
    default to an empty list before trying to get its length or iterate.

*** Environment Variables ***
-   `FAL_KEY`: Required. Fal.ai API key (e.g., "key_id:key_secret").
-   `FALAI_IMAGE_MODEL_ID`: Optional. Defaults to "fal-ai/gpt-image-1".
    This should be the base application ID.
-   `FAL_GPT_IMAGE_1_PRICE_PER_SECOND`: Optional. Price per second for cost estimation.

*** Dependencies ***
-   `aiohttp`, `python-dotenv`, `fal-client`.
-   `Pillow`: Required only for generating dummy images/masks for testing.
"""

import os
import sys
import logging
import time
import asyncio
import base64
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, Any, Optional, List, Union, Tuple
import json
from io import BytesIO

from dotenv import load_dotenv
import aiohttp
import fal_client

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None
    logger = logging.getLogger(__name__) # Define logger early for this message
    logger.warning("Pillow (PIL) not installed. Test features for dummy image/mask generation will be disabled.")


# --- Configuration ---
load_dotenv()
FAL_API_KEY = os.getenv("FAL_KEY")
FALAI_IMAGE_MODEL_ID_DEFAULT = "fal-ai/gpt-image-1"
FALAI_IMAGE_MODEL_ID = os.getenv("FALAI_IMAGE_MODEL", FALAI_IMAGE_MODEL_ID_DEFAULT)


FAL_GPT_IMAGE_1_PRICE_PER_SECOND_DEFAULT = "0.0016584"
FAL_GPT_IMAGE_1_PRICE_PER_SECOND = Decimal(os.getenv("FAL_GPT_IMAGE_1_PRICE_PER_SECOND", FAL_GPT_IMAGE_1_PRICE_PER_SECOND_DEFAULT))

FAL_QUEUE_BASE_URL = "https://queue.fal.run"
DEFAULT_SUBMIT_TIMEOUT_SEC = 30
DEFAULT_POLLING_INTERVAL_SEC = 5 
DEFAULT_POLLING_TIMEOUT_SEC = 420 # Increased for potentially long Fal operations
DEFAULT_RESULT_TIMEOUT_SEC = 60
DEFAULT_API_RETRY_COUNT_FAL = 1
DEFAULT_UPLOAD_RETRY_COUNT_FAL = 1
DEFAULT_UPLOAD_RETRY_DELAY_SEC = 2

DEFAULT_IMAGE_SIZE_FAL_TEST = "1024x1024"
DEFAULT_IMAGE_QUALITY_FAL_TEST = "medium"
DEFAULT_IMAGE_BACKGROUND_FAL_TEST = "opaque"

DEFAULT_IMAGE_SIZE_FAL = "1024x1024"
DEFAULT_IMAGE_QUALITY_FAL = "medium"
DEFAULT_IMAGE_BACKGROUND_FAL = "opaque"
DEFAULT_NUM_IMAGES_FAL = 1

# --- Setup Paths and Logging ---
BASE_TEMP_DIR = Path("temp_files")
INTERNAL_TEST_IMAGE_DIR = BASE_TEMP_DIR / "_internal_test_images"
INTERNAL_TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Use a consistent version number for test output directories
TEST_VERSION_SUFFIX = "v4_5_0_final"
TEXT2IMG_TEST_OUTPUT_DIR = BASE_TEMP_DIR / f"image_test_fal_rest_{TEST_VERSION_SUFFIX}_text2img"
TEXT2IMG_TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDIT_BASE_IMAGE_PATH = TEXT2IMG_TEST_OUTPUT_DIR / f"test_text2img_{TEST_VERSION_SUFFIX}_1_for_edit.png"


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
    ]
)
logger = logging.getLogger(__name__) # Main module logger
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING) # fal-client uses httpx
logging.getLogger("fal_client").setLevel(logging.INFO) # Set fal_client to INFO to see its actions


class FalApiError(Exception):
    def __init__(self, message, status_code=None, error_body=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_body = str(error_body)
    def __str__(self):
        return (f"{super().__str__()} "
                f"(Status: {self.status_code if self.status_code else 'N/A'}, "
                f"Body: {self.error_body[:200]}{'...' if len(self.error_body) > 200 else ''})")

class PromptToImageFalRest:
    def __init__(self, api_key: Optional[str] = None, model_id: str = FALAI_IMAGE_MODEL_ID,
                 price_per_second: Decimal = FAL_GPT_IMAGE_1_PRICE_PER_SECOND,
                 submit_retries: int = DEFAULT_API_RETRY_COUNT_FAL, 
                 upload_retries: int = DEFAULT_UPLOAD_RETRY_COUNT_FAL,
                 polling_timeout: int = DEFAULT_POLLING_TIMEOUT_SEC):
        self.api_key = api_key or FAL_API_KEY
        if not self.api_key or ':' not in self.api_key:
            raise ValueError("Fal API key invalid or not found. Check FAL_KEY environment variable (format: 'key_id:key_secret').")
        
        self.configured_model_id = model_id # ID from .env or default
        self.price_per_second = price_per_second
        self.submit_retries_rest = submit_retries
        self.submit_retries_client = submit_retries
        self.upload_retries = upload_retries
        self.polling_timeout = polling_timeout
        
        # For direct REST generate_image, use configured_model_id as is
        self.generate_submit_url_rest = f"{FAL_QUEUE_BASE_URL}/{self.configured_model_id}"
        
        # Construct the edit_app_id for fal-client, aiming for "fal-ai/gpt-image-1/edit-image"
        base_for_edit = self.configured_model_id
        if "fal-ai/gpt-image-1" in base_for_edit: # Check if the known base is part of it
            # If it is, ensure we use the exact "fal-ai/gpt-image-1" part
            parts = base_for_edit.split("fal-ai/gpt-image-1")
            base_for_edit = "fal-ai/gpt-image-1" # Reconstruct the known base
        else:
            # If "fal-ai/gpt-image-1" is not in the configured_model_id, it's an unexpected configuration.
            # fal-ai/gpt-image-1/edit-image is the documented endpoint.
            logger.warning(
                f"Configured model_id '{self.configured_model_id}' does not appear to be based on 'fal-ai/gpt-image-1'. "
                f"Forcing edit operations to target the documented 'fal-ai/gpt-image-1/edit-image' app ID. "
                f"If this is incorrect for your specific Fal setup, adjust FALAI_IMAGE_MODEL_ID or the hardcoded target."
            )
            base_for_edit = "fal-ai/gpt-image-1" # Default to the known correct base for edit
        self.edit_app_id_for_client = f"{base_for_edit}/edit-image"


        self._base_headers_rest = {"Authorization": f"Key {self.api_key}", "Accept": "application/json"}
        # Statistics
        self.total_requests = 0; self.total_successful_requests = 0; self.total_failed_requests = 0
        self.total_accumulated_cost = Decimal("0.0")
        self.total_api_call_duration_sum_successful = 0.0
        self.total_billing_duration_sec_sum_successful = 0.0
        self._session_aiohttp: Optional[aiohttp.ClientSession] = None # For aiohttp REST calls
        logger.info(
            f"PromptToImageFalRest (v4.5.0) initialized. "
            f"Configured App ID for Generate (REST): {self.configured_model_id} -> URL: {self.generate_submit_url_rest}. "
            f"App ID for Edit (fal-client): {self.edit_app_id_for_client}."
        )

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        if self._session_aiohttp is None or self._session_aiohttp.closed: 
            self._session_aiohttp = aiohttp.ClientSession()
        return self._session_aiohttp

    async def close_sessions(self):
        if self._session_aiohttp and not self._session_aiohttp.closed: 
            await self._session_aiohttp.close()
            self._session_aiohttp = None
            logger.info("Aiohttp session closed for PromptToImageFalRest.")
        # `fal-client` manages its own `httpx.AsyncClient` internally for each call by default,
        # so no explicit session closing is typically needed for it unless a persistent client is manually created.

    def _calculate_cost_from_duration(self, duration_sec: float) -> Decimal:
        if duration_sec <= 0: return Decimal("0.0")
        try: return (Decimal(str(duration_sec)) * self.price_per_second).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        except InvalidOperation: logger.error(f"Invalid duration for cost calculation: {duration_sec}"); return Decimal("0.0")

    async def _make_rest_request(self, method: str, url: str, timeout_sec: int, is_json_response: bool=True, data: Optional[Any]=None, custom_headers: Optional[Dict[str,str]]=None) -> Any:
        session = await self._get_aiohttp_session()
        headers_to_use = self._base_headers_rest.copy()
        if custom_headers is not None: headers_to_use.update(custom_headers) # Use update
        if method.upper() == "POST" and data is not None and 'Content-Type' not in headers_to_use: headers_to_use['Content-Type'] = 'application/json'
        try:
            async with session.request(method, url, timeout=aiohttp.ClientTimeout(total=timeout_sec), data=data, headers=headers_to_use) as response:
                if not (200 <= response.status < 300):
                    response_text = await response.text(); logger.error(f"Fal API Error (REST): {method} {url} - Status {response.status} - Body: {response_text[:500]}")
                    raise FalApiError(f"API request (REST) to {url} failed with status {response.status}", status_code=response.status, error_body=response_text)
                if not is_json_response: return await response.read()
                response_text = await response.text()
                try: return json.loads(response_text)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Fal API (REST) {method} {url} - Status {response.status} returned non-JSON (len {len(response_text)}): {response_text[:200]}. Error: {json_err}")
                    raise FalApiError(f"Failed to decode JSON (REST) from {url}. Status: {response.status}", status_code=response.status, error_body=response_text)
        except asyncio.TimeoutError: logger.error(f"Fal API Timeout (REST): {method} {url} after {timeout_sec}s"); raise FalApiError(f"API request (REST) to {url} timed out after {timeout_sec}s", status_code=408)
        except aiohttp.ClientError as e: logger.error(f"Fal API ClientError (REST): {method} {url} - {type(e).__name__}: {e}"); raise FalApiError(f"API client error (REST) ({type(e).__name__}) for {url}: {e}")

    async def _upload_file_async(self, file_path: Path, log_prefix: str) -> str:
        upload_attempts = 0
        while upload_attempts <= self.upload_retries:
            try:
                logger.debug(f"{log_prefix}: Uploading local file: {file_path} (Attempt {upload_attempts + 1}/{self.upload_retries + 1})")
                start_time = time.monotonic(); uploaded_url = await fal_client.upload_file_async(str(file_path)); duration = time.monotonic() - start_time
                logger.info(f"{log_prefix}: File uploaded to {uploaded_url} ({duration:.3f}s)")
                return uploaded_url
            except Exception as e: 
                logger.warning(f"{log_prefix}: Upload attempt {upload_attempts + 1} for {file_path.name} failed: {type(e).__name__}: {e}")
                upload_attempts += 1
                if upload_attempts <= self.upload_retries: logger.info(f"{log_prefix}: Retrying upload in {DEFAULT_UPLOAD_RETRY_DELAY_SEC}s..."); await asyncio.sleep(DEFAULT_UPLOAD_RETRY_DELAY_SEC)
                else: logger.error(f"{log_prefix}: Upload failed for {file_path.name} after {self.upload_retries + 1} attempts."); raise FalApiError(f"Failed to upload file {file_path.name} after retries: {e}")
        # This line should ideally be unreachable due to the raise in the else block.
        raise FalApiError(f"Upload logic error for local file {file_path.name}.")


    async def _execute_fal_job_rest(self, submission_target_url: str, payload: Dict[str, Any], output_image_path: Path, log_prefix_extra: str="", semaphore: Optional[asyncio.Semaphore]=None) -> Dict[str, Any]:
        # ... (This method remains largely the same as it's proven for text-to-image via REST) ...
        if semaphore: await semaphore.acquire()
        overall_start_time = time.monotonic(); self.total_requests += 1
        # Log prefix uses self.configured_model_id for generate_image
        log_prefix = f"FalImgRST ({self.configured_model_id}{log_prefix_extra} -> {output_image_path.name})"
        result_dict: Dict[str, Any] = {"status": "failed", "image_path": None, "cost": 0.0, "api_call_duration_sec": 0.0, "billing_duration_sec": 0.0, "revised_prompt": None, "error_message": "Process initiated (REST).", "fal_request_id": None, "raw_final_response_snippet": None}
        try: 
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            fal_request_id, status_url, response_url = None, None, None
            submit_attempts = 0
            while submit_attempts <= self.submit_retries_rest:
                try:
                    logger.debug(f"{log_prefix}: Submitting job to {submission_target_url} (Attempt {submit_attempts + 1}). Payload: {str(payload)[:300]}...")
                    submit_data = await self._make_rest_request("POST", submission_target_url, DEFAULT_SUBMIT_TIMEOUT_SEC, data=json.dumps(payload))
                    fal_request_id = submit_data.get("request_id"); status_url = submit_data.get("status_url"); response_url = submit_data.get("response_url")
                    if not all([fal_request_id, status_url, response_url]): raise FalApiError("Submission response missing critical fields.", error_body=str(submit_data))
                    logger.info(f"{log_prefix}: Job submitted via {submission_target_url}. Request ID: {fal_request_id}. StatusURL: {status_url}"); result_dict["fal_request_id"] = fal_request_id
                    break
                except FalApiError as e:
                    submit_attempts += 1; logger.warning(f"{log_prefix}: Submission failed (Attempt {submit_attempts}): {e}")
                    if submit_attempts > self.submit_retries_rest: result_dict["error_message"] = f"REST submit failed after {submit_attempts} attempts: {e}"; raise
                    await asyncio.sleep(2**(submit_attempts-1))
            polling_start_time = time.monotonic(); final_status_data = None; self._last_poll_time_rest = 0 # type: ignore
            while True: 
                current_poll_time = time.monotonic()
                if current_poll_time - polling_start_time > self.polling_timeout: raise FalApiError(f"REST Polling timed out for {fal_request_id} after {self.polling_timeout}s.", status_code=408)
                await asyncio.sleep(max(0.1, DEFAULT_POLLING_INTERVAL_SEC - (current_poll_time - self._last_poll_time_rest))) # type: ignore
                self._last_poll_time_rest = time.monotonic() # type: ignore
                try:
                    status_data = await self._make_rest_request("GET", status_url + "?logs=0", DEFAULT_SUBMIT_TIMEOUT_SEC) 
                    current_status = status_data.get("status")
                    logger.info(f"{log_prefix}: Request {fal_request_id} status from {status_url}: {current_status} (Queue Pos: {status_data.get('queue_position','N/A')})")
                    if current_status == "COMPLETED": final_status_data = status_data; break
                    elif current_status in ["IN_QUEUE", "IN_PROGRESS"]: pass 
                    elif current_status in ["FAILED", "CANCELLED", "ERROR"]: raise FalApiError(f"Job {fal_request_id} ended with Fal status: {current_status}", status_code=500, error_body=str(status_data)[:500])
                    else: raise FalApiError(f"Job {fal_request_id} returned unknown status from Fal: {current_status}", error_body=str(status_data)[:500])
                except FalApiError as e_poll: logger.warning(f"{log_prefix}: Error during status poll ({status_url}): {e_poll}. Retrying.");
            logger.info(f"{log_prefix}: Job {fal_request_id} COMPLETED. Fetching final result from {response_url}")
            final_result_data = await self._make_rest_request("GET", response_url, DEFAULT_RESULT_TIMEOUT_SEC)
            result_dict["raw_final_response_snippet"] = str(final_result_data)[:1000]
            # Cost and metric extraction... (same as before)
            extracted_cost, billing_duration_sec, cost_source = Decimal("0.0"), 0.0, "calculated"
            data_to_parse_metrics_from = final_result_data if isinstance(final_result_data, dict) else {}
            api_cost_value = data_to_parse_metrics_from.get('cost'); api_metrics_object = data_to_parse_metrics_from.get('metrics')
            if api_cost_value is None and isinstance(final_status_data, dict): api_cost_value = final_status_data.get('cost')
            if api_metrics_object is None and isinstance(final_status_data, dict): api_metrics_object = final_status_data.get('metrics')
            if api_cost_value is not None:
                try: extracted_cost = Decimal(str(api_cost_value)); cost_source = "direct_api_cost"
                except InvalidOperation: logger.warning(f"{log_prefix}: Invalid 'cost' API value: {api_cost_value}")
            if isinstance(api_metrics_object, dict) and "inference_time" in api_metrics_object:
                try: billing_duration_sec = float(api_metrics_object["inference_time"])
                except (ValueError, TypeError): logger.warning(f"{log_prefix}: Invalid 'inference_time': {api_metrics_object.get('inference_time')}")
            if cost_source == "direct_api_cost": logger.info(f"{log_prefix}: Cost from API: ${extracted_cost:.6f}. API Billing Duration: {billing_duration_sec:.3f}s")
            elif billing_duration_sec > 0: extracted_cost = self._calculate_cost_from_duration(billing_duration_sec); logger.info(f"{log_prefix}: Cost from API duration: ${extracted_cost:.6f} (Dur: {billing_duration_sec:.3f}s)")
            else: wc_dur = time.monotonic() - overall_start_time; extracted_cost = self._calculate_cost_from_duration(wc_dur); billing_duration_sec = wc_dur; cost_source = "fallback_wallclock"; logger.warning(f"{log_prefix}: No cost/API duration. Cost from wall-clock ({wc_dur:.3f}s): ${extracted_cost:.6f}")
            result_dict.update({"cost": float(extracted_cost), "billing_duration_sec": billing_duration_sec})
            actual_resp = data_to_parse_metrics_from; img_list = actual_resp.get("images")
            if img_list is None and isinstance(actual_resp.get("response"), dict): img_list = actual_resp.get("response",{}).get("images"); result_dict["revised_prompt"] = actual_resp.get("response",{}).get("revised_prompt")
            elif img_list is not None: result_dict["revised_prompt"] = actual_resp.get("revised_prompt")
            if not img_list or not isinstance(img_list, list) or not img_list[0]: raise FalApiError("No 'images' array in final response.", error_body=str(actual_resp)[:200])
            img_info = img_list[0]; b64_data, url_data = img_info.get("b64_json"), img_info.get("url"); img_bytes = None
            if b64_data:
                try: img_bytes = base64.b64decode(b64_data)
                except Exception as e: logger.warning(f"{log_prefix}: B64 decode error: {e}. Try URL."); img_bytes=None
            if not img_bytes and url_data: logger.info(f"{log_prefix}: Downloading from URL: {url_data}"); img_bytes = await self._make_rest_request("GET", url_data, DEFAULT_RESULT_TIMEOUT_SEC, is_json_response=False, custom_headers={})
            if not img_bytes: raise FalApiError("Failed to get image bytes.")
            with open(output_image_path, "wb") as f: f.write(img_bytes); logger.info(f"{log_prefix}: Image saved to {output_image_path}")
            result_dict.update({"status": "success", "image_path": str(output_image_path), "error_message": None})
            self.total_successful_requests += 1; self.total_accumulated_cost += extracted_cost; self.total_billing_duration_sec_sum_successful += billing_duration_sec
        except FalApiError as e: logger.error(f"{log_prefix}: Fal API (REST) processing error: {e}"); result_dict["error_message"] = str(e)
        except Exception as e: logger.error(f"{log_prefix}: Unexpected error (REST): {type(e).__name__} - {e}", exc_info=True); result_dict["error_message"] = f"Unexpected (REST): {type(e).__name__} - {e}"
        finally:
            wc_dur = time.monotonic()-overall_start_time; result_dict["api_call_duration_sec"]=wc_dur
            if result_dict.get("status") == "success": self.total_api_call_duration_sum_successful += wc_dur
            if result_dict.get("status")=="failed" and self.total_requests > (self.total_successful_requests + self.total_failed_requests): self.total_failed_requests+=1
            if semaphore: semaphore.release()
        return result_dict


    async def generate_image(self, prompt: str, output_image_path: Path, size: str=DEFAULT_IMAGE_SIZE_FAL, quality: str=DEFAULT_IMAGE_QUALITY_FAL, background: str=DEFAULT_IMAGE_BACKGROUND_FAL, num_images: int=DEFAULT_NUM_IMAGES_FAL, semaphore: Optional[asyncio.Semaphore]=None) -> Dict[str, Any]:
        payload = {"prompt":prompt, "image_size":size, "quality":quality, "background":background, "num_images":num_images}
        payload = {k:v for k,v in payload.items() if v is not None}
        return await self._execute_fal_job_rest(self.generate_submit_url_rest, payload, output_image_path, log_prefix_extra=" [Text2Img]", semaphore=semaphore)

    async def edit_image(self, image_paths_or_urls: List[Union[str,Path]], prompt: str, output_image_path: Path, mask_path_or_url: Optional[Union[str,Path]]=None, size: str=DEFAULT_IMAGE_SIZE_FAL, quality: str=DEFAULT_IMAGE_QUALITY_FAL, num_images: int=DEFAULT_NUM_IMAGES_FAL, semaphore: Optional[asyncio.Semaphore]=None) -> Dict[str, Any]:
        if semaphore: await semaphore.acquire()
        overall_start_time = time.monotonic(); self.total_requests += 1
        
        target_edit_app_id = self.edit_app_id_for_client
        log_prefix = f"FalImgCLIENT ({target_edit_app_id} -> {output_image_path.name})"
        result_dict: Dict[str, Any] = {"status": "failed", "image_path": None, "cost": 0.0, "api_call_duration_sec": 0.0, "billing_duration_sec": 0.0, "revised_prompt": None, "error_message": "Process initiated (fal-client).", "fal_request_id": None, "raw_final_response_snippet": None}
        
        try:
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            upload_log_prefix = f"{log_prefix} [Upload]"
            uploaded_image_urls = [await self._upload_file_async(Path(item), upload_log_prefix) if isinstance(item,Path) or (isinstance(item,str) and not item.startswith('http')) else item for item in image_paths_or_urls]
            if not uploaded_image_urls: raise FalApiError("No valid image URLs for editing input.")

            payload = {"image_urls": uploaded_image_urls, "prompt": prompt, "image_size": size, "quality": quality, "num_images": num_images}
            if mask_path_or_url:
                mask_url = await self._upload_file_async(Path(mask_path_or_url), upload_log_prefix + "[Mask]") if isinstance(mask_path_or_url,Path) or (isinstance(mask_path_or_url,str) and not mask_path_or_url.startswith('http')) else mask_path_or_url
                if mask_url: payload["mask_url"] = mask_url; logger.info(f"{log_prefix}: Using mask_url: {mask_url}")
            payload = {k:v for k,v in payload.items() if v is not None}

            handler = None; fal_request_id = None
            for attempt in range(self.submit_retries_client + 1):
                try:
                    logger.debug(f"{log_prefix}: Submitting job via fal-client to '{target_edit_app_id}' (Attempt {attempt + 1}). Payload: {str(payload)[:300]}...")
                    handler = await fal_client.submit_async(target_edit_app_id, arguments=payload)
                    fal_request_id = handler.request_id; result_dict["fal_request_id"] = fal_request_id
                    logger.info(f"{log_prefix}: Job submitted via fal-client. App: {target_edit_app_id}, Request ID: {fal_request_id}"); break
                except Exception as e:
                    logger.warning(f"{log_prefix}: fal-client submit failed (Attempt {attempt + 1}): {type(e).__name__}: {e}")
                    if attempt >= self.submit_retries_client: result_dict["error_message"] = f"fal-client submit failed after {attempt+1} attempts: {type(e).__name__}: {e}"; raise
                    await asyncio.sleep(2**attempt)
            
            polling_start_time = time.monotonic(); final_fal_client_result = None; self._last_fal_client_poll_time = 0 # type: ignore
            while True:
                current_poll_time = time.monotonic()
                if current_poll_time - polling_start_time > self.polling_timeout: raise FalApiError(f"fal-client Polling timed out for {fal_request_id} after {self.polling_timeout}s.", status_code=408)
                await asyncio.sleep(max(0.1, DEFAULT_POLLING_INTERVAL_SEC - (current_poll_time - self._last_fal_client_poll_time))) # type: ignore
                self._last_fal_client_poll_time = time.monotonic() # type: ignore
                try:
                    status_obj = await fal_client.status_async(target_edit_app_id, request_id=fal_request_id, with_logs=True)
                    
                    if isinstance(status_obj, fal_client.Completed):
                        logger.info(f"{log_prefix}: Request {fal_request_id} status (fal-client): COMPLETED")
                        final_fal_client_result = await fal_client.result_async(target_edit_app_id, request_id=fal_request_id)
                        break
                    elif isinstance(status_obj, fal_client.InProgress):
                        logs = getattr(status_obj, 'logs', None)
                        logs = logs if logs is not None else [] 
                        logger.info(f"{log_prefix}: Request {fal_request_id} status (fal-client): InProgress. Logs ({len(logs)}): {str(logs)[:100]}...")
                    elif isinstance(status_obj, fal_client.Queued):
                        q_pos = getattr(status_obj, 'queue_position', 'N/A')
                        logger.info(f"{log_prefix}: Request {fal_request_id} status (fal-client): Queued (Queue Pos: {q_pos})")
                    # Check for fal_client.FalError to catch specific Fal operational errors (like 422 for bad input)
                    elif isinstance(status_obj, fal_client.FalError): 
                        # This implies the job itself failed on Fal's side and fal-client is reporting that known error.
                        error_message = getattr(status_obj, 'message', str(status_obj))
                        error_details = getattr(status_obj, 'details', None)
                        logger.error(f"Job {fal_request_id} (fal-client) reported FalError: {error_message}. Details: {error_details}")
                        raise FalApiError(f"Job {fal_request_id} (fal-client) resulted in FalError: {error_message}", error_body=str(error_details or status_obj))
                    else: 
                        logger.error(f"Job {fal_request_id} (fal-client) returned unexpected status object: {type(status_obj).__name__} - {status_obj}")
                        raise FalApiError(f"Job {fal_request_id} (fal-client) returned unhandled status: {type(status_obj).__name__}", error_body=str(status_obj))
                
                except fal_client.FalError as e_fs: # This catches issues like the 422 file download error if surfaced this way
                    error_message = getattr(e_fs, 'message', str(e_fs))
                    error_details = getattr(e_fs, 'details', None)
                    logger.error(f"{log_prefix}: FalError during status/result for {fal_request_id}: {error_message}. Details: {error_details}. Assuming job failed.")
                    raise FalApiError(f"Job {fal_request_id} (fal-client) resulted in FalError: {error_message}", error_body=str(error_details or e_fs))
                except Exception as e_poll_fc: # Catch other unexpected errors during polling
                    logger.warning(f"{log_prefix}: Non-FalError during fal-client status/result for {fal_request_id}: {type(e_poll_fc).__name__}: {e_poll_fc}. Retrying.")

            result_dict["raw_final_response_snippet"] = str(final_fal_client_result)[:1000]
            billing_duration_sec = time.monotonic() - overall_start_time 
            extracted_cost = self._calculate_cost_from_duration(billing_duration_sec)
            logger.warning(f"{log_prefix}: For fal-client edit job, cost estimated using total wall-clock time ({billing_duration_sec:.3f}s): ${extracted_cost:.6f}. Direct metrics depend on fal-client result structure.")
            result_dict.update({"cost": float(extracted_cost), "billing_duration_sec": billing_duration_sec})

            images_list = final_fal_client_result.get("images") if isinstance(final_fal_client_result, dict) else None
            if not images_list or not isinstance(images_list, list) or not images_list[0]: raise FalApiError("No 'images' array in fal-client final response.", error_body=str(final_fal_client_result)[:200])
            
            image_info = images_list[0]; image_b64_data, image_url_data = image_info.get("b64_json"), image_info.get("url"); img_bytes = None
            if image_b64_data:
                try: img_bytes = base64.b64decode(image_b64_data)
                except Exception as e: logger.warning(f"{log_prefix}: B64 decode error (fal-client): {e}. Try URL."); img_bytes=None
            if not img_bytes and image_url_data:
                logger.info(f"{log_prefix}: Downloading from URL (fal-client result): {image_url_data}")
                img_bytes = await self._make_rest_request("GET", image_url_data, DEFAULT_RESULT_TIMEOUT_SEC, is_json_response=False, custom_headers={})
            if not img_bytes: raise FalApiError("Failed to get image bytes from fal-client result.")
            with open(output_image_path, "wb") as f: f.write(img_bytes); logger.info(f"{log_prefix}: Image (via fal-client) saved to {output_image_path}")
            result_dict.update({"status": "success", "image_path": str(output_image_path), "error_message": None})
            self.total_successful_requests += 1; self.total_accumulated_cost += extracted_cost; self.total_billing_duration_sec_sum_successful += billing_duration_sec
        except Exception as e:
            logger.error(f"{log_prefix}: Fal processing error (fal-client path): {type(e).__name__}: {e}", exc_info=True)
            result_dict["error_message"] = f"{type(e).__name__}: {e}"
        finally:
            wc_dur = time.monotonic() - overall_start_time; result_dict["api_call_duration_sec"] = wc_dur
            if result_dict.get("status") == "success": self.total_api_call_duration_sum_successful += wc_dur
            if result_dict.get("status")=="failed" and self.total_requests > (self.total_successful_requests + self.total_failed_requests): self.total_failed_requests+=1
            if semaphore: semaphore.release()
        return result_dict

    def get_stats(self) -> Dict[str, Any]:
        return {"total_requests_attempted": self.total_requests, "total_requests_successful": self.total_successful_requests, "total_requests_failed": self.total_failed_requests, "total_accumulated_cost_successful": self.total_accumulated_cost, "total_api_call_duration_sum_successful_sec": self.total_api_call_duration_sum_successful, "total_billing_duration_sec_sum_successful_sec": self.total_billing_duration_sec_sum_successful}

# --- Task Runner Helpers (No change from v4.4.4) ---
async def run_fal_image_gen_task_rest(handler: PromptToImageFalRest, prompt_str: str, out_path: Path, task_id: int, sem: asyncio.Semaphore, size_str: str, quality_str: str, background_str: str, num_images_int: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    logger.info(f"--- Fal REST Task {task_id} (Text2Img): Waiting for semaphore for prompt '{prompt_str}' ---")
    task_inputs = {"prompt": prompt_str, "output_path": out_path}
    result = await handler.generate_image(prompt=prompt_str, output_image_path=out_path, size=size_str, quality=quality_str, background=background_str, num_images=num_images_int, semaphore=sem)
    error_msg_val = result.get('error_message'); error_msg_display = 'N/A' if error_msg_val is None else (str(error_msg_val)[:100] + '...' if len(str(error_msg_val)) > 100 else str(error_msg_val))
    logger.info(f"--- Fal REST Task {task_id} (Text2Img): Finished with Status: {result.get('status')}. Error: {error_msg_display} ---")
    return result, task_inputs

async def run_fal_image_edit_task_rest(handler: PromptToImageFalRest, image_inputs_list: List[Union[str,Path]], prompt_str: str, out_path: Path, task_id: int, sem: asyncio.Semaphore, size_str: str, quality_str: str, num_images_int: int, mask_input: Optional[Union[str,Path]] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    logger.info(f"--- Fal CLIENT Task {task_id} (ImgEdit): Waiting for semaphore for prompt '{prompt_str}' ---")
    task_inputs = {"input_images": image_inputs_list, "mask_image": mask_input, "prompt": prompt_str, "output_path": out_path}
    result = await handler.edit_image(image_paths_or_urls=image_inputs_list, prompt=prompt_str, output_image_path=out_path, mask_path_or_url=mask_input, size=size_str, quality=quality_str, num_images=num_images_int, semaphore=sem)
    error_msg_val = result.get('error_message'); error_msg_display = 'N/A' if error_msg_val is None else (str(error_msg_val)[:100] + '...' if len(str(error_msg_val)) > 100 else str(error_msg_val))
    logger.info(f"--- Fal CLIENT Task {task_id} (ImgEdit): Finished with Status: {result.get('status')}. Error: {error_msg_display} ---")
    return result, task_inputs
    
# --- Dummy image/mask creators (No change from v4.4.4) ---
def _create_dummy_pil_image(filepath: Path, size: tuple[int,int] = (200,200), color: Union[str, tuple[int,int,int]] = "grey", text: Optional[str] = None) -> bool:
    if not Image or not ImageDraw: return False
    try:
        img = Image.new('RGB', size, color=color); d = ImageDraw.Draw(img)
        if text:
            try: text_bbox = d.textbbox((0,0), text); text_width = text_bbox[2] - text_bbox[0]; text_height = text_bbox[3] - text_bbox[1]; x = (size[0] - text_width) / 2; y = (size[1] - text_height) / 2; d.text((x,y), text, fill=(0,0,0) if color=="white" or color==(255,255,255) else (255,255,255))
            except Exception: d.text((10,10), text, fill=(0,0,0) if color=="white" else (255,255,255))
        filepath.parent.mkdir(parents=True, exist_ok=True); img.save(filepath)
        logger.info(f"Created/verified dummy test image: {filepath}")
        return True
    except Exception as e: logger.error(f"Could not create dummy image {filepath}: {e}"); return False

def _create_dummy_pil_mask(filepath: Path, image_size: tuple[int,int] = (200,200), transparent_shape: str = "circle") -> bool:
    if not Image or not ImageDraw: return False
    try:
        mask = Image.new('RGBA', image_size, (0,0,0,255)); draw = ImageDraw.Draw(mask) 
        shape_margin = int(min(image_size) * 0.1)
        if transparent_shape == "circle":
            center_x, center_y = image_size[0]//2, image_size[1]//2; radius = min(image_size)//2 - shape_margin
            draw.ellipse([(center_x-radius, center_y-radius), (center_x+radius, center_y+radius)], fill=(0,0,0,0))
        elif transparent_shape == "custom_circle_offset":
            eye_center_x, eye_center_y = int(image_size[0] * 0.35), int(image_size[1] * 0.30)
            eye_radius = int(min(image_size) * 0.05) 
            draw.ellipse([(eye_center_x-eye_radius, eye_center_y-eye_radius), (eye_center_x+eye_radius, eye_center_y+eye_radius)], fill=(0,0,0,0))
        else: draw.rectangle((shape_margin, shape_margin, image_size[0]-shape_margin, image_size[1]-shape_margin), fill=(0,0,0,0))
        filepath.parent.mkdir(parents=True, exist_ok=True); mask.save(filepath, "PNG")
        logger.info(f"Created/verified dummy test mask ({transparent_shape}): {filepath}")
        return True
    except Exception as e: logger.error(f"Could not create dummy mask {filepath}: {e}"); return False

# --- Test Main Functions (Versioned test names) ---
async def main_test_fal_text2img_final(): # Renamed for final version
    print(f"--- Running PromptToImageFalRest Test ({TEST_VERSION_SUFFIX} - Text-to-Image via REST) ---")
    if not FAL_API_KEY or ':' not in FAL_API_KEY: print("ERROR: FAL_KEY TNS invalid.", file=sys.stderr); return
    current_app_id_setting = os.getenv("FALAI_IMAGE_MODEL", FALAI_IMAGE_MODEL_ID_DEFAULT)
    print(f"Target Fal Application ID (from env/default): {current_app_id_setting}. Price/Sec (fallback): ${FAL_GPT_IMAGE_1_PRICE_PER_SECOND:.8f}")
    image_handler_rest = None
    try:
        image_handler_rest = PromptToImageFalRest(model_id=current_app_id_setting, submit_retries=0, polling_timeout=max(120,DEFAULT_POLLING_TIMEOUT_SEC))
        test_prompts_data = [
            {"prompt": f"{TEST_VERSION_SUFFIX} REST Test: A stunningly detailed photorealistic portrait of a wise old elephant, desert background at sunset.", "filename": f"test_text2img_{TEST_VERSION_SUFFIX}_1_for_edit.png"},
            {"prompt": f"{TEST_VERSION_SUFFIX} REST Test: Whimsical illustration of a cat riding a bicycle made of cheese on the moon.", "filename": f"test_text2img_{TEST_VERSION_SUFFIX}_2.png"}
        ]
        print(f"Output images: {TEXT2IMG_TEST_OUTPUT_DIR.resolve()}"); TEXT2IMG_TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        semaphore=asyncio.Semaphore(2); tasks_with_inputs=[]
        print(f"\nCreating {len(test_prompts_data)} Fal REST text-to-image task(s)...")
        for i, data in enumerate(test_prompts_data):
            output_path_obj = TEXT2IMG_TEST_OUTPUT_DIR / data["filename"]
            TASK_NAME = f"FalRestText2ImgTask_{TEST_VERSION_SUFFIX}_{i+1}"
            task = asyncio.create_task(run_fal_image_gen_task_rest( image_handler_rest, data["prompt"], output_path_obj, task_id=i+1, sem=semaphore, size_str=DEFAULT_IMAGE_SIZE_FAL_TEST, quality_str=DEFAULT_IMAGE_QUALITY_FAL_TEST, background_str=DEFAULT_IMAGE_BACKGROUND_FAL_TEST, num_images_int=DEFAULT_NUM_IMAGES_FAL), name=TASK_NAME)
            tasks_with_inputs.append(task)
        await _run_and_summarize_tasks(f"Text-to-Image ({TEST_VERSION_SUFFIX} REST)", tasks_with_inputs, image_handler_rest)
    except Exception as e: logger.error(f"Error in text-to-image test: {e}", exc_info=True); print(f"CRITICAL ERROR text-to-image test: {e}")
    finally:
        if image_handler_rest: await image_handler_rest.close_sessions()

async def main_test_fal_edit_single_image_final():
    print(f"\n--- Running PromptToImageFalRest Test ({TEST_VERSION_SUFFIX} - Single Image Edit via fal-client) ---")
    if not FAL_API_KEY or ':' not in FAL_API_KEY: print("ERROR: FAL_KEY invalid.", file=sys.stderr); return
    if not EDIT_BASE_IMAGE_PATH.exists(): print(f"Error: Base image ({EDIT_BASE_IMAGE_PATH}) not found. Run text-to-image test first.", file=sys.stderr); return

    image_handler_edit = None
    try:
        image_handler_edit = PromptToImageFalRest(model_id="fal-ai/gpt-image-1", submit_retries=0, polling_timeout=max(180,DEFAULT_POLLING_TIMEOUT_SEC))
        
        test_output_dir = BASE_TEMP_DIR / f"image_test_fal_rest_{TEST_VERSION_SUFFIX}_imgedit"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        local_dummy_for_upload_path = INTERNAL_TEST_IMAGE_DIR / f"local_dummy_for_edit_{TEST_VERSION_SUFFIX}.png"
        _create_dummy_pil_image(local_dummy_for_upload_path, text=f"Fal Upload Test {TEST_VERSION_SUFFIX}")
        
        uploaded_fal_url_for_test = ""
        try:
            logger.info(f"Attempting to upload {local_dummy_for_upload_path} for edit test...")
            # Use a specific log prefix for this utility upload if desired, or rely on fal-client's own logging
            uploaded_fal_url_for_test = await fal_client.upload_file_async(str(local_dummy_for_upload_path))
            logger.info(f"Successfully uploaded {local_dummy_for_upload_path} to {uploaded_fal_url_for_test}")
            test_cases_data_list = [
                {"input_images": [str(EDIT_BASE_IMAGE_PATH)], "prompt": f"{TEST_VERSION_SUFFIX} Give the elephant in the image large, vibrant butterfly wings and make the sky a swirling galaxy.", "filename": f"test_edit_elephant_wings_{TEST_VERSION_SUFFIX}.png"},
                {"input_images": [uploaded_fal_url_for_test], "prompt": f"{TEST_VERSION_SUFFIX} Render this image as a glowing neon sign on a brick wall.", "filename": f"test_edit_fal_hosted_neon_{TEST_VERSION_SUFFIX}.png"}
            ]
        except Exception as e_upload:
            logger.error(f"Failed to upload {local_dummy_for_upload_path} for edit test: {e_upload}. Skipping second edit test case.", exc_info=True)
            test_cases_data_list = [
                {"input_images": [str(EDIT_BASE_IMAGE_PATH)], "prompt": f"{TEST_VERSION_SUFFIX} Give the elephant in the image large, vibrant butterfly wings and make the sky a swirling galaxy.", "filename": f"test_edit_elephant_wings_{TEST_VERSION_SUFFIX}.png"},
            ]

        print(f"Output images: {test_output_dir.resolve()}");
        semaphore = asyncio.Semaphore(1); tasks_with_inputs = []
        for i, data in enumerate(test_cases_data_list):
            output_path_obj = test_output_dir / data["filename"]
            TASK_NAME = f"FalClientImgEditTask_{TEST_VERSION_SUFFIX}_{i+1}"
            task = asyncio.create_task(run_fal_image_edit_task_rest(image_handler_edit, data["input_images"], data["prompt"], output_path_obj, task_id=i+1, sem=semaphore, size_str=DEFAULT_IMAGE_SIZE_FAL_TEST, quality_str=DEFAULT_IMAGE_QUALITY_FAL_TEST, num_images_int=DEFAULT_NUM_IMAGES_FAL), name=TASK_NAME)
            tasks_with_inputs.append(task)
        await _run_and_summarize_tasks(f"Image Edit (General - {TEST_VERSION_SUFFIX} fal-client)", tasks_with_inputs, image_handler_edit)
    except Exception as e: logger.error(f"Error in single image edit test: {e}", exc_info=True); print(f"CRITICAL ERROR single edit test: {e}")
    finally:
        if image_handler_edit: await image_handler_edit.close_sessions()

async def main_test_fal_edit_speech_bubble_final():
    print(f"\n--- Running PromptToImageFalRest Test ({TEST_VERSION_SUFFIX} - Speech Bubble via fal-client) ---")
    if not FAL_API_KEY or ':' not in FAL_API_KEY: print("ERROR: FAL_KEY invalid.", file=sys.stderr); return
    if not EDIT_BASE_IMAGE_PATH.exists(): print(f"Error: Base image ({EDIT_BASE_IMAGE_PATH}) not found. Run text-to-image test first.", file=sys.stderr); return
    
    image_handler_speech = None
    try:
        image_handler_speech = PromptToImageFalRest(model_id="fal-ai/gpt-image-1", submit_retries=0, polling_timeout=max(180,DEFAULT_POLLING_TIMEOUT_SEC))
        test_case_data = {"input_images": [str(EDIT_BASE_IMAGE_PATH)], "prompt": f"{TEST_VERSION_SUFFIX} Add a large, clear, classic comic book speech bubble near the elephant's head, containing the text 'I remember everything!' in a bold, readable font.", "filename": f"test_edit_elephant_speech_{TEST_VERSION_SUFFIX}.png"}
        test_output_dir = BASE_TEMP_DIR / f"image_test_fal_rest_{TEST_VERSION_SUFFIX}_speech"
        print(f"Output image: {test_output_dir.resolve()}"); test_output_dir.mkdir(parents=True, exist_ok=True)
        output_path_obj = test_output_dir / test_case_data["filename"]
        semaphore = asyncio.Semaphore(1); tasks_with_inputs = []
        TASK_NAME=f"FalClientSpeechTask_{TEST_VERSION_SUFFIX}"
        task = asyncio.create_task(run_fal_image_edit_task_rest(image_handler_speech, test_case_data["input_images"], test_case_data["prompt"], output_path_obj, task_id=1, sem=semaphore, size_str=DEFAULT_IMAGE_SIZE_FAL_TEST, quality_str=DEFAULT_IMAGE_QUALITY_FAL_TEST, num_images_int=DEFAULT_NUM_IMAGES_FAL), name=TASK_NAME)
        tasks_with_inputs.append(task)
        await _run_and_summarize_tasks(f"Image Edit (Speech Bubble - {TEST_VERSION_SUFFIX} fal-client)", tasks_with_inputs, image_handler_speech)
    except Exception as e: logger.error(f"Error in speech bubble test: {e}", exc_info=True); print(f"CRITICAL ERROR speech bubble test: {e}")
    finally:
        if image_handler_speech: await image_handler_speech.close_sessions()

async def main_test_fal_edit_masked_inpainting_final():
    print(f"\n--- Running PromptToImageFalRest Test ({TEST_VERSION_SUFFIX} - Masked Inpainting via fal-client) ---")
    if not FAL_API_KEY or ':' not in FAL_API_KEY: print("ERROR: FAL_KEY invalid.", file=sys.stderr); return
    if not EDIT_BASE_IMAGE_PATH.exists(): print(f"Error: Base image ({EDIT_BASE_IMAGE_PATH}) not found. Run text-to-image test.", file=sys.stderr); return
    if not Image or not ImageDraw: print("Pillow not available. Mask test cannot proceed.", file=sys.stderr); return

    mask_image_name = f"test_mask_elephant_eye_{TEST_VERSION_SUFFIX}.png"
    mask_image_path = INTERNAL_TEST_IMAGE_DIR / mask_image_name
    if not _create_dummy_pil_mask(mask_image_path, image_size=(1024,1024), transparent_shape="custom_circle_offset"):
         print(f"Failed to create mask for inpainting: {mask_image_path}", file=sys.stderr); return

    image_handler_mask = None
    try:
        image_handler_mask = PromptToImageFalRest(model_id="fal-ai/gpt-image-1", submit_retries=0, polling_timeout=max(240, DEFAULT_POLLING_TIMEOUT_SEC))
        test_case_data = {"input_images": [str(EDIT_BASE_IMAGE_PATH)], "mask_image": mask_image_path, "prompt": f"{TEST_VERSION_SUFFIX} Change the elephant's eye in the transparent masked area to be a glowing, fiery red ember. The rest of the image should remain exactly the same.", "filename": f"test_inpainted_elephant_eye_{TEST_VERSION_SUFFIX}.png"}
        test_output_dir = BASE_TEMP_DIR / f"image_test_fal_rest_{TEST_VERSION_SUFFIX}_inpainting"
        print(f"Output image: {test_output_dir.resolve()}"); test_output_dir.mkdir(parents=True, exist_ok=True)
        output_path_obj = test_output_dir / test_case_data["filename"]
        semaphore = asyncio.Semaphore(1); tasks_with_inputs = []
        TASK_NAME = f"FalClientMaskInpaintTask_{TEST_VERSION_SUFFIX}"
        task = asyncio.create_task(run_fal_image_edit_task_rest(image_handler_mask, test_case_data["input_images"], test_case_data["prompt"], output_path_obj, task_id=1, sem=semaphore, mask_input=test_case_data["mask_image"], size_str="1024x1024", quality_str=DEFAULT_IMAGE_QUALITY_FAL_TEST, num_images_int=DEFAULT_NUM_IMAGES_FAL), name=TASK_NAME)
        tasks_with_inputs.append(task)
        await _run_and_summarize_tasks(f"Image Edit (Masked Inpainting - {TEST_VERSION_SUFFIX} fal-client)", tasks_with_inputs, image_handler_mask)
        logger.warning("Reminder: Masked inpainting effectiveness for '%s' depends on underlying model logic and `mask_url` support.", image_handler_mask.edit_app_id_for_client)
    except Exception as e: logger.error(f"Error in masked inpainting test: {e}", exc_info=True); print(f"CRITICAL ERROR masked inpaint test: {e}")
    finally:
        if image_handler_mask: await image_handler_mask.close_sessions()


async def _run_and_summarize_tasks(test_suite_name: str, tasks_with_inputs: list, handler_instance: Optional[PromptToImageFalRest]):
    # ... (This summary function is robust and doesn't need changes) ...
    if not tasks_with_inputs: logger.warning(f"No tasks provided for {test_suite_name}."); return
    batch_start_time = time.monotonic()
    results_or_exceptions = await asyncio.gather(*tasks_with_inputs, return_exceptions=True)
    batch_total_duration = time.monotonic() - batch_start_time
    print(f"--- {test_suite_name} Batch processing finished in {batch_total_duration:.3f}s ---")
    print(f"\n--- {test_suite_name} Individual Task Results ---")
    successful_task_count = 0; batch_total_cost_this_run = Decimal("0.0")
    for i, gather_item in enumerate(results_or_exceptions):
        task_obj = tasks_with_inputs[i]; task_name = task_obj.get_name() if hasattr(task_obj, 'get_name') else f"Task {i+1}"
        print(f"\n--- Result for {task_name} ---")
        if isinstance(gather_item, Exception): print(f"  Status: FAILED_WITH_EXCEPTION\n  Error Type: {type(gather_item).__name__}\n  Error Details: {str(gather_item)}")
        elif isinstance(gather_item, tuple) and len(gather_item) == 2:
            res_dict, inputs_dict = gather_item
            print(f"  Input Prompt: \"{inputs_dict.get('prompt', 'N/A')}\"")
            if inputs_dict.get('input_images'): print(f"  Input Image(s): {[str(p) for p in inputs_dict['input_images']]}")
            if inputs_dict.get('mask_image'): print(f"  Input Mask: {str(inputs_dict['mask_image'])}")
            status = res_dict.get('status', 'Status N/A'); api_time = res_dict.get('api_call_duration_sec', 0.0)
            print(f"  Status: {status.upper()}\n  Wall-Clock API Call Time: {api_time:.3f}s")
            cost_val = Decimal(str(res_dict.get('cost', '0.0'))); billing_time_val = res_dict.get('billing_duration_sec', 0.0)
            print(f"  Reported Cost: ${cost_val:.6f}\n  Reported Billing Duration: {billing_time_val:.3f}s")
            if status == "success": 
                successful_task_count += 1; batch_total_cost_this_run += cost_val
                print(f"  Output Image Path: {res_dict.get('image_path', 'N/A')}")
                if res_dict.get('revised_prompt'): print(f"  Revised Prompt: {res_dict.get('revised_prompt')}")
            else: print(f"  Error Message: {res_dict.get('error_message', 'No specific error message.')}")
            print(f"  Fal Request ID: {res_dict.get('fal_request_id', 'N/A')}")
        else: print(f"  Status: UNEXPECTED_RESULT_TYPE ({type(gather_item)})")
    print(f"\n--- {test_suite_name} Run Summary ---")
    print(f"Total Wall-Clock Time for Batch: {batch_total_duration:.3f}s")
    print(f"Tasks Attempted: {len(tasks_with_inputs)}, Successful: {successful_task_count}")
    print(f"Total Cost for Successful Tasks in this Batch: ${batch_total_cost_this_run:.6f}")
    if handler_instance: 
        stats = handler_instance.get_stats()
        print("\n--- Handler Cumulative Statistics ---")
        for key, value in stats.items():
            if isinstance(value, Decimal): print(f"  {key.replace('_', ' ', 1).replace('_', ' ').title()}: ${value:.6f}") # Improved title casing
            elif isinstance(value, float): print(f"  {key.replace('_', ' ', 1).replace('_', ' ').title()}: {value:.3f}s")
            else: print(f"  {key.replace('_', ' ', 1).replace('_', ' ').title()}: {value}")

async def main_all_tests():
    """Runs all test suites sequentially in the same event loop."""
    print(f"--- Running ALL Fal Image Tests ({TEST_VERSION_SUFFIX}) ---")
    await main_test_fal_text2img_final()
    # Ensure EDIT_BASE_IMAGE_PATH exists before running edit tests
    if not EDIT_BASE_IMAGE_PATH.exists():
        logger.error(f"CRITICAL: Base image for edit tests ({EDIT_BASE_IMAGE_PATH}) was not created. Skipping edit tests.")
    else:
        await main_test_fal_edit_single_image_final()
        await main_test_fal_edit_speech_bubble_final()
        await main_test_fal_edit_masked_inpainting_final()
    print(f"\n--- ALL Fal Image Tests ({TEST_VERSION_SUFFIX}) Completed ---")

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3,8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    actual_model_id_for_run = os.getenv("FALAI_IMAGE_MODEL", FALAI_IMAGE_MODEL_ID_DEFAULT)
    print(f"SCRIPT MAIN: FALAI_IMAGE_MODEL_ID (from .env or script default) for text2img test handlers: '{actual_model_id_for_run}'")
    logger.info("Edit tests will explicitly use 'fal-ai/gpt-image-1' as the base for PromptToImageFalRest "
                "to construct the '.../edit-image' app ID for fal-client.")

    try:
        asyncio.run(main_all_tests())
    except KeyboardInterrupt: print("\nTest run interrupted by user (KeyboardInterrupt).")
    except Exception as e_asyncio_run:
        print(f"Critical error during asyncio.run execution: {type(e_asyncio_run).__name__}: {e_asyncio_run}")
        logger.error("Critical error during asyncio.run", exc_info=True)