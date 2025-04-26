# File: openrouter_engine_.py
"""
OpenRouter API Engine (openrouter_engine_.py)

Provides a reusable class for interacting with the OpenRouter API (https://openrouter.ai/),
primarily focusing on the chat completions endpoint. Designed for easy integration
into various projects.

Key Features:
- Fetches available models on initialization.
- Handles chat completion requests (streaming and non-streaming).
- Uses `httpx` for asynchronous HTTP requests.
- Implements basic retry logic with exponential backoff.
- Tracks estimated cost and request timing per call and cumulatively.
- Retrieves accurate cost and token counts via the /generation endpoint after completion,
  with retries for potential timing issues (404 errors).
- Designed with async-first approach but can be used synchronously if needed (via asyncio.run).
- Reads configuration (API key) from .env file.

API Info (OpenRouter):
- Website: https://openrouter.ai/docs/api-reference
- Authentication: Bearer token via `Authorization` header. **Requires OPENROUTER_KEY environment variable.**
- Base URL: https://openrouter.ai/api/v1
- Key Endpoints:
    - `/models` (GET): Lists available models. Fetched on initialization.
    - `/chat/completions` (POST): Main endpoint for generating text. Supports streaming (SSE).
    - `/generation?id={id}` (GET): Retrieves detailed stats (cost, tokens) for a completed generation ID. Crucial for accurate tracking.
- Rate Limits: Vary by model and user tier. Handled partially by retries.
- Cost: Varies significantly by model. Tracked per request.
- Identifiers: Recommends `HTTP-Referer` and `X-Title` headers for app identification.

Usage:
1. Ensure `OPENROUTER_KEY` is set in your `.env` file.
2. Instantiate `OpenRouterEngine`.
3. Use the `chat_completion` method for generating text.
4. Use `get_available_models`, `get_total_cost`, `get_total_request_time` for info.

Example (Async):
   import asyncio
   from openrouter_engine_ import OpenRouterEngine

   async def main():
       engine = OpenRouterEngine(app_url="http://localhost", app_title="MyTestApp")
       try:
           await engine.wait_for_initialization()
           if not engine.is_available():
               print("Engine initialization failed.")
               return

           print("Available models:", engine.get_available_models())

           messages = [{"role": "user", "content": "Explain quantum entanglement simply."}]
           model_id = "openai/gpt-3.5-turbo" # Or choose another from available models

           # Non-streaming
           result, stats = await engine.chat_completion(model_id, messages)
           if result:
               print("Non-Streaming Result:", result)
               print("Stats:", stats)

           # Streaming
           print("\nStreaming Result:")
           stream_generator, stats_async_func = await engine.chat_completion(model_id, messages, stream=True)
           if stream_generator:
               async for chunk in stream_generator:
                   print(chunk, end="", flush=True)
               print("\nStream finished.")
               stream_stats = await stats_async_func() # Get stats after stream consumed
               print("Stream Stats:", stream_stats)

           print(f"\nTotal Cost: ${engine.get_total_cost():.6f}")
           print(f"Total Request Time: {engine.get_total_request_time():.3f}s")
       finally:
            await engine.close() # Ensure client is closed

   if __name__ == "__main__":
       asyncio.run(main())

Notes:
- Cost and token count retrieval involves a second API call (`/generation`) after the main completion,
  which adds a small latency but provides accuracy. Includes retries for 404 errors.
- Streaming responses yield content chunks; cost/stats are logged internally *after* the
  stream completes or retrieved via the returned async function.
"""


import os
import sys
import json
import logging
import time
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Callable, Awaitable
from dotenv import load_dotenv
import httpx # Ensure httpx is installed (`pip install httpx`)

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_RETRY_COUNT = 1
DEFAULT_TIMEOUT = 120 # seconds
STATS_FETCH_RETRIES = 3 # Number of retries for fetching stats if 404 received
STATS_FETCH_DELAY = 1.5 # Delay in seconds between stats fetch retries

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity
logger = logging.getLogger(__name__)

class OpenRouterError(Exception):
    """Custom exception for OpenRouter API errors."""
    pass

class OpenRouterEngine:
    """
    Asynchronous client for interacting with the OpenRouter API.
    Handles model listing, chat completions (sync & async), cost/time tracking, and retries.
    """
    def __init__(self, api_key: Optional[str] = None, app_url: Optional[str] = None, app_title: Optional[str] = None, api_retries: int = DEFAULT_API_RETRY_COUNT):
        """
        Initializes the OpenRouterEngine.

        Args:
            api_key (Optional[str]): OpenRouter API key. Defaults to OPENROUTER_KEY env var.
            app_url (Optional[str]): Your site URL for HTTP-Referer header.
            app_title (Optional[str]): Your site name for X-Title header.
            api_retries (int): Number of retries for failed API calls.
        """
        logger.info("Initializing OpenRouterEngine...")
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            logger.error("OpenRouter API key not found in environment variables (OPENROUTER_KEY) or arguments.")
            raise ValueError("OpenRouter API key is required.")

        self.base_url = OPENROUTER_BASE_URL
        self.api_retries = api_retries
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if app_url:
            self.headers["HTTP-Referer"] = app_url
        if app_title:
            self.headers["X-Title"] = app_title

        logger.info(f"Headers set (excluding Authorization): { {k:v for k,v in self.headers.items() if k != 'Authorization'} }")

        self.available_models: Dict[str, Dict[str, Any]] = {}
        self.initialized_event = asyncio.Event() # Use Event for async init status
        self.init_error: Optional[Exception] = None

        # Tracking metrics (Internal - reflects cumulative successful operations)
        self.total_cost = Decimal("0.0")
        self.total_request_time = 0.0 # Sum of individual successful request durations
        self.request_count = 0 # Total attempts

        self.client = httpx.AsyncClient(headers=self.headers, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
        self._init_task = asyncio.create_task(self._initialize_async())

    async def _initialize_async(self):
        """Asynchronously fetches available models on initialization."""
        logger.info("Fetching available models from OpenRouter...")
        try:
            self.available_models = await self._fetch_available_models_async()
            logger.info(f"Successfully fetched {len(self.available_models)} models.")
            self.initialized_event.set()
        except Exception as e:
            self.init_error = e
            logger.error(f"Failed to initialize OpenRouterEngine: Could not fetch models. Error: {e}", exc_info=False)
            self.initialized_event.set()

    async def wait_for_initialization(self):
        """Waits until the asynchronous initialization is complete."""
        await self.initialized_event.wait()
        if self.init_error:
            raise OpenRouterError(f"Initialization failed: {self.init_error}") from self.init_error

    def is_available(self) -> bool:
        """Checks if the engine initialized successfully (models fetched)."""
        return self.initialized_event.is_set() and self.init_error is None

    async def close(self):
        """Closes the underlying httpx client."""
        logger.info("Closing OpenRouterEngine httpx client...")
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                logger.info("Initialization task cancelled.")
            except Exception as e:
                 logger.warning(f"Error during init task cancellation: {e}")
        await self.client.aclose()
        logger.info("OpenRouterEngine client closed.")

    async def _fetch_available_models_async(self) -> Dict[str, Dict[str, Any]]:
        """Internal async method to fetch and process model list."""
        models_url = f"{self.base_url}/models"
        try:
            response = await self.client.get(models_url)
            response.raise_for_status()
            data = response.json()
            if "data" not in data or not isinstance(data["data"], list):
                 raise OpenRouterError("Invalid format in /models response: 'data' key missing or not a list.")
            return {model['id']: model for model in data['data'] if 'id' in model}
        except httpx.RequestError as e:
            logger.error(f"Network error fetching models: {e}")
            raise OpenRouterError(f"Network error fetching models: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching models: {e.response.status_code} - {e.response.text}")
            raise OpenRouterError(f"HTTP error fetching models: {e.response.status_code}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from /models: {e}")
            raise OpenRouterError("Failed to decode /models response.") from e
        except Exception as e:
             logger.error(f"An unexpected error occurred fetching models: {e}", exc_info=True)
             raise OpenRouterError(f"Unexpected error fetching models: {e}") from e

    def get_available_models(self) -> List[str]:
        """Returns a list of available model IDs."""
        if not self.is_available():
            logger.warning("Attempting to get models before initialization completed or after failure.")
            return []
        # import code; code.interact(local={**globals(), **locals()})
        return list(self.available_models.keys())

    async def _get_generation_stats(self, generation_id: str) -> Dict[str, Any]:
        """
        Fetches detailed cost and token stats for a given generation ID.
        Includes retries with delays specifically for initial 404 errors.
        """
        stats_url = f"{self.base_url}/generation?id={generation_id}"
        last_error = None
        for attempt in range(STATS_FETCH_RETRIES + 1):
            try:
                if attempt > 0:
                    logger.debug(f"Retrying stats fetch for {generation_id} (Attempt {attempt + 1}/{STATS_FETCH_RETRIES + 1})...")
                else:
                    logger.debug(f"Fetching generation stats for ID: {generation_id}")

                response = await self.client.get(stats_url)
                response.raise_for_status()
                data = response.json()
                if "data" not in data or not isinstance(data["data"], dict):
                    raise OpenRouterError(f"Invalid format in /generation response for ID {generation_id}")

                stats = data["data"]
                # --- !! CORRECTED FIELD NAME HERE !! ---
                # Use "total_cost" instead of "cost"
                # Ensure it's treated as a string before Decimal conversion for robustness if API returns number directly
                cost_value = stats.get("total_cost", "0.0")
                cost = Decimal(str(cost_value)) if cost_value is not None else Decimal("0.0")
                # -----------------------------------------
                prompt_tokens = int(stats.get("native_tokens_prompt") or 0)
                completion_tokens = int(stats.get("native_tokens_completion") or 0)

                logger.debug(f"Successfully fetched stats for {generation_id}. Raw Total Cost: {cost_value}")
                return {
                    "cost": cost, # This is now the Decimal value of total_cost
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "raw_stats": stats
                }
            except httpx.HTTPStatusError as e:
                last_error = e
                # Specifically retry on 404, assuming temporary unavailability
                if e.response.status_code == 404 and attempt < STATS_FETCH_RETRIES:
                    logger.warning(f"Stats fetch for {generation_id} returned 404 (Attempt {attempt + 1}). Waiting {STATS_FETCH_DELAY}s before retrying...")
                    await asyncio.sleep(STATS_FETCH_DELAY)
                    continue # Go to next retry iteration
                else:
                    # For other HTTP errors or final 404 retry, log and return error state
                    logger.error(f"HTTP error fetching stats for {generation_id}: {e.response.status_code} - {e.response.text}")
                    return {"cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": f"HTTP {e.response.status_code}"}
            except httpx.RequestError as e:
                last_error = e
                logger.error(f"Network error fetching stats for {generation_id}: {e}")
                # Don't retry network errors for stats, return error state
                return {"cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": str(e)}
            except json.JSONDecodeError as e:
                 last_error = e
                 logger.error(f"Failed to decode JSON response from /generation for ID {generation_id}: {e}")
                 return {"cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": "JSON Decode Error"}
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error fetching stats for {generation_id}: {e}", exc_info=True)
                return {"cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": f"Unexpected: {e}"}

        # If loop finishes without returning (should only happen if all retries fail on 404)
        logger.error(f"Stats fetch for {generation_id} failed after {STATS_FETCH_RETRIES + 1} attempts, last error: {last_error}")
        error_msg = f"Failed after retries"
        if isinstance(last_error, httpx.HTTPStatusError):
            error_msg = f"HTTP {last_error.response.status_code} after retries"
        elif last_error:
             error_msg = f"{type(last_error).__name__} after retries"
        return {"cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": error_msg}


    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        max_retries: Optional[int] = None,
        **kwargs: Any
    ) -> Tuple[Optional[Dict[str, Any] | AsyncGenerator[str, None]], Optional[Dict[str, Any] | Callable[[], Awaitable[Dict[str, Any]]]]]:
        """Performs a chat completion request to OpenRouter."""
        await self.wait_for_initialization()

        if model not in self.available_models:
            logger.error(f"Model '{model}' not found in available models. Please check the ID.")
            return None, None

        endpoint = f"{self.base_url}/chat/completions"
        payload = {"model": model, "messages": messages, "stream": stream, **kwargs}
        retries = max_retries if max_retries is not None else self.api_retries

        generation_id_holder = [None]

        async def _get_stats_for_stream():
            """Awaitable function passed back to the caller for stream stats."""
            if generation_id_holder[0]:
                stats = await self._get_generation_stats(generation_id_holder[0])
                self._log_and_track_stats(stats)
                return stats
            else:
                logger.warning("Could not retrieve stream stats: Generation ID was not captured.")
                return {"cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": "Missing Generation ID"}


        start_time = time.monotonic()
        last_exception = None
        for attempt in range(retries + 1):
            attempt_start_time = time.monotonic()
            try:
                 logger.info(f"Attempt {attempt + 1}/{retries + 1}: Calling OpenRouter chat completion (model: {model}, stream: {stream})...")
                 logger.debug(f"Payload (messages omitted for brevity): {{'model': '{model}', 'stream': {stream}, other_args: {kwargs}}}")

                 if not stream:
                     # === Non-Streaming Case ===
                     response = await self.client.post(endpoint, json=payload)
                     self.request_count += 1
                     response.raise_for_status()
                     request_time = time.monotonic() - attempt_start_time
                     self.total_request_time += request_time
                     response_json = response.json()
                     generation_id = response_json.get("id")
                     if not generation_id:
                         logger.error("Successful response but missing 'id' field. Cannot fetch stats.")
                         return response_json, {"cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": "Missing Generation ID in response"}
                     stats = await self._get_generation_stats(generation_id)
                     self._log_and_track_stats(stats, request_time)
                     logger.info(f"Non-streaming request successful. Time: {request_time:.3f}s")
                     return response_json, stats

                 else:
                     # === Streaming Case ===
                     stats_fetcher = _get_stats_for_stream
                     async def _stream_generator():
                         stream_start_time = time.monotonic()
                         stream_success = False
                         try:
                             async with self.client.stream("POST", endpoint, json=payload) as response:
                                self.request_count += 1
                                logger.info("Stream connected. Reading response...")
                                first_chunk = True
                                async for line in response.aiter_lines():
                                    if line.startswith(":") : logger.debug(f"SSE Comment: {line}"); continue
                                    if line.startswith("data: "):
                                        data_str = line[len("data: "):].strip()
                                        if data_str == "[DONE]": logger.debug("Received [DONE] marker."); break
                                        if not data_str: continue
                                        try:
                                            chunk_json = json.loads(data_str)
                                            if first_chunk and "id" in chunk_json: generation_id_holder[0] = chunk_json["id"]; logger.debug(f"Captured GID: {generation_id_holder[0]}"); first_chunk = False
                                            if "error" in chunk_json and chunk_json["error"]: error_detail = chunk_json["error"]; logger.error(f"Error in stream: {error_detail}"); raise OpenRouterError(f"Provider stream error: {error_detail.get('message','Unknown')}")
                                            delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                                            content_chunk = delta.get("content")
                                            if content_chunk: yield content_chunk
                                        except json.JSONDecodeError: logger.warning(f"Skipping non-JSON: {line}", exc_info=False)
                                        except IndexError: logger.warning(f"Unexpected chunk struct: {chunk_json}", exc_info=False)
                                    elif line.strip(): logger.warning(f"Unexpected line: {line}")
                                response.raise_for_status()
                                stream_success = True
                         finally:
                            stream_duration = time.monotonic() - stream_start_time
                            if stream_success: logger.info(f"Stream OK. Duration: {stream_duration:.3f}s. Use stats_fetcher() for cost."); self.total_request_time += stream_duration
                            else: logger.warning(f"Stream Error/Interrupt. Duration: {stream_duration:.3f}s.")
                     return _stream_generator(), stats_fetcher

            # --- Exception Handling and Retry Logic ---
            except httpx.RequestError as e: last_exception = e; logger.warning(f"Network error on attempt {attempt + 1}: {e}")
            except httpx.HTTPStatusError as e:
                 last_exception = e; logger.warning(f"HTTP error on attempt {attempt + 1}: {e.response.status_code} - {e.response.text}")
                 if 400 <= e.response.status_code < 500 and e.response.status_code not in [429, 408]: logger.error("Client error. Aborting."); self.request_count += 1; raise OpenRouterError(f"Client Error: {e.response.status_code}, aborting.") from e
            except json.JSONDecodeError as e: last_exception = e; logger.error(f"Failed to decode JSON on attempt {attempt + 1}: {e}")
            except OpenRouterError as e: last_exception = e; logger.error(f"OpenRouter error on attempt {attempt + 1}: {e}"); self.request_count += 1; raise
            except Exception as e: last_exception = e; error_type = type(e).__name__; logger.error(f"Unexpected error on attempt {attempt + 1} [{error_type}]: {e}", exc_info=True)

            # --- Retry ---
            if attempt < retries:
                wait_time = 2 ** attempt; logger.info(f"Retrying in {wait_time} seconds..."); await asyncio.sleep(wait_time)
            else:
                logger.error(f"Max retries ({retries}) reached. Failed.")
                if not (stream and self.request_count > attempt): self.request_count += 1
                if isinstance(last_exception, Exception): raise OpenRouterError(f"Failed after {retries + 1} attempts.") from last_exception
                else: raise OpenRouterError(f"Failed after {retries + 1} attempts; unspecified error.")
        return None, None

    def _log_and_track_stats(self, stats: Dict[str, Any], request_time: Optional[float] = None):
        """Logs stats and updates cumulative totals. Called *after* stats are successfully fetched."""
        cost = stats.get("cost", Decimal("0.0"))
        p_tokens = stats.get("prompt_tokens", 0)
        c_tokens = stats.get("completion_tokens", 0)
        t_tokens = stats.get("total_tokens", 0)
        error = stats.get("error")
        time_str = f", Request Time: {request_time:.3f}s" if request_time is not None else ""

        if not error:
            self.total_cost += cost
            # Log cost with sufficient precision
            logger.info(f"Request Stats: Cost: ${cost:.8f}, Tokens: {t_tokens} ({p_tokens} prompt + {c_tokens} completion){time_str}")
        else:
            logger.warning(f"Stats retrieval failed: Error: {error}{time_str}")


    def get_total_cost(self) -> Decimal:
        """Returns the total estimated cost tracked by this instance (from successful stat fetches)."""
        return self.total_cost

    def get_total_request_time(self) -> float:
        """Returns the total time spent in *successful* requests tracked by this instance."""
        return self.total_request_time

    def get_request_count(self) -> int:
        """Returns the total number of requests *attempted* by this instance."""
        return self.request_count

# --- Main Execution Block (for testing) ---
async def run_completion_task(
    engine: OpenRouterEngine,
    model: str,
    messages: List[Dict[str, Any]],
    use_stream: bool,
    task_id: int
) -> Dict[str, Any]:
    """Helper function to run a single completion and gather results/stats."""
    task_start_time = time.monotonic()
    result_data = {
        "task_id": task_id, "model": model, "stream": use_stream, "status": "FAILED", "content_preview": None,
        "cost": Decimal("0.0"), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration": 0.0, "error": None
    }
    stats = None
    content_buffer = ""

    try:
        response, stats_or_fetcher = await engine.chat_completion(model, messages, stream=use_stream, temperature=0.7, max_tokens=70)
        if use_stream:
            if response and stats_or_fetcher:
                async for chunk in response: content_buffer += chunk; # Limit preview length in production if needed
                content_buffer = content_buffer[:100] + "..." if len(content_buffer) > 100 else content_buffer
                stats = await stats_or_fetcher()
            else: result_data["error"] = "Stream initiation failed"
        else: # Non-streaming
            if response and stats_or_fetcher:
                stats = stats_or_fetcher
                try: content_buffer = response.get("choices")[0].get("message").get("content","")[:100]; content_buffer += "..." if len(content_buffer)==100 else ""
                except (AttributeError, IndexError, TypeError): content_buffer = "[Error parsing content]"
            else: result_data["error"] = "Non-stream request failed"
        # Process stats
        if stats:
            if not stats.get("error"):
                result_data["status"] = "SUCCESS"
                result_data["cost"] = stats.get("cost", Decimal("0.0"))
                result_data["prompt_tokens"] = stats.get("prompt_tokens", 0)
                result_data["completion_tokens"] = stats.get("completion_tokens", 0)
                result_data["total_tokens"] = stats.get("total_tokens", 0)
            else: result_data["error"] = f"Stats fetch failed: {stats.get('error')}"
        result_data["content_preview"] = content_buffer if content_buffer else "[No content]"
    except OpenRouterError as e: logger.error(f"Task {task_id} ({model}) failed: {e}"); result_data["error"] = str(e)
    except Exception as e: logger.error(f"Task {task_id} ({model}) failed unexpectedly: {e}", exc_info=True); result_data["error"] = f"Unexpected: {type(e).__name__}"
    result_data["duration"] = time.monotonic() - task_start_time
    return result_data

async def main_test():
    """Tests the OpenRouterEngine with concurrent requests."""
    print("--- Running OpenRouterEngine Concurrent Test ---")
    if not OPENROUTER_API_KEY: print("Error: `OPENROUTER_KEY` not found...", file=sys.stderr); return
    print("OpenRouter Key found.\n--- Pre-computation checks passed. ---")

    engine = None
    try:
        # --- Initialization ---
        engine = OpenRouterEngine(app_url="http://test-storyshot-concurrent.local", app_title="StoryShotConcurrentTest", api_retries=1)
        await engine.wait_for_initialization()
        if not engine.is_available(): print("Engine initialization failed.", file=sys.stderr); return
        available_models = engine.get_available_models()
        print(f"\nEngine Initialized. Found {len(available_models)} models.")

        # --- Model Selection ---
        models_to_test = []; candidate_models = ["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku-20240307", "google/gemini-flash-1.5", "mistralai/mistral-7b-instruct", "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"]
        for model in candidate_models:
            if model in available_models: models_to_test.append(model)
        if not models_to_test: print("Error: No candidate models available.", file=sys.stderr); return
        max_concurrent = 3; models_to_test = models_to_test[:max_concurrent]
        print(f"\nSelected models for concurrent testing: {models_to_test}")

        # --- Prepare Tasks ---
        tasks = []; base_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i, model_id in enumerate(models_to_test):
            messages = base_messages + [{"role": "user", "content": f"Tell me a very different short story about a brave AI. Use model {model_id} for task {i+1}."}]
            use_stream = (i % 2 == 0); task = asyncio.create_task(run_completion_task(engine, model_id, messages, use_stream, task_id=i+1), name=f"Task_{i+1}_{model_id}")
            tasks.append(task)

        # --- Run Concurrently ---
        print(f"\n--- Running {len(tasks)} tasks concurrently ---"); concurrent_start_time = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True); concurrent_end_time = time.monotonic()
        total_concurrent_duration = concurrent_end_time - concurrent_start_time
        print(f"--- Concurrent execution finished in {total_concurrent_duration:.3f} seconds ---")

        # --- Process Results ---
        print("\n--- Individual Task Results ---"); successful_tasks = 0; total_cost_all_tasks = Decimal("0.0"); total_prompt_tokens = 0; total_completion_tokens = 0
        for i, res in enumerate(results):
            print("-" * 20); task_name = tasks[i].get_name()
            if isinstance(res, Exception): logger.error(f"Task {task_name} failed catastrophically: {res}", exc_info=res); print(f"Task ID {i+1} ({task_name}): CATASTROPHIC FAILURE\n  Error: {res}")
            elif isinstance(res, dict):
                 print(f"Task ID {res['task_id']} ({res['model']}) - Stream: {res['stream']}")
                 print(f"  Status:   {res['status']}")
                 print(f"  Duration: {res['duration']:.3f}s")
                 if res['status'] == "SUCCESS":
                     successful_tasks += 1; cost = res['cost']; total_cost_all_tasks += cost; p_tokens = res['prompt_tokens']; c_tokens = res['completion_tokens']; t_tokens = p_tokens + c_tokens
                     total_prompt_tokens += p_tokens; total_completion_tokens += c_tokens
                     print(f"  Cost:     ${cost:.8f}") # Display cost with precision
                     print(f"  Tokens:   {t_tokens} ({p_tokens} prompt + {c_tokens} completion)"); print(f"  Preview:  {res['content_preview']}")
                 else: print(f"  Error:    {res['error']}"); print(f"  Preview:  {res['content_preview']}")
            else: print(f"Task ID {i+1}: UNEXPECTED RESULT TYPE: {type(res)}")

        # --- Final Summary ---
        print("\n" + "=" * 30 + " Concurrent Run Summary " + "=" * 30)
        print(f"Total Wall-Clock Time: {total_concurrent_duration:.3f} seconds")
        print(f"Tasks Attempted / Successful: {len(tasks)} / {successful_tasks}")
        print(f"Total Cost (Sum of successful tasks): ${total_cost_all_tasks:.8f}") # Display total cost with precision
        print(f"Total Tokens (Sum of successful tasks): {total_prompt_tokens + total_completion_tokens} ({total_prompt_tokens} prompt + {total_completion_tokens} completion)")
        print("-" * 76)
        print("Engine Internal Metrics:"); print(f"  Engine Total Requests Attempted: {engine.get_request_count()}")
        print(f"  Engine Total Cost (Successful Stats): ${engine.get_total_cost():.8f}") # Display engine total cost with precision
        print(f"  Engine Total Successful Request Duration (Sum): {engine.get_total_request_time():.3f}s")
        print("=" * 76)

    except OpenRouterError as e: print(f"\nAn OpenRouter specific error occurred: {e}", file=sys.stderr)
    except Exception as e: logger.error(f"\nAn unexpected error occurred: {e}", exc_info=True)
    finally:
        if engine: await engine.close()

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8): pass
    try: asyncio.run(main_test())
    except KeyboardInterrupt: print("\nTest interrupted by user.")
    except RuntimeError as e:
       if "cannot schedule new futures after shutdown" in str(e): print("\nRuntimeError: Cannot schedule new futures after shutdown.", file=sys.stderr)
       else: logger.error(f"RuntimeError during execution: {e}", exc_info=True)
