# File: sentence2storyboard_.py
"""
Sentence to Storyboard Module (sentence2storyboard_.py)

Handles generating a sequence of image prompts for a given sentence based on
its corresponding audio duration. Uses an LLM via OpenRouterEngine to create the prompts.

This module is part of the 'storyshot' project. It's called by `storyshot_.py`
after TTS success to generate the visual descriptions for each sentence segment.

API Info (Relies on OpenRouterEngine):
- Leverages the `OpenRouterEngine` class for LLM interaction.
- Target Model (Configurable): Defaulting to a Gemini 1.5 model,
  but can be changed during instantiation or by editing the constant.
- Key Function: Takes a sentence and its audio duration, calculates the needed
  number of ~2-second visual "shots," prompts the LLM to generate descriptive
  image prompts for these shots, ensuring the output is a Python list.

Workflow:
1. Receives sentence text and audio duration from the orchestrator (`storyshot_.py`).
2. Calculates the number of prompts required (ceil(duration / 2.0)).
3. Constructs a specific prompt for the LLM, instructing it to generate exactly
   that many image prompts as a Python list of strings.
4. Calls the OpenRouterEngine's `chat_completion` method.
5. Parses the LLM's response, specifically looking for a Python list string.
   Uses `ast.literal_eval` for safe parsing.
6. Validates the parsed list (is it a list? does it have the right number of prompts?).
7. Returns a dictionary containing the prompts list, cost, timing, and token info.

Key Features & Outputs:
- Async Ready: Uses `async`/`await`.
- Cost/Time Tracking: Leverages `OpenRouterEngine`'s stats for cost, time, and tokens.
  Maintains its own cumulative totals.
- Robust Parsing: Uses `ast.literal_eval` for safe parsing of the expected list output.
- Result Dictionary: The `generate_prompts_for_sentence` method returns:
    {
        "status": "success" | "failed",
        "prompts": ["prompt 1", ...],  # List of strings
        "cost": Decimal("..."),        # Cost from OpenRouter stats
        "llm_request_time": float,     # Request duration from OpenRouter stats
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "error_message": str | None    # Details on failure
    }
- Retries: Relies on the retry mechanism built into `OpenRouterEngine`.

Usage Notes / Integration with storyshot_.py:
- Dependencies: Requires `openrouter_engine_.py`, `python-dotenv`.
- Initialization: An `SentenceToStoryboard` instance needs an initialized
  `OpenRouterEngine` passed to it.
- Caching: This module does *not* handle caching; caching based on sentence text
  is managed by the calling script (`storyshot_.py`).
- Error Handling: Raises `SentenceToStoryboardError` on critical failure, or
  returns a dict with status="failed".
"""

import os
import sys
import logging
import time
import asyncio
import math
import ast # For safely evaluating the string representation of a list
from decimal import Decimal
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv # <<<<<<<<<<<<<<<<<<< ADDED IMPORT

# --- Import Project Modules ---
try:
    # Assuming openrouter_engine_.py is in the same directory or Python path
    from openrouter_engine_ import OpenRouterEngine, OpenRouterError
except ImportError as e:
    print(f"Error: Could not import OpenRouterEngine from openrouter_engine_.py. Details: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
load_dotenv() # <<<<<<<<<<<<<<<<<<< ADDED CALL

# !!! IMPORTANT: Verify this model ID on openrouter.ai/models !!!
# Replace 'google/gemini-1.5-pro-preview-0325' with a currently valid model ID
DEFAULT_LLM_MODEL = "google/gemini-2.5-pro-preview-03-25" # Example: Using latest tag. CHECK THIS.
TARGET_SHOT_DURATION_SECONDS = 2.0

# --- Setup Logging ---
# Keep logging setup as is
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity if needed

class SentenceToStoryboardError(Exception):
    """Custom exception for SentenceToStoryboard errors."""
    pass

class SentenceToStoryboard:
    """
    Generates storyboard image prompts for sentences using an LLM via OpenRouter.
    """
    def __init__(self, engine: OpenRouterEngine, llm_model: str = DEFAULT_LLM_MODEL):
        """
        Initializes the SentenceToStoryboard handler.

        Args:
            engine (OpenRouterEngine): An initialized instance of the OpenRouterEngine.
            llm_model (str): The specific OpenRouter model ID to use for generation.
        """
        if not isinstance(engine, OpenRouterEngine) or not engine.is_available():
            raise ValueError("An initialized and available OpenRouterEngine instance is required.")

        self.engine = engine
        # Check if the chosen model is actually available in the engine instance
        # or_engine.available_models.keys()
        # import code; code.interact(local={**locals(), **globals()})
        if llm_model not in self.engine.get_available_models():
             logger.warning(f"Chosen LLM model '{llm_model}' not found in engine's list at init time. Generation may fail if not updated.")
             # You could raise an error here, or just allow it and let chat_completion handle it
             # raise ValueError(f"Model '{llm_model}' not found in OpenRouterEngine available models.")
        self.llm_model = llm_model
        logger.info(f"SentenceToStoryboard initialized. Using LLM: {self.llm_model} via OpenRouter.")

        # Internal cumulative tracking (for the lifetime of this instance)
        self.total_requests = 0
        self.total_successful_requests = 0
        self.total_failed_requests = 0
        self.total_cost = Decimal("0.0")
        self.total_llm_time = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _calculate_num_prompts(self, audio_duration_seconds: float) -> int:
        """Calculates the number of prompts needed based on target shot duration."""
        if audio_duration_seconds <= 0:
            return 1 # Need at least one prompt even for very short audio
        # Use math.ceil for correct ceiling division
        num_prompts = math.ceil(audio_duration_seconds / TARGET_SHOT_DURATION_SECONDS)
        return max(1, int(num_prompts)) # Ensure at least 1

    def _format_llm_prompt(self, sentence_text: str, num_prompts: int) -> List[Dict[str, str]]:
        """Creates the message list for the OpenRouter chat completion request."""
        system_prompt = """You are an AI assistant specialized in creating visual storyboard sequences. Your task is to generate image prompts based on a sentence. Respond ONLY with a valid Python list of strings, containing the requested number of prompts. Do not include any additional text, explanations, markdown, or conversational elements."""

        user_prompt = f"""Generate exactly {num_prompts} distinct image prompts that visually represent the progression of the following sentence:
"{sentence_text}"

Each prompt should describe a single, static scene suitable for an AI image generator (like DALL-E 3 or Midjourney). Aim for clear, descriptive language. Maintain reasonable visual consistency across the prompts for this sentence.

Output Format Reminder: Your response MUST be ONLY a Python list containing {num_prompts} string elements.
Example: ["A close-up of worried eyes reflecting a computer screen.", "A wider shot showing a person typing frantically in a dimly lit room.", "An exterior shot of a building at night."]

Generate the prompts now.
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_llm_response(self, response_text: Optional[str], expected_num_prompts: int) -> List[str]:
        """Safely parses the LLM response to extract the Python list of prompts."""
        if not response_text:
            raise SentenceToStoryboardError("LLM response was empty.")

        # Clean potential markdown backticks and surrounding text
        cleaned_text = response_text.strip()
        list_start_index = cleaned_text.find('[')
        list_end_index = cleaned_text.rfind(']')

        if list_start_index == -1 or list_end_index == -1 or list_start_index >= list_end_index:
            # Try removing common prefixes if list not found easily
            prefixes_to_try = ["```python", "```json", "```", "Here is the list:", "Sure, here's the list:", "Okay, here is the list:\n"]
            original_cleaned_text = cleaned_text
            for prefix in prefixes_to_try:
                 if cleaned_text.startswith(prefix):
                     cleaned_text = cleaned_text[len(prefix):].strip()
                     list_start_index = cleaned_text.find('[')
                     list_end_index = cleaned_text.rfind(']')
                     if list_start_index != -1 and list_end_index != -1 and list_start_index < list_end_index:
                         break # Found it after stripping prefix
            else: # If loop finished without break
                 cleaned_text = original_cleaned_text # Restore if prefixes didn't help
                 logger.error(f"Could not find valid list structure ('[...]') in response: '{cleaned_text[:200]}...'")
                 raise SentenceToStoryboardError(f"Could not find a valid list structure ('[...]') in the response.")


        list_str = cleaned_text[list_start_index : list_end_index + 1]

        try:
            # Use ast.literal_eval for safety - it only parses Python literals
            parsed_object = ast.literal_eval(list_str)
        except (ValueError, SyntaxError, TypeError, MemoryError) as e: # Added MemoryError just in case
            logger.error(f"ast.literal_eval failed on snippet: '{list_str[:200]}...'. Error: {e}")
            raise SentenceToStoryboardError(f"Failed to parse LLM response as Python list: {e}.") from e

        if not isinstance(parsed_object, list):
            raise SentenceToStoryboardError(f"Parsed response is not a list, but a {type(parsed_object).__name__}.")

        # Validate content type (are elements strings?)
        if not all(isinstance(item, str) for item in parsed_object):
             non_strings = [item for item in parsed_object if not isinstance(item, str)]
             logger.warning(f"List contains non-string elements: {non_strings[:3]}...")
             # Attempt to stringify them as a fallback? Or error out. Let's error for now.
             raise SentenceToStoryboardError(f"List contains non-string elements.")

        # Validate length (optional, warn instead of error)
        if len(parsed_object) != expected_num_prompts:
            logger.warning(f"LLM generated {len(parsed_object)} prompts, but {expected_num_prompts} were expected for sentence. Using the {len(parsed_object)} generated ones.")
            # Decide whether to raise an error or just warn. Warning seems more pragmatic.

        return parsed_object # Return the list of strings

    async def generate_prompts_for_sentence(self, sentence_text: str, audio_duration_seconds: float) -> Dict[str, Any]:
        """
        Generates image prompts for a sentence using the configured LLM.

        Args:
            sentence_text (str): The sentence to visualize.
            audio_duration_seconds (float): The duration of the corresponding audio.

        Returns:
            Dict[str, Any]: A dictionary containing status, prompts, cost, time, etc.
                           See class docstring for structure.
        """
        start_time = time.monotonic()
        self.total_requests += 1
        num_prompts_needed = self._calculate_num_prompts(audio_duration_seconds)
        log_prefix = f"Storyboard Gen (Sentence: \"{sentence_text[:40]}...\")"
        logger.info(f"{log_prefix}: Requesting {num_prompts_needed} prompts for {audio_duration_seconds:.2f}s duration.")

        # --- Prepare result dictionary structure ---
        result = {
            "status": "failed", "prompts": [], "cost": Decimal("0.0"),
            "llm_request_time": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "error_message": None
        }

        # --- Pre-check if model is known to be unavailable ---
        # (Optional, but might save an API call if detected at init)
        # if self.llm_model not in self.engine.get_available_models():
        #     result["error_message"] = f"Model '{self.llm_model}' known to be unavailable from init check."
        #     logger.error(f"{log_prefix}: {result['error_message']}")
        #     self.total_failed_requests += 1
        #     return result

        try:
            messages = self._format_llm_prompt(sentence_text, num_prompts_needed)

            # --- Call OpenRouter Engine ---
            response_json, stats = await self.engine.chat_completion(
                model=self.llm_model,
                messages=messages,
                stream=False,
                # Optional: Add temperature, max_tokens etc. if needed
                # temperature=0.7,
                # max_tokens=num_prompts_needed * 150 # Estimate tokens per prompt + overhead
            )

            # --- Process Response and Stats ---
            req_duration = time.monotonic() - start_time # Measure full local duration

            # Check if the call succeeded systemically (model found, basic request OK)
            if response_json is None and stats is None:
                 # This happens if the model wasn't found in chat_completion
                 # Error is already logged by OpenRouterEngine
                 result["error_message"] = f"Model '{self.llm_model}' not found or invalid."
                 raise SentenceToStoryboardError(result["error_message"])

            if stats and not stats.get("error"):
                 result.update({
                     "cost": stats.get("cost", Decimal("0.0")),
                     "prompt_tokens": stats.get("prompt_tokens", 0),
                     "completion_tokens": stats.get("completion_tokens", 0),
                     "total_tokens": stats.get("total_tokens", 0),
                     "llm_request_time": req_duration # Use local duration as proxy for request time
                 })
                 # Update internal tracking
                 self.total_cost += result["cost"]
                 self.total_llm_time += result["llm_request_time"]
                 self.total_prompt_tokens += result["prompt_tokens"]
                 self.total_completion_tokens += result["completion_tokens"]
            else:
                 # Handle case where chat_completion succeeded but stats retrieval failed
                 stats_error = stats.get('error', 'Unknown stats error') if stats else "Stats object missing/None"
                 result["error_message"] = f"LLM call succeeded but stats retrieval failed: {stats_error}"
                 logger.warning(f"{log_prefix}: {result['error_message']}")
                 result["llm_request_time"] = req_duration # Still record duration

            # Ensure response_json is valid before proceeding
            if not response_json or "choices" not in response_json or not response_json["choices"]:
                # This case might happen if the API returned 200 OK but empty/malformed JSON
                # Or if stats retrieval failed *and* response was somehow None (less likely)
                raise SentenceToStoryboardError(f"Invalid or empty response structure from OpenRouter API. Response Json: {response_json}")

            # Extract the text content
            llm_text_output = response_json["choices"][0].get("message", {}).get("content")

            # Parse the text to get the list
            prompts_list = self._parse_llm_response(llm_text_output, num_prompts_needed)

            # Success!
            result["status"] = "success"
            result["prompts"] = prompts_list
            result["error_message"] = None # Clear any previous warning message about stats
            self.total_successful_requests += 1
            logger.info(f"{log_prefix}: Successfully generated {len(prompts_list)} prompts. Cost: ${result['cost']:.6f}, Time: {result['llm_request_time']:.3f}s")

        # Catch specific exceptions first
        except SentenceToStoryboardError as e: # Errors from this module's logic (parsing, validation)
             self.total_failed_requests += 1
             error_msg = f"Prompt generation/parsing error: {e}"
             logger.error(f"{log_prefix}: {error_msg}", exc_info=False)
             result["error_message"] = str(e) # Use the specific error message
        except OpenRouterError as e: # Errors from the OpenRouterEngine communication
            self.total_failed_requests += 1
            error_msg = f"OpenRouter API error: {e}"
            logger.error(f"{log_prefix}: {error_msg}", exc_info=False)
            result["error_message"] = error_msg
        # Catch any other unexpected exceptions
        except Exception as e:
            self.total_failed_requests += 1
            error_msg = f"Unexpected error: {type(e).__name__} - {e}"
            logger.error(f"{log_prefix}: {error_msg}", exc_info=True)
            result["error_message"] = error_msg

        return result

    # --- Getter methods for cumulative stats ---
    def get_stats(self) -> Dict[str, Any]:
        """ Returns cumulative statistics for this instance. """
        return {
            "total_requests": self.total_requests,
            "successful": self.total_successful_requests,
            "failed": self.total_failed_requests,
            "total_cost": self.total_cost,
            "total_llm_time": self.total_llm_time,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }

# --- Main Execution Block (for testing) ---
async def main_test():
    """Tests the SentenceToStoryboard module."""
    print("--- Running SentenceToStoryboard Test ---")
    # load_dotenv() is called globally now
    if not os.getenv("OPENROUTER_KEY"):
        print("Error: OPENROUTER_KEY not found in environment variables.", file=sys.stderr)
        return

    or_engine = None
    try:
        # --- Initialize OpenRouterEngine ---
        print("Initializing OpenRouterEngine...")
        or_engine = OpenRouterEngine(app_title="StoryShot_StoryboardTest")
        await or_engine.wait_for_initialization()
        if not or_engine.is_available():
             print("OpenRouterEngine initialization failed.", file=sys.stderr)
             return
        print(f"OpenRouterEngine Initialized. Found {len(or_engine.get_available_models())} models.")
        # or_engine.available_models.keys()
        # import code; code.interact(local={**globals(), **locals()})

        # --- Initialize SentenceToStoryboard ---
        # It will now log a warning if the default model isn't found at init
        storyboard_gen = SentenceToStoryboard(or_engine)

        # --- Test Data ---
        test_cases = [
            {"sentence": "The quick brown fox jumps over the lazy dog.", "duration": 3.8},
            {"sentence": "A lone figure stood silhouetted against the setting sun.", "duration": 5.1},
            {"sentence": "Warning lights flashed.", "duration": 1.5},
            {"sentence": "The city skyline glittered under a blanket of stars, silent and watchful.", "duration": 7.0},
        ]

        # --- Run Tests ---
        print(f"\n--- Generating Storyboard Prompts (using model: {storyboard_gen.llm_model}) ---")
        tasks = []
        for i, case in enumerate(test_cases):
            task = asyncio.create_task(
                storyboard_gen.generate_prompts_for_sentence(case["sentence"], case["duration"]),
                name=f"StoryboardTask_{i+1}"
            )
            tasks.append(task)

        test_results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- Display Results ---
        print("\n--- Test Results ---")
        for i, result in enumerate(test_results):
             print("-" * 20)
             task_name = tasks[i].get_name()
             sentence = test_cases[i]['sentence']
             duration = test_cases[i]['duration']
             expected_prompts = storyboard_gen._calculate_num_prompts(duration)
             print(f"{task_name} (Sentence: \"{sentence[:50]}...\", Duration: {duration:.1f}s, Expected Prompts: {expected_prompts})")

             if isinstance(result, Exception): # Caught by gather
                 print(f"  Status: CRITICAL FAILURE")
                 print(f"  Error: {result}")
             elif isinstance(result, dict): # Normal return path
                 print(f"  Status: {result['status']}")
                 if result['status'] == 'success':
                     print(f"  Prompts ({len(result['prompts'])} generated):")
                     for j, prompt in enumerate(result['prompts']):
                         print(f"    {j+1}. {prompt}")
                     print(f"  Cost: ${result['cost']:.6f}")
                     print(f"  Time: {result['llm_request_time']:.3f}s")
                     print(f"  Tokens: {result['total_tokens']} ({result['prompt_tokens']}p + {result['completion_tokens']}c)")
                 else: # Status was 'failed'
                     print(f"  Error: {result['error_message']}")
             else: # Should not happen
                print(f"  Status: UNEXPECTED RESULT TYPE: {type(result)}")

        # --- Display Cumulative Stats ---
        print("\n" + "=" * 20 + " Cumulative Stats " + "=" * 20)
        final_stats = storyboard_gen.get_stats()
        print(f"Total Requests: {final_stats['total_requests']} (Success: {final_stats['successful']}, Failed: {final_stats['failed']})")
        print(f"Total Cost: ${final_stats['total_cost']:.6f}")
        print(f"Total LLM Time (Sum): {final_stats['total_llm_time']:.3f}s")
        total_tokens = final_stats['total_prompt_tokens'] + final_stats['total_completion_tokens']
        print(f"Total Tokens: {total_tokens} ({final_stats['total_prompt_tokens']}p + {final_stats['total_completion_tokens']}c)")
        print("=" * 58)


    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
    finally:
        if or_engine:
            print("Closing OpenRouterEngine client...")
            await or_engine.close()
            print("Client closed.")

if __name__ == "__main__":
    # Handle potential issues with asyncio event loop policy on Windows for httpx/aiohttp
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_test())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
