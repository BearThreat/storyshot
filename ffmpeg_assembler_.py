# File: ffmpeg_assembler_.py
"""
FFmpeg Assembly Module (ffmpeg_assembler_.py) - V5 (Robust Input Checks)

Handles the assembly of individual video clips and audio segments into a final
movie file using FFmpeg. It is orchestrated by storyshot_.py.

Responsibilities:
- Take paths to successfully generated sentence video clips and audio files.
- Concatenate video clips for each sentence.
- Merge the concatenated video with the sentence's audio, trimming to shortest.
- Concatenate all processed sentence clips into a final movie.
- Handle potential FFmpeg errors.
- Perform basic verification on intermediate and final outputs.

V5 Changes:
- Modified `assemble_movie` to perform detailed input eligibility checks *before*
  creating tasks for `_assemble_sentence_clip`.
- Ensures assembly is attempted only for sentences with valid TTS audio
  and at least one valid video clip based on cache data.
- Skips ineligible sentences gracefully.
- Uses optional stream mapping (`-map 0:v? -map 0:a?`) in final concat for resilience.
- Keeps V4 behaviour of retaining intermediate clips on final concat/verification failure.

Requires FFmpeg and FFprobe to be installed and accessible in the system's PATH.
"""

import os
import sys
import logging
import asyncio
import shlex
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

# --- Configuration ---
TEMP_DIR = Path("temp_files")
FFMPEG_LIST_SUBDIR = TEMP_DIR / "ffmpeg_concat_lists"
FINAL_CLIPS_SUBDIR = TEMP_DIR / "final_clips"
TEMP_VIDEO_SUBDIR = TEMP_DIR / "intermediate_videos"

# --- Setup Logging (Only configure if no handlers exist) ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     console_handler = logging.StreamHandler(sys.stdout)
     console_handler.setFormatter(log_formatter)
     logger.addHandler(console_handler)
     logger.propagate = False # Prevent double logging if root logger also configured
logger.setLevel(logging.INFO) # Set level for this logger instance
logging.getLogger("asyncio").setLevel(logging.WARNING)

class FFmpegAssemblyError(Exception):
    """Custom exception for FFmpeg assembly errors."""
    pass

class FFmpegAssembler:
    """Handles video and audio assembly using FFmpeg. (V5)"""

    def __init__(self, final_movie_path: str | Path = "final_movie.mp4"):
        self.final_movie_path = Path(final_movie_path)
        self.list_dir = FFMPEG_LIST_SUBDIR
        self.clip_dir = FINAL_CLIPS_SUBDIR
        self.temp_video_dir = TEMP_VIDEO_SUBDIR

        # Create directories
        self.list_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.temp_video_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FFmpegAssembler V5 initialized. Output: {self.final_movie_path}")
        logger.info("V5 Changes: Added eligibility checks, optional final map streams.")

    async def _run_command_async(self, command: str, command_args: List[str], log_prefix: str) -> Tuple[bool, str, str]:
        # (This function remains the same as V3/V4)
        full_command_str = f"{command} {' '.join(shlex.quote(str(arg)) for arg in command_args)}"
        logger.debug(f"{log_prefix}: Executing: {full_command_str}")
        start_time = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            command,
            *command_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode('utf-8', errors='replace').strip()
        stderr = stderr_bytes.decode('utf-8', errors='replace').strip()
        duration = time.monotonic() - start_time

        if process.returncode == 0:
            logger.debug(f"{log_prefix}: Command successful ({duration:.3f}s).")
            if stderr: logger.debug(f"{log_prefix}: Stderr Output (Success):\n---\n{stderr}\n---")
            if stdout: logger.debug(f"{log_prefix}: Stdout Output (Success):\n---\n{stdout}\n---")
            return True, stdout, stderr
        else:
            error_message = f"{log_prefix}: Command failed with return code {process.returncode} ({duration:.3f}s)."
            logger.error(error_message)
            if stdout: logger.error(f"{log_prefix}: Stdout (Failure):\n---\n{stdout}\n---")
            if stderr: logger.error(f"{log_prefix}: Stderr (Failure):\n---\n{stderr}\n---")
            return False, stdout, stderr

    async def _run_ffmpeg_async(self, command_args: List[str], log_prefix: str = "FFmpeg") -> Tuple[bool, str, str]:
        return await self._run_command_async("ffmpeg", command_args, log_prefix)

    async def _run_ffprobe_async(self, command_args: List[str], log_prefix: str = "FFprobe") -> Tuple[bool, str, str]:
        return await self._run_command_async("ffprobe", command_args, log_prefix)

    async def _assemble_sentence_clip(self, sentence_key: str, sentence_cache_entry: Dict[str, Any], sentence_hash: str) -> Optional[Path]:
        """
        Assembles the video and audio for a single sentence. (V4 Logic - Seems Robust)
        This is called *after* eligibility checks in assemble_movie (V5).
        """
        log_prefix = f"AssembleSentence[{sentence_hash}]"
        temp_files_to_clean = []
        final_clip_path_obj = self.clip_dir / f"final_sentence_clip_{sentence_hash}.mp4"

        try:
            # 1. Gather Inputs (Should already be validated by caller in V5, but double-check paths)
            tts_output = sentence_cache_entry.get("tts_output")
            sb_prompts = sentence_cache_entry.get("storyboard_prompts")

            # Basic null checks (more detailed checks happened before task creation in V5)
            if not tts_output or not sb_prompts:
                logger.warning(f"{log_prefix}: Skipping - Missing TTS or Storyboard data.")
                return None

            audio_path_str = tts_output.get("audio_path")
            if not audio_path_str:
                 logger.warning(f"{log_prefix}: Skipping - Audio path missing in TTS output.")
                 return None
            audio_path = Path(audio_path_str)
            if not audio_path.exists():
                logger.error(f"{log_prefix}: CRITICAL - Source audio file NOT FOUND: {audio_path}.")
                return None
            else:
                 try:
                     audio_size = audio_path.stat().st_size
                     logger.info(f"{log_prefix}: Verified source audio file exists: {audio_path.name} ({audio_size} bytes)")
                     if audio_size < 100: # Arbitrary small size check
                          logger.warning(f"{log_prefix}: Source audio file {audio_path.name} is suspiciously small ({audio_size} bytes).")
                 except Exception as stat_err:
                      logger.warning(f"{log_prefix}: Could not get size for source audio {audio_path.name}: {stat_err}")

            # Gather successful video paths
            image_outputs = sb_prompts.get("image_outputs", [])
            successful_video_paths = []
            if isinstance(image_outputs, list):
               for img_idx, img_out in enumerate(image_outputs):
                   if isinstance(img_out, dict) and img_out.get("status") == "success":
                        vid_out = img_out.get("video_output")
                        if isinstance(vid_out, dict) and vid_out.get("status") == "success":
                             vid_path_str = vid_out.get("video_path")
                             if vid_path_str:
                                 vid_path = Path(vid_path_str)
                                 if vid_path.exists():
                                     successful_video_paths.append(vid_path)
                                 else:
                                     logger.warning(f"{log_prefix}: Source video #{img_idx+1} file NOT FOUND: {vid_path_str}")
                             else:
                                  logger.warning(f"{log_prefix}: Source video path missing for image #{img_idx+1}")

            if not successful_video_paths:
                logger.warning(f"{log_prefix}: Skipping assembly - No available/successful source video clips found for sentence.")
                return None
            logger.info(f"{log_prefix}: Preparing to assemble {len(successful_video_paths)} videos and audio {audio_path.name}.")


            # 2. Create Video Concatenation List File (remains same)
            concat_list_path = self.list_dir / f"concat_list_{sentence_hash}.txt"
            temp_files_to_clean.append(concat_list_path)
            try:
                with open(concat_list_path, "w", encoding="utf-8") as f:
                    for vid_path in successful_video_paths:
                        # Use resolved absolute paths with forward slashes within single quotes
                        abs_path_str = str(vid_path.resolve()).replace('\\', '/')
                        f.write(f"file '{abs_path_str}'\n")
                logger.debug(f"{log_prefix}: Created video concat list: {concat_list_path.name}")
            except IOError as e: raise FFmpegAssemblyError(f"Failed to write video concat list: {e}") from e

            # 3. Concatenate Video Clips (Robust Re-encode - remains same)
            temp_concat_video_path = self.temp_video_dir / f"intermediate_concat_video_{sentence_hash}.mp4"
            temp_files_to_clean.append(temp_concat_video_path)
            logger.info(f"{log_prefix}: Re-encoding intermediate video ({len(successful_video_paths)} clips) -> {temp_concat_video_path.name}...")
            # Command: concat demuxer, re-encode with fast preset, video only
            concat_cmd = [
                "-y",                 # Overwrite output without asking
                "-f", "concat",       # Use concat demuxer
                "-safe", "0",         # Allow unsafe file paths (needed for absolute)
                "-i", str(concat_list_path.resolve()), # Input list file
                "-c:v", "libx264",    # Re-encode video stream
                "-preset", "fast",    # Faster encoding preset
                "-crf", "23",         # Constant Rate Factor (quality, lower is better)
                "-map", "0:v",        # Map only the video stream from input group 0
                "-an",                # No audio in this intermediate file
                str(temp_concat_video_path.resolve()) # Output path
            ]
            success, _, stderr_concat = await self._run_ffmpeg_async(concat_cmd, f"{log_prefix}-VidConcatReEncode")
            if not success: raise FFmpegAssemblyError(f"Intermediate video concat failed. Stderr: {stderr_concat[-500:]}")
            concat_vid_size = temp_concat_video_path.stat().st_size if temp_concat_video_path.exists() else -1
            if concat_vid_size < 1000: # Basic size check
                raise FFmpegAssemblyError(f"Intermediate video {temp_concat_video_path.name} invalid (size: {concat_vid_size} bytes). Encoding likely failed.")
            logger.info(f"{log_prefix}: Intermediate video re-encoded ({concat_vid_size} bytes).")

            # 4. Combine Intermediate Video and Sentence Audio (remains same)
            logger.info(f"{log_prefix}: Merging video '{temp_concat_video_path.name}' + audio '{audio_path.name}' -> {final_clip_path_obj.name}...")
            # Command: copy video stream, re-encode audio, trim to shortest input stream duration
            merge_cmd = [
                "-y",
                "-i", str(temp_concat_video_path.resolve()), # Input 0: video
                "-i", str(audio_path.resolve()),            # Input 1: audio
                "-map", "0:v:0",        # Map video stream from input 0
                "-map", "1:a:0",        # Map audio stream from input 1
                "-c:v", "copy",         # Copy video stream without re-encoding
                "-c:a", "aac",          # Re-encode audio to AAC (commonly compatible)
                "-b:a", "128k",         # Audio bitrate
                "-shortest",            # Finish encoding when the shortest input stream ends
                str(final_clip_path_obj.resolve()) # Output path
            ]
            success, _, stderr_merge = await self._run_ffmpeg_async(merge_cmd, f"{log_prefix}-AudVidMerge")
            if not success: raise FFmpegAssemblyError(f"Audio/Video merge command failed. Stderr: {stderr_merge[-500:]}")
            final_clip_size = final_clip_path_obj.stat().st_size if final_clip_path_obj.exists() else -1
            if final_clip_size < 1500: # Basic size check
                raise FFmpegAssemblyError(f"Final sentence clip {final_clip_path_obj.name} invalid (size: {final_clip_size} bytes). Merge likely failed.")
            logger.info(f"{log_prefix}: Merge command finished for {final_clip_path_obj.name} ({final_clip_size} bytes).")

            # 5. VERIFY AUDIO (using enhanced check - remains same V4->V5)
            logger.info(f"{log_prefix}: Verifying audio stream in {final_clip_path_obj.name}...")
            has_audio, audio_details = await self._check_stream_details(final_clip_path_obj, 'audio')
            if not has_audio:
                logger.error(f"{log_prefix}: >>> AUDIO VERIFICATION FAILED <<< Ffprobe detected no valid audio stream ({audio_details}) in {final_clip_path_obj.name} after merge!")
                raise FFmpegAssemblyError(f"Merged clip {final_clip_path_obj.name} missing valid audio stream ({audio_details}).")
            else:
                logger.info(f"{log_prefix}: Audio stream VERIFIED in {final_clip_path_obj.name} ({audio_details}).")

            logger.info(f"{log_prefix}: Successfully created AND verified final sentence clip: {final_clip_path_obj.name}")
            return final_clip_path_obj

        except FFmpegAssemblyError as e:
            logger.error(f"{log_prefix}: Assembly failed: {e}", exc_info=False)
            if final_clip_path_obj.exists():
                try: final_clip_path_obj.unlink()
                except OSError: pass
            return None
        except Exception as e:
            logger.error(f"{log_prefix}: Assembly failed unexpectedly: {e}", exc_info=True)
            if final_clip_path_obj.exists():
                try: final_clip_path_obj.unlink()
                except OSError: pass
            return None
        finally:
            # 6. Clean Up Temporary Files (intermediate video, list file)
            logger.debug(f"{log_prefix}: Cleaning up sentence temp files...")
            for temp_path in temp_files_to_clean:
                if temp_path.exists():
                    try: temp_path.unlink(); logger.debug(f"{log_prefix}: Removed temp: {temp_path.name}")
                    except OSError as e: logger.warning(f"{log_prefix}: Failed to cleanup {temp_path.name}: {e}")

    async def _check_stream_details(self, filepath: Path, stream_type: str = 'audio') -> Tuple[bool, str]:
        """
        Uses ffprobe to check for a stream type and extracts basic details. (V4 Logic - Seems Robust)

        Returns:
            Tuple[bool, str]: (stream_exists: bool, details: str)
            Details string includes codec, duration/bitrate if found, or error message.
        """
        log_prefix = f"FFprobeDetailCheck[{filepath.name}][{stream_type}]"
        if not filepath.exists():
             details = "File not found"
             logger.warning(f"{log_prefix}: {details}")
             return False, details

        # Request codec name, duration, and bitrate. Use ENTRY delimiters.
        entries_to_show = "stream=codec_name,duration,bit_rate"
        ffprobe_args = [
            "-v", "error",
            "-select_streams", f"{stream_type[0]}:0", # Select first stream of type 'a' or 'v'
            "-show_entries", entries_to_show,
            "-of", "default=nw=1:nk=1", # Format: value per line (no key/wrapper)
            str(filepath.resolve())
        ]
        logger.debug(f"{log_prefix}: Running ffprobe detail check...")
        success, stdout, stderr = await self._run_ffprobe_async(ffprobe_args, log_prefix)

        stream_found = False
        details = "Not Found"

        if success and stdout:
            # Output should be lines like: codec_name\nduration\nbit_rate
            lines = stdout.splitlines()
            codec = lines[0] if len(lines) > 0 else "N/A"
            duration_str = lines[1] if len(lines) > 1 else "N/A"
            bitrate_str = lines[2] if len(lines) > 2 else "N/A"

            # Basic validation: check if duration is valid number > 0 or bitrate > 0
            try: duration = float(duration_str)
            except ValueError: duration = -1.0
            try: bitrate = int(bitrate_str)
            except ValueError: bitrate = -1

            # Check for valid codec AND (valid duration OR valid bitrate)
            if codec != "N/A" and codec != "unknown" and (duration > 0 or bitrate > 0):
                 stream_found = True
                 details = f"Codec: {codec}, Duration: {duration_str}s, Bitrate: {bitrate_str}bps"
                 logger.debug(f"{log_prefix}: Check PASSED. Details: {details}")
            else:
                 details = f"Stream entry found but seems invalid (Codec: {codec}, Duration: {duration_str}, Bitrate: {bitrate_str})"
                 logger.warning(f"{log_prefix}: Check FAILED. {details}")

        elif not success:
            details = f"ffprobe command failed (Stderr: {stderr})" if stderr else "ffprobe command failed (No stderr)"
            logger.warning(f"{log_prefix}: Check FAILED. {details}")
        else: # success but no stdout
            details = "ffprobe command succeeded but produced no output (Stream likely missing)"
            logger.warning(f"{log_prefix}: Check FAILED. {details}")

        return stream_found, details

    async def assemble_movie(self, cache_data: Dict[str, Any], ordered_sentence_keys: List[str], get_sentence_hash_func: callable) -> Optional[Path]:
        """
        Orchestrates the assembly of the final movie. (V5: Robust Input Checks)
        Always attempts assembly based on available valid data in the cache.
        Skips sentences missing required audio or video inputs.
        """
        start_time = time.monotonic()
        logger.info("--- Starting Final Movie Assembly V5 (Robust Checks) ---")

        assembly_tasks = []
        task_to_key_map = {}
        eligible_keys_for_assembly = [] # Track keys that *could* be assembled

        # --- Step 1a: Filter Sentences and Prepare Assembly Tasks ---
        logger.info("Checking sentence eligibility and preparing assembly tasks...")
        skipped_sentences = 0
        for key in ordered_sentence_keys:
            sentence_hash = get_sentence_hash_func(key) # Get hash for logging prefix

            # --- Detailed Input Eligibility Check ---
            cache_entry = cache_data.get(key)
            is_eligible = False
            skip_reason = "Unknown"

            if not cache_entry:
                skip_reason = "Cache entry missing"
            else:
                # Check TTS
                tts_output = cache_entry.get("tts_output")
                if not tts_output or tts_output.get("status") != "success":
                    skip_reason = "TTS failed or missing"
                elif not tts_output.get("audio_path") or not Path(tts_output["audio_path"]).exists():
                    skip_reason = f"TTS audio file missing/not found ({tts_output.get('audio_path')})"
                else:
                    # Check Storyboard and Videos (need at least one valid video)
                    sb_prompts = cache_entry.get("storyboard_prompts")
                    if not sb_prompts or sb_prompts.get("status") != "success":
                        skip_reason = "Storyboard prompts failed or missing"
                    else:
                        image_outputs = sb_prompts.get("image_outputs", [])
                        has_any_valid_video = False
                        if isinstance(image_outputs, list):
                            for img_idx, img_out in enumerate(image_outputs):
                                if isinstance(img_out, dict) and img_out.get("status") == "success":
                                    vid_out = img_out.get("video_output")
                                    if isinstance(vid_out, dict) and vid_out.get("status") == "success":
                                        vid_path_str = vid_out.get("video_path")
                                        if vid_path_str and Path(vid_path_str).exists():
                                            has_any_valid_video = True
                                            break # Found one valid video, that's enough
                        if not has_any_valid_video:
                            skip_reason = "No successfully generated video clips found"
                        else:
                            is_eligible = True # All checks passed

            if is_eligible:
                logger.debug(f"AssembleMovie: Sentence '{key[:30]}...' is ELIGIBLE for assembly.")
                eligible_keys_for_assembly.append(key)
                task_name=f"Assemble_{sentence_hash}_{key[:10]}"
                # Create task for eligible sentence
                task = asyncio.create_task( self._assemble_sentence_clip(key, cache_entry, sentence_hash), name=task_name )
                assembly_tasks.append(task)
                task_to_key_map[task_name] = key
            else:
                skipped_sentences += 1
                logger.warning(f"AssembleMovie: Skipping sentence '{key[:30]}...' for assembly. Reason: {skip_reason}.")

        logger.info(f"Eligibility check complete: {len(eligible_keys_for_assembly)} eligible, {skipped_sentences} skipped.")

        if not assembly_tasks:
             logger.error("No eligible sentences found to assemble clips. Cannot create final movie.")
             # No cleanup needed as no tasks ran
             return None

        # --- Step 1b: Run Assembly Tasks Concurrently for Eligible Sentences ---
        logger.info(f"Executing assembly for {len(assembly_tasks)} eligible sentences concurrently...")
        results = await asyncio.gather(*assembly_tasks, return_exceptions=True)
        logger.info("Finished waiting for sentence clip assembly tasks.")

        # --- Step 1c: Collect results IN ORDER based on original eligible keys ---
        key_to_result_map = {}
        successful_clip_count, failed_clip_count = 0, 0
        generated_sentence_clips = [] # Keep track for potential cleanup/debug
        for i, result in enumerate(results):
            task_name = assembly_tasks[i].get_name()
            original_key = task_to_key_map.get(task_name)
            if not original_key:
                logger.error(f"INTERNAL ERROR: Task '{task_name}' not in map during result processing!");
                failed_clip_count += 1; continue

            if isinstance(result, Exception):
                logger.error(f"AssembleMovie: Task {task_name} ('{original_key[:30]}...') FAILED during execution: {result}", exc_info=False)
                key_to_result_map[original_key] = None; failed_clip_count += 1
            elif result is None:
                logger.warning(f"AssembleMovie: Task {task_name} ('{original_key[:30]}...') returned None (likely internal assembly failure).")
                key_to_result_map[original_key] = None; failed_clip_count += 1
            elif isinstance(result, Path) and result.exists():
                 logger.debug(f"AssembleMovie: Task {task_name} ('{original_key[:30]}...') succeeded: {result.name}")
                 key_to_result_map[original_key] = result; successful_clip_count += 1
                 generated_sentence_clips.append(result) # Track successful clips
            else:
                 logger.error(f"AssembleMovie: Task {task_name} ('{original_key[:30]}...') returned unexpected result: {result}")
                 key_to_result_map[original_key] = None; failed_clip_count += 1

        logger.info(f"Assembly task results: {successful_clip_count} success, {failed_clip_count} failed/skipped.")

        # --- Step 1d: Populate final_clip_paths_in_order with VERIFIED clips ---
        final_clip_paths_in_order: List[Path] = []
        verified_clips_for_cleanup = [] # Only list clips that pass verification
        logger.info("Performing secondary verification on generated sentence clips...")
        verification_tasks = []
        path_to_key_map = {}
        for clip_path in generated_sentence_clips: # Iterate over successfully generated GENERATED clips
            if clip_path and clip_path.exists():
                task = asyncio.create_task(self._check_stream_details(clip_path, 'audio'), name=f"Verify_{clip_path.name}")
                verification_tasks.append(task)
                path_to_key_map[task] = clip_path

        verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

        verified_paths_map = {} # Map path -> (bool, details)
        # Process verification_results (same as V4)
        for i, verify_result in enumerate(verification_results):
            task = verification_tasks[i]
            clip_path = path_to_key_map[task]
            if isinstance(verify_result, Exception):
                 logger.error(f"Secondary verification error for {clip_path.name}: {verify_result}")
                 verified_paths_map[clip_path] = (False, f"Exception: {verify_result}")
            elif isinstance(verify_result, tuple) and len(verify_result) == 2:
                 verified_paths_map[clip_path] = verify_result # (bool, details)
            else:
                 logger.error(f"Unexpected verification result type for {clip_path.name}: {type(verify_result)}")
                 verified_paths_map[clip_path] = (False, "Unexpected result type")

        # Now populate the ordered list based on the *original sentence order* and verification map
        for key in ordered_sentence_keys: # Use the master list order
             final_clip_path = key_to_result_map.get(key) # Get the generated path for this key, if any
             if final_clip_path and final_clip_path.exists():
                  is_verified, details = verified_paths_map.get(final_clip_path, (False, "Verification not run?"))
                  if is_verified:
                      logger.debug(f"Adding verified clip {final_clip_path.name} ('{key[:30]}...') to final movie list.")
                      final_clip_paths_in_order.append(final_clip_path)
                      verified_clips_for_cleanup.append(final_clip_path) # Mark for eventual cleanup IF FINAL MOVIE SUCCEEDS
                  else:
                      logger.error(f"AssembleMovie: EXCLUSION! Generated clip {final_clip_path.name} ('{key[:30]}...') FAILED secondary verification ({details}).")
             # No need for else, we only care about successfully generated and verified clips

        logger.info(f"Proceeding with {len(final_clip_paths_in_order)} successfully generated and verified sentence clips for final movie.")

        if not final_clip_paths_in_order:
            logger.error("No sentence clips were successfully generated or passed verification. Cannot create final movie.")
            self._cleanup_temp_files([], "no verified clips") # Cleanup lists/temps if any
            # Note: generated_sentence_clips might still exist in FINAL_CLIPS_SUBDIR if verification failed, which is intended V4 behaviour
            return None

        # --- Step 2: Create Final Concatenation List ---
        final_concat_list_path = self.list_dir / "final_movie_list.txt"
        files_to_cleanup_finally = [final_concat_list_path] # Always cleanup this list if we get here
        try:
             with open(final_concat_list_path, "w", encoding="utf-8") as f:
                 logger.debug(f"Writing final concat list ({final_concat_list_path.name}) with {len(final_clip_paths_in_order)} entries.")
                 for clip_path in final_clip_paths_in_order:
                     resolved_path_str = str(clip_path.resolve()).replace('\\', '/')
                     f.write(f"file '{resolved_path_str}'\n")
             logger.info(f"Created final movie concat list: {final_concat_list_path.name}")
        except IOError as e:
             logger.error(f"CRITICAL: Failed to write final concat list {final_concat_list_path}: {e}")
             # Only cleanup the list file itself on write failure
             self._cleanup_temp_files([final_concat_list_path], "list write failure")
             # Keep generated_sentence_clips
             return None

        # --- Step 3: Concatenate Final Clips (Forced Re-encode + Optional Map) ---
        logger.info(f"Concatenating {len(final_clip_paths_in_order)} clips into {self.final_movie_path} (re-encode)...")
        # Use optional maps for robustness in case a verified clip somehow misses a stream type
        final_concat_cmd_reencode = [
            "-y", "-f", "concat", "-safe", "0", "-i", str(final_concat_list_path.resolve()),
            "-map", "0:v?", "-map", "0:a?", # Use optional maps streams from input 0
            "-c:v", "libx264", "-preset", "medium", "-crf", "22", # Video Encode settings
            "-c:a", "aac", "-b:a", "192k",                         # Audio Encode settings
            "-movflags", "+faststart",                             # Optimize for web streaming
            str(self.final_movie_path.resolve())                   # Output path
        ]
        success, _, stderr_final = await self._run_ffmpeg_async(final_concat_cmd_reencode, "FinalConcat-ReEncode")

        if not success:
            logger.error(f"CRITICAL: Final movie concatenation FAILED.")
            if self.final_movie_path.exists():
                try: self.final_movie_path.unlink()
                except OSError: pass
            # Keep verified_clips_for_cleanup (the successfully generated sentence clips) and the list file for debugging
            logger.warning(f"Keeping generated sentence clips in {self.clip_dir} and list {final_concat_list_path.name} for debugging.")
            self._cleanup_temp_files([], "final concat failure - keep sentence clips") # Minimal cleanup of other temps if any
            return None
        else:
             final_movie_size = self.final_movie_path.stat().st_size if self.final_movie_path.exists() else -1
             logger.info(f"Final movie concatenation successful ({final_movie_size} bytes).")


        # --- Step 4: Final Verification ---
        logger.info(f"Verifying final movie streams: {self.final_movie_path}")
        final_has_video, video_details = await self._check_stream_details(self.final_movie_path, 'video')
        final_has_audio, audio_details = await self._check_stream_details(self.final_movie_path, 'audio')

        if not final_has_video or not final_has_audio:
             logger.error(f">>> FINAL MOVIE VERIFICATION FAILED <<<")
             logger.error(f"  Video stream: {final_has_video} ({video_details})")
             logger.error(f"  Audio stream: {final_has_audio} ({audio_details})")
             logger.warning("The final movie is likely corrupted or incomplete.")
             logger.warning(f"Keeping generated sentence clips in {self.clip_dir} and list {final_concat_list_path.name} for debugging.")
             self._cleanup_temp_files([], "final verification failed - keep sentence clips") # Minimal cleanup
             return None
        else:
             logger.info("Final movie verification PASSED.")
             logger.info(f"  Video details: {video_details}")
             logger.info(f"  Audio details: {audio_details}")


        # --- Step 5: Final Clean Up (Clean verified clips and list file) ---
        files_to_cleanup_finally.extend(verified_clips_for_cleanup) # Add verified clips to cleanup list
        self._cleanup_temp_files(files_to_cleanup_finally, "final success")

        duration = time.monotonic() - start_time
        logger.info(f"--- Final Movie Assembly V5 Successful ({duration:.3f}s) ---")
        logger.info(f"Output file: {self.final_movie_path}")
        return self.final_movie_path


    def _cleanup_temp_files(self, files_to_delete: List[Path], stage_info: str ="cleanup"):
        """Cleans up specified files (list files, intermediate clips)."""
        if not files_to_delete: logger.debug(f"Cleanup ({stage_info}): No files specified."); return
        logger.info(f"Cleanup ({stage_info}): Attempting to delete {len(files_to_delete)} files...")
        deleted_count = 0
        for file_path in files_to_delete:
            if file_path is None: continue
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                try: file_path_obj.unlink(); deleted_count += 1; logger.debug(f"Cleanup ({stage_info}): Deleted {file_path_obj.name}")
                except OSError as e: logger.warning(f"Cleanup ({stage_info}): Failed delete {file_path_obj.name}: {e}")
            # Removed the warning for non-existent files during cleanup, as it can be noisy if cleanup runs multiple times
        logger.info(f"Cleanup ({stage_info}): Finished. Deleted {deleted_count}/{len(files_to_delete)} specified files.")


# --- Main Execution Block (Test Logic) ---
async def main_test():
    # Configure logging specifically for test
    test_logger = logging.getLogger(__name__)
    test_logger.setLevel(logging.DEBUG)
    if not test_logger.hasHandlers():
        log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        test_logger.addHandler(console_handler)
        test_logger.propagate = False

    print("--- Running FFmpegAssembler Test (Conceptual V5 - Robust Checks) ---")
    assembler = FFmpegAssembler(final_movie_path="test_output_v5.mp4")

    # --- Basic Command Checks ---
    print("\nTesting ffmpeg -version...")
    success, _, _ = await assembler._run_ffmpeg_async(["-version"], "FFmpegVersionTest")
    print(f"FFmpeg check: {'Success' if success else 'Failed'}")
    print("\nTesting ffprobe -version...")
    success_p, _, _ = await assembler._run_ffprobe_async(["-version"], "FFprobeVersionTest")
    print(f"FFprobe check: {'Success' if success_p else 'Failed'}")

    # --- Test _check_stream_details ---
    print("\nTesting _check_stream_details...")
    dummy_file_path = Path("dummy_check_v5.txt")
    try:
        dummy_file_path.touch()
        print(f"\nTesting details on dummy file ({dummy_file_path}):")
        has_audio, details_a = await assembler._check_stream_details(dummy_file_path, 'audio')
        print(f"  -> Has Audio: {has_audio} (Expected: False), Details: {details_a}")
        has_video, details_v = await assembler._check_stream_details(dummy_file_path, 'video')
        print(f"  -> Has Video: {has_video} (Expected: False), Details: {details_v}")
    except Exception as e: print(f"Error: {e}")
    finally:
         if dummy_file_path.exists(): dummy_file_path.unlink()

    # --- Test on actual media if available ---
    silent_vid_path = Path("silent_video.mp4") # Create a short, silent mp4 for testing if needed
    if silent_vid_path.exists():
        print(f"\nTesting details on '{silent_vid_path}' (EXPECTED: Video=True, Audio=False):")
        try:
             has_vid, details_vid = await assembler._check_stream_details(silent_vid_path, 'video')
             print(f"  -> Has Video: {has_vid}, Details: {details_vid}")
             has_aud, details_aud = await assembler._check_stream_details(silent_vid_path, 'audio')
             print(f"  -> Has Audio: {has_aud}, Details: {details_aud}")
        except Exception as e: print(f"Error testing {silent_vid_path}: {e}")
    else: print(f"\nSkipping details check on '{silent_vid_path}' - file not found.")

    example_audio_path = Path("temp_files/audio/audio_ff5bd87305.mp3") # Example from previous runs
    if example_audio_path.exists():
        print(f"\nTesting details on '{example_audio_path.name}' (EXPECTED: Audio=True, Video=False):")
        try:
             has_aud, details_aud = await assembler._check_stream_details(example_audio_path, 'audio')
             print(f"  -> Has Audio: {has_aud}, Details: {details_aud}")
             has_vid, details_vid = await assembler._check_stream_details(example_audio_path, 'video')
             print(f"  -> Has Video: {has_vid}, Details: {details_vid}")
        except Exception as e: print(f"Error testing {example_audio_path.name}: {e}")
    else: print(f"\nSkipping details check on '{example_audio_path.name}' - file not found.")


    print("\nNOTE: Full `assemble_movie` test requires a populated cache from storyshot_.py.")
    print("\n--- End of ffmpeg_assembler_.py Test V5 ---")


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
       asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_test())
    except KeyboardInterrupt:
        print("\nTest interrupted.")