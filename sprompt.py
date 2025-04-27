# sprompt.py
# Monitors project files and concatenates them into project_superprompt_output.md
# Useful for feeding project context to an LLM.

import os
import time
import glob
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent, FileMovedEvent

# --- Configuration ---
FILES_TO_WATCH = ['./project_superprompt.md'] + sorted(glob.glob('./*_.py'))
OUTPUT_FILENAME = 'project_superprompt_output.md'
WATCH_DIRECTORY = '.' # Watch the current directory
DEBOUNCE_TIME = 0.5 # Seconds to wait after a change before updating

# --- Globals ---
last_update_time = 0
update_scheduled = False

# --- Core Function ---
def update_superprompt():
    """Reads watched files and writes concatenated content to the output file."""
    global last_update_time, update_scheduled

    print(f"--- Updating {OUTPUT_FILENAME} at {time.strftime('%H:%M:%S')} ---")
    all_content = []
    files_processed = []

    # Define the order - project_superprompt.md first, then python scripts
    ordered_files = ['./project_superprompt.md'] + sorted(glob.glob('./*_.py'))

    for filepath in ordered_files:
        filename = os.path.basename(filepath)
        files_processed.append(filename)
        separator = f"\n\n{'='*10} File: {filename} {'='*10}\n\n"
        all_content.append(separator)
        try:
            # Attempt to read with UTF-8, fallback to latin-1 if needed
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_content.append(f.read())
            except UnicodeDecodeError:
                print(f"Warning: UTF-8 decode failed for {filename}, trying latin-1.")
                with open(filepath, 'r', encoding='latin-1') as f:
                    all_content.append(f.read())
        except FileNotFoundError:
            print(f"Warning: File not found during concatenation: {filename}")
            all_content.append(f"[File {filename} not found at time of update]\n")
        except IOError as e:
            print(f"Error reading file {filename}: {e}")
            all_content.append(f"[Error reading file {filename}]\n")

    # Check if any _*.py files were missed by initial glob (e.g. created later)
    current_py_files = sorted(glob.glob('./*_.py'))
    newly_added = [f for f in current_py_files if os.path.basename(f) not in [os.path.basename(p) for p in ordered_files]]
    # This check might be redundant if watchdog handles creation well, but good fallback. Needs testing.


    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            outfile.write("".join(all_content))
        print(f"Successfully updated {OUTPUT_FILENAME} including: {', '.join(files_processed)}")
    except IOError as e:
        print(f"Error writing output file {OUTPUT_FILENAME}: {e}")

    last_update_time = time.time()
    update_scheduled = False

def schedule_update(delay=DEBOUNCE_TIME):
    """Schedules an update if one isn't already pending."""
    global update_scheduled
    if not update_scheduled:
        update_scheduled = True
        # In a real async setup you'd use loop.call_later, here we just rely on event timing
        # The check at the start of update_superprompt handles basic debouncing
        # For simplicity in sync watchdog, we'll just let the next event trigger it
        # or call it directly if needed, the internal check will manage rate.
        # Let's trigger it directly after a small delay handled by the event handler itself.
        # No need for threading.Timer here if we handle delay in the handler.
        pass # Debounce logic is handled inside update_superprompt activation

# --- Watchdog Event Handler ---
class SuperpromptEventHandler(FileSystemEventHandler):
    """Handles file system events and triggers the superprompt update."""
    def __init__(self):
        super().__init__()
        self.last_event_time = 0
        self.files_to_watch_basenames = {os.path.basename(f) for f in FILES_TO_WATCH} # Cache basenames
        print(f"DEBUG: Watching basenames: {self.files_to_watch_basenames}") # DEBUG

    def on_any_event(self, event):
        print(f"\nDEBUG: ======= Event Received =======") # Separator
        print(f"DEBUG: Event object: {event}") # Print the whole event object
        print(f"DEBUG: Event type: {event.event_type}")
        print(f"DEBUG: Event src_path: {getattr(event, 'src_path', 'N/A')}") # Use getattr for safety
        print(f"DEBUG: Event dest_path: {getattr(event, 'dest_path', 'N/A')}") # Use getattr for safety
        print(f"DEBUG: Is directory: {event.is_directory}")

        if event.is_directory:
            print("DEBUG: Ignoring directory event.")
            print(f"DEBUG: ======= Event Handling End =======\n")
            return

        # Determine the relevant filesystem path associated with the event
        path_to_check = None
        if isinstance(event, FileMovedEvent):
            # For moves, the destination name is what we care about matching
            path_to_check = event.dest_path
            print(f"DEBUG: Event is MOVE, using dest_path for check: {path_to_check}")
        elif hasattr(event, 'src_path'):
            # For create, delete, modify, use the source path
            path_to_check = event.src_path
            print(f"DEBUG: Event is {event.event_type}, using src_path for check: {path_to_check}")
        else:
             print("DEBUG: Event has no src_path or dest_path, cannot determine relevant path.")
             print(f"DEBUG: ======= Event Handling End =======\n")
             return # Cannot proceed

        # Extract the base filename (e.g., "my_script_.py") from the full path
        # os.path.basename handles './filename' correctly -> 'filename'
        relevant_basename = os.path.basename(path_to_check)
        print(f"DEBUG: Calculated relevant_basename: '{relevant_basename}' (Type: {type(relevant_basename)})")

        # Get the set of filenames we are monitoring (ensure it's up-to-date if files can be added)
        # Note: self.files_to_watch_basenames is set in __init__ and not updated dynamically here
        watched_set = self.files_to_watch_basenames
        print(f"DEBUG: Comparing against watched set: {watched_set}")

        betroffen = False
        # Check if the extracted basename is in our set of watched files
        if relevant_basename in watched_set:
             betroffen = True
             print(f"DEBUG: *** Match FOUND: '{relevant_basename}' is in the watched set.")
        else:
             # Explain why it didn't match
             print(f"DEBUG: *** Match FAILED: '{relevant_basename}' is NOT in the watched set {watched_set}.")
             # Check for common issues:
             if f"./{relevant_basename}" in watched_set:
                 print("DEBUG: Hint: Found a match if './' prefix was present in watched_set?")
             if relevant_basename.strip() in watched_set:
                 print("DEBUG: Hint: Found a match if whitespace was stripped?")

        # Handle newly created *.py files (optional, but good to consider)
        # Note: This logic doesn't add the file to FILES_TO_WATCH for future concatenation runs
        # unless update_superprompt() re-globs every time.
        if not betroffen and event.event_type == 'created' and relevant_basename.endswith('_.py'):
             betroffen = True
             print(f"DEBUG: Matched based on 'created' event for a new '_*.py' file: '{relevant_basename}'")


        # Proceed only if the file matched ('betroffen' is True)
        if betroffen:
            print(f"DEBUG: 'betroffen' is TRUE. Proceeding to debounce check.")
            current_time = time.time()
            time_since_last = current_time - self.last_event_time
            print(f"DEBUG: Current time: {current_time:.2f}, Last trigger time: {self.last_event_time:.2f}, Diff: {time_since_last:.2f}s, Debounce threshold: {DEBOUNCE_TIME}s")

            # Debounce logic: Ensure enough time has passed since the last update trigger
            if time_since_last > DEBOUNCE_TIME:
                print(f"DEBUG: Debounce PASSED ({time_since_last:.2f}s > {DEBOUNCE_TIME}s). Calling update_superprompt() !!!")
                update_superprompt() # Call the update function
                self.last_event_time = time.time() # IMPORTANT: Reset timer *after* triggering the update
                print(f"DEBUG: Updated last_event_time to: {self.last_event_time:.2f}")
            else:
                print(f"DEBUG: Debounce FAILED ({time_since_last:.2f}s <= {DEBOUNCE_TIME}s). Event too soon, skipping update call.")
        else:
             print(f"DEBUG: 'betroffen' is FALSE. No update triggered for this event.")

        print(f"DEBUG: ======= Event Handling End =======\n")



# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Superprompt Monitor...")
    print(f"Watching directory: {os.path.abspath(WATCH_DIRECTORY)}")
    print(f"Target files pattern: project_superprompt.md, *_*.py")
    print(f"Output file: {OUTPUT_FILENAME}")
    print("Press Ctrl+C to stop.")

    # Initial generation
    print("Performing initial superprompt generation...")
    update_superprompt()

    # Setup watchdog observer
    event_handler = SuperpromptEventHandler()
    # observer = Observer() # <-- Comment out or delete old line
    observer = PollingObserver() # <--- USE POLLING OBSERVER
    observer.schedule(event_handler, path=WATCH_DIRECTORY, recursive=False) # Don't watch subdirectories

    observer.start()

    try:
        while True:
            # Keep the main thread alive. Observer runs in background thread.
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Superprompt Monitor...")
        observer.stop()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        observer.stop()

    observer.join() # Wait for observer thread to finish
    print("Superprompt Monitor stopped.")

