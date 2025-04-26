text only storyboard app called storyshot
aiming towards an AI transcript2short_movie_story app
heavily leaverages LLMs, openai's brilliant gpt-4o image generator (gpt-image-1, unique for it's very advanced textual understanding/prompt adherence),  image2video generators (primarily kling 1.6 through fal.ai), and text2speech model (through fal.ai).

# storyshot_.py
backend workflow is as follows
transcript is broken into sentences via nltk
each sentence is used as a key in an ordered dict to keep track of speech/image/video caching on future runs (might need to collect additional information in each sentence dict element, save all new files (video/audio/image, ect..) in a persistent temp folder) and save the link to the files in the dict
normalize the sentence text (lowercase, remove leading/trailing whitespace) before using it as a key
the dict should be saved in a json text file in the same dir. Always load the dict before each run
cached dict data is stored in `\cache_data\` and larger files (img/video/audio) in `\temp_files\` 
Dependency Management: `storyshot_.py` will orchestrate calls to the other modules. Ensure clear function interfaces.

# text2speech_.py
each (new) sentence of the transcript then gets text2speech applied to it (this operation can be done async). Include output info in each relevant dict (specifically note audio length b/c that will determine the number of images we'll need to generate for each sentence). Manage sentence->audio caching

# openrouter_engine_.py
In order to call LLMs from openrouter(useful for it's very broad selection of constantly updated LLMs), we need to use the openrouter_engine.py file. This script will handle the API calls to openrouter and return the results, as well as track the cost and time of each request. The script class should on startup save a list of all the LLMs it can call. 

# sentence2storyboard_.py
so now we have the transcript and audio for the entire transcript, next we need to storyboard the images for each sentence. We'll do this by calling an LLM via openrouter_engine_.py to write an image prompt describing a still image for every 2 seconds in each sentence(/speech audio clip). 2 seconds is chosen b/c it's about the average shot length in movies AND it gives little time for the video to warp strangely. Let the last shot soak up the remainder of the seconds (i.e. the last shot should always be between 2-3.99 seconds long). Prompts are saved in an ordered dict (inside of the original sentence dict element). This block is also done async, remember to manage sentence->prompt caching. 
Visual Cohesion: Getting the LLM to generate _sequences_ of prompts that result in visually coherent shots for a single sentence is non-trivial. Prompt engineering will be key. You might need to provide the LLM with the previous prompt(s) as context. We'll worry about total coherence once the mvp is built

# prompt2image_.py
next we'll generate images for each (new) prompt for each sentence via gpt-image-1. Make async and manage sentence->prompt->image caching as, this operation is very long (1-2 minutes) and expensive (~$0.20 per image)

# image2video_.py
Next we'll generate 2 second videos for each image. To keep the memory pattern the same, save links/data in the sentence->prompt->image->video dict. This is also a very long and expensive async operation


# (back in storyshot_.py)
for each sentence concatenate all the video shots and *then* put the sentence audio over it (shortening to whichever is shortest, usually the audio). I'm a bit fuzzy as what exactly how to do this, though I think we use ffmpeg 

# project superprompt concatenation server (sprompt.py)
python watchdog server that simply appends all files ending in "\_.py" to each other and the project_superprompt.md file whenever a change is detected in any relevant file. This way prompting llms should be simple as copying the contents of project_superprompt_output.md 

# logging system
log api errors. Build in api retries for each api (set to 1 initially)

# .env variables
OPENAI_KEY=
FAL_KEY=
OPENROUTER_KEY=
FALAI_TTS_MODEL="fal-ai/playai/tts/v3" 
OPENAI_IMAGE_MODEL="gpt-image-1"  
FALAI_VIDEO_MODEL="fal-ai/wan-i2v" 

# cost and time tracking
Implement the cost and operation time tracking metrics found in text2speech_.py (well, you know what I mean. make sure the dict tracks/records these metrics both by request, parallel operation, and entire run). This data will be used to make design decisions about the app's performance and cost. When tracking time across concurrent operations, NEVER sum the times of each operation, instead actually measure the time from start to finish of the entire operation (which is often the same as the longest operation). The wall-clock time. NEVER Sum concurrent Durations(why would you ever, it's kinda useless except to find an avg)!!!

my software design choices are as follows. Modular, class-based, script-first testing, pragmatic approach. I care more about utility, functionality, and simplicity than "correctness". The app is cmd line based. python scripts each are (usually) a single script class with a if __main__.py section that handles individual script testing (not really unit tests, just a simple place to iterate on until the core functions work as expected). Vibe coding never runs perfectly the first time :). When we discover something unintuitive about the api structure or anything else, always add the insight somewhere relevant inside the top of the script in a giant block info comment(for example this block comment will include api info on fal.ai, openai, ect...). Add a bool to each script to handle cases when you _want_ to regenerate something even if it's cached. Write code async ready but *structure everything sync* as all the testing and foreseeable runs will be sync only, and caching should handle speed issues while testing. Use .env config file for API keys, model names, default paths. For now hardcode llm prompts near where they are used in triple curly braced strings. Use requirements.txt to manage Python dependencies. Development is done using wsl 2 vscode on windows 10, so always double check for potential file access and permissions issues, default to reliability over speed. 
