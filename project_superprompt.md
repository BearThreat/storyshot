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
Implement the cost and operation time tracking metrics found in text2speech_.py (well, you know what I mean. make sure the dict tracks/records these metrics both by request, parallel operation, and entire run). This data will be used to make design decisions about the app's performance and cost. When tracking time across concurrent operations, ALWAYS track the wall-clock time. 

# input and output tracking
log all input and output data. This data will be used to make design decisions about the app's performance and cost as well as potential caching opportunities. Do not truncate the data(espeshially prompts), just log it. If the data is too large, save as a file or unsuitable for terminal display(images,video,audio,ect..) then link to it in the log.

# fal-ai api info
ALWAYS use REST instead of their python fal-client library
Queue
For requests that take longer than several seconds, as it is usually the case with AI applications, we have built a queue system.

Utilizing our queue system offers you a more granulated control to handle unexpected surges in traffic. It further provides you with the capability to cancel requests if needed and grants you the observability to monitor your current position within the queue. Besides that using the queue system spares you from the headache of keeping around long running https requests.

Queue endpoints
You can interact with all queue features through a set of endpoints added to you function URL via the queue subdomain. The endpoints are as follows:

Endpoint	Method	Description
queue.fal.run/{appId}	POST	Adds a request to the queue
queue.fal.run/{appId}/requests/{request_id}/status	GET	Gets the status of a request
queue.fal.run/{appId}/requests/{request_id}/status/stream	GET	Streams the status of a request until it’s completed
queue.fal.run/{appId}/requests/{request_id}	GET	Gets the response of a request
queue.fal.run/{appId}/requests/{request_id}/cancel	PUT	Cancels a request
For instance, should you want to use the curl command to submit a request to the aforementioned endpoint and add it to the queue, your command would appear as follows:

Terminal window
curl -X POST https://queue.fal.run/fal-ai/fast-sdxl \
  -H "Authorization: Key $FAL_KEY" \
  -d '{"prompt": "a cat"}'

Here’s an example of a response with the request_id:

{
  "request_id": "80e732af-660e-45cd-bd63-580e4f2a94cc",
  "response_url": "https://queue.fal.run/fal-ai/fast-sdxl/requests/80e732af-660e-45cd-bd63-580e4f2a94cc",
  "status_url": "https://queue.fal.run/fal-ai/fast-sdxl/requests/80e732af-660e-45cd-bd63-580e4f2a94cc/status",
  "cancel_url": "https://queue.fal.run/fal-ai/fast-sdxl/requests/80e732af-660e-45cd-bd63-580e4f2a94cc/cancel"
}

The payload helps you to keep track of your request with the request_id, and provides you with the necessary information to get the status of your request, cancel it or get the response once it’s ready, so you don’t have to build these endpoints yourself.

Request status
Once you have the request id you may use this request id to get the status of the request. This endpoint will give you information about your request’s status, it’s position in the queue or the response itself if the response is ready.

Terminal window
curl -X GET https://queue.fal.run/fal-ai/fast-sdxl/requests/{request_id}/status

Here’s an example of a response with the IN_QUEUE status:

{
  "status": "IN_QUEUE",
  "queue_position": 0,
  "response_url": "https://queue.fal.run/fal-ai/fast-sdxl/requests/80e732af-660e-45cd-bd63-580e4f2a94cc"
}

Status types
Queue status can have one of the following types and their respective properties:

IN_QUEUE:

queue_position: The current position of the task in the queue.
response_url: The URL where the response will be available once the task is processed.
IN_PROGRESS:

logs: An array of logs related to the request. Note that it needs to be enabled, as explained in the next section.
response_url: The URL where the response will be available.
COMPLETED:

logs: An array of logs related to the request. Note that it needs to be enabled, as explained in the next section.
response_url: The URL where the response is available.
Logs
Logs are disabled by default. In order to enable logs for your request, you need to send the logs=1 query parameter when getting the status of your request. For example:

Terminal window
curl -X GET https://queue.fal.run/fal-ai/fast-sdxl/requests/{request_id}/status?logs=1

When enabled, the logs attribute in the queue status contains an array of log entries, each represented by the RequestLog type. A RequestLog object has the following attributes:

message: a string containing the log message.
level: the severity of the log, it can be one of the following:
STDERR | STDOUT | ERROR | INFO | WARN | DEBUG
source: indicates the source of the log.
timestamp: a string representing the time when the log was generated.
These logs offer valuable insights into the status and progress of your queued tasks, facilitating effective monitoring and debugging.

Streaming status
If you want to keep track of the status of your request in real-time, you can use the streaming endpoint. The response is text/event-stream and each event is a JSON object with the status of the request exactly as the non-stream endpoint.

This endpoint will keep the connection open until the status of the request changes to COMPLETED.

It supports the same logs query parameter as the status.

Terminal window
curl -X GET https://queue.fal.run/fal-ai/fast-sdxl/requests/{request_id}/status/stream

Here is an example of a stream of status updates:

Terminal window
$ curl https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/status/stream?logs=1 --header "Authorization: Key $FAL_KEY"

data: {"status": "IN_PROGRESS", "request_id": "3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "response_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "status_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/status", "cancel_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/cancel", "logs": [], "metrics": {}}

data: {"status": "IN_PROGRESS", "request_id": "3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "response_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "status_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/status", "cancel_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/cancel", "logs": [{"timestamp": "2024-12-20T15:37:17.120314", "message": "INFO:TRYON:Preprocessing images...", "labels": {}}, {"timestamp": "2024-12-20T15:37:17.286519", "message": "INFO:TRYON:Running try-on model...", "labels": {}}], "metrics": {}}

data: {"status": "IN_PROGRESS", "request_id": "3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "response_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "status_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/status", "cancel_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/cancel", "logs": [], "metrics": {}}

: ping

data: {"status": "IN_PROGRESS", "request_id": "3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "response_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "status_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/status", "cancel_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/cancel", "logs": [], "metrics": {}}

data: {"status": "COMPLETED", "request_id": "3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "response_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf", "status_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/status", "cancel_url": "https://queue.fal.run/fashn/tryon/requests/3e3e5b55-45fb-4e5c-b4d1-05702dffc8bf/cancel", "logs": [{"timestamp": "2024-12-20T15:37:32.161184", "message": "INFO:TRYON:Finished running try-on model.", "labels": {}}], "metrics": {"inference_time": 17.795265674591064}}

Cancelling a request
If your request is still in the queue and not already being processed you may still cancel it.

Terminal window
curl -X PUT https://queue.fal.run/fal-ai/fast-sdxl/requests/{request_id}/cancel

Getting the response
Once you get the COMPLETED status, the response will be available along with its logs.

Terminal window
curl -X GET https://queue.fal.run/fal-ai/fast-sdxl/requests/{request_id}

Here’s an example of a response with the COMPLETED status:

{
  "status": "COMPLETED",
  "logs": [
    {
      "message": "2020-05-04 14:00:00.000000",
      "level": "INFO",
      "source": "stdout",
      "timestamp": "2020-05-04T14:00:00.000000Z"
    }
  ],
  "response": {
    "message": "Hello World!"
  }
}

my software design choices are as follows. Modular, class-based, script-first testing, pragmatic approach. I care more about utility, functionality, and simplicity than "correctness". The app is cmd line based. python scripts each are (usually) a single script class with a if __main__.py section that handles individual script testing (not really unit tests, just a simple place to iterate on until the core functions work as expected). Vibe coding never runs perfectly the first time :). When we discover something unintuitive about the api structure or anything else, always add the insight somewhere relevant inside the top of the script in a giant block info comment(for example this block comment will include api info on fal.ai, openai, ect...). Add a bool to each script to handle cases when you _want_ to regenerate something even if it's cached. Write code async ready but *structure everything sync* as all the testing and foreseeable runs will be sync only, and caching should handle speed issues while testing. Use .env config file for API keys, model names, default paths. For now hardcode llm prompts near where they are used in triple curly braced strings. Use requirements.txt to manage Python dependencies. Development is done using wsl 2 vscode on windows 10, so always double check for potential file access and permissions issues, default to reliability over speed. 
