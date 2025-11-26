THE “PHYSICS TEST” INSTRUCTIONS

Silence → Audio Buffer Extraction → Whisper → < 1.5s total latency

This is the Gatekeeper Test.
If this fails, the product is impossible on CPU.
If this works, the rest of the MVP becomes a matter of assembly.

Below is the exact checklist in the order you MUST implement it.

✅ PHASE 1 — Setup the Audio Stream (Must Be Exact)
1. Use sounddevice input stream

Configure it like this:

samplerate = 16000

channels = 1

dtype = 'float32'

blocksize = 320 (this gives 20ms frames)

2. Build a Rolling Ring Buffer

Create an array that holds:

1.2 seconds of audio = 16000 * 1.2 = 19200 samples
19200 samples / 320 per frame = 60 frames


So the buffer needs to hold 60 frames.

Every new frame:

Append frame

If buffer length > 60 → drop the oldest

This ensures the buffer always contains the last 1.2 seconds.

✅ PHASE 2 — Add Silence Detection Using Silero VAD

Load Silero early in the file:

model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


Define thresholds:

Speech threshold = 0.5

Silence threshold = 0.3

Silence duration trigger = 0.6 seconds
→ 0.6 / 0.02 per frame = 30 frames

Create variables:

silence_counter = 0
speaking = False

In the audio callback:

Infer VAD probability:

speech_prob = model(torch.from_numpy(audio_chunk), 16000).item()


Logic:

if speech_prob > 0.5:
    speaking = True
    silence_counter = 0
else:
    if speaking:
        silence_counter += 1


Trigger conditions:

If:

User was speaking

Silence for ≥ 30 frames → TRIGGER

if speaking and silence_counter >= 30:
    trigger_event = True
    speaking = False
    silence_counter = 0


This gives you the moment when the user finishes a thought.

✅ PHASE 3 — Extract the 1.2s Audio Window

When trigger_event == True:

Concatenate the 60 frames:

audio_data = np.concatenate(ring_buffer, axis=0)


Convert float32 PCM → int16 (Whisper likes this):

audio_int16 = (audio_data * 32767).astype(np.int16)


Store in a variable for Whisper.

✅ PHASE 4 — Run Whisper Immediately (Blocking)

Load Whisper BEFORE starting audio stream:

stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")


When silence trigger fires:

start_time = time.time()

segments, _ = stt.transcribe(audio_int16, beam_size=1)

text = " ".join([s.text for s in segments])


Measure latency:

stt_latency = time.time() - start_time
print("STT Latency:", stt_latency)

Your target:

Average: 0.9–1.2s

Worst-case: <1.4s

NEVER >1.5s

If you're above these numbers → the MVP is in danger.

✅ PHASE 5 — Print Total Latency

Start timing the moment silence is detected:

global_start = time.time()


At the end:

print("TOTAL LATENCY:", time.time() - global_start)


Goal:

95% < 1.5 seconds

99% < 1.7 seconds

Max < 1.9 seconds

If this holds under Zoom + Chrome → You passed the Physics Test.

🧪 Stress Test (Mandatory)

Run this script with:

Zoom running

Background blur ON

Chrome with 10+ tabs

Slack

Screen recording ON

If performance collapses → Whisper needs optimization.

💥 YOUR DEVELOPMENT ORDER (DO NOT CHANGE)
1. Audio stream with 20ms frames
2. Ring buffer (60 frames)
3. Silero VAD simple silence detection
4. Trigger event
5. Extract 1.2 seconds
6. Whisper tiny.en (int8)
7. Measure full latency
8. Repeat 20 times

When this is stable →
You unlock the rest of the MVP.

If this is NOT stable →
Stop. Fix it before doing anything else.