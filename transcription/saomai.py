from parser import * 
args = parse_args()
import whisper
from pydub import AudioSegment
import tensorflow as tf
#help load pyannote faster 
from pyannote.audio import Pipeline

print(tf.config.list_physical_devices('GPU'))  # Should list available GPUs

print('test')
# Convert audio file to WAV
audio_path = "sao_mai.m4a"
audio = AudioSegment.from_file(audio_path)
wav_path = "sao_mai.wav"
audio.export(wav_path, format="wav")

# Step 1: Speaker Diarization

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=args.hf_token)
diarization = pipeline(wav_path, num_speakers = 2)


print('step1')
# Step 2: Transcription using Whisper
model = whisper.load_model("medium")
result = model.transcribe(wav_path, language="vi", word_timestamps = True)

print('step2')

# Step 3: Combine diarization and transcription
transcription = result["text"]
print("Full Transcription:")
print(transcription)

print('step3')

# Save speaker-separated output
# Save speaker-separated output
with open("speaker_transcription.txt", "w") as file:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end

        # Get words spoken in this segment
        words = [
            word["word"]
            for segment in result["segments"]
            for word in segment["words"]
            if start <= word["start"] < end
        ]

        # Write speaker and transcription to file
        segment_text = f"{start:.2f}-{end:.2f}: Speaker {speaker}\n"
        segment_text += " ".join(words) + "\n\n"
        file.write(segment_text)

print('done')