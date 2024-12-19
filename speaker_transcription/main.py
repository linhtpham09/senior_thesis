from predict import Predictor 
import pickle 
import json 
import tempfile
from cog import Path
# Save the setup state
def save_setup_state(predictor, filename="predictor_state.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(predictor, f)

# Load the setup state
def load_setup_state(filename="predictor_state.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    try:
        # Load existing predictor state
        predictor = load_setup_state()
        print("Loaded predictor state from file.")
    except FileNotFoundError:
        # Run setup if state doesn't exist
        predictor = Predictor()
        predictor.setup()
        save_setup_state(predictor)
        print("Saved predictor state to file.")

     
    predictor.audio_pre.process('sao_mai.m4a')
    print('audio')
    
    if predictor.audio_pre.error: 
        print(predictor.audio_pre.error)
        result = predictor.diarization_post.empty_result()
    else: 
        result = predictor.run_diarization()
    print('diarization ran')
    
    #run transcription 
    predictor.run_transcription(predictor.audio_pre.output_path, result['segments'], whisper_prompt=None)
    
        # format segments
    result["segments"] = predictor.diarization_post.format_segments(
        result["segments"])
    
    predictor.audio_pre.cleanup()
    
    output = Path(tempfile.mkdtemp()) / "output.json"
    print("Output", output)
    with open(output, "w") as f:
        f.write(json.dumps(result, indent=2))