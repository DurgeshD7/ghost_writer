import os
import whisper

class AudioTranscriber:
    def __init__(self, model_size="medium"):
        """
        Initialize the transcriber with specified model size.
        
        Args:
            model_size (str): Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully!")

    def transcribe(self, audio_file_path):
        """
        Transcribes the given audio file using OpenAI's Whisper model.
        """
        # Check if the file exists
        if not os.path.isfile(audio_file_path):
            print(f"Error: Audio file not found at path: {audio_file_path}")
            return None

        try:
            # Perform transcription
            print(f"Processing file: {audio_file_path}")
            result = self.model.transcribe(audio_file_path, language="en")
            transcription = result['text']
            print("Transcription complete!")
            return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

# Helper function for direct use
def transcribe_audio_file(audio_file_path, model_size="medium"):
    """
    Helper function to quickly transcribe an audio file without manually creating a transcriber instance.
    """
    transcriber = AudioTranscriber(model_size)
    return transcriber.transcribe(audio_file_path)

if __name__ == "__main__":
    # Example usage
    audio_file = r"Audio File Path here"  # Replace with your audio file path
    
    # Using the helper function
    transcription = transcribe_audio_file(audio_file)
    if transcription:
        print("\nFinal Transcription:")
        print(transcription)
    else:
        print("Transcription failed. Please check the file and try again.")
