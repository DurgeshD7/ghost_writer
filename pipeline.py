from speech_to_text import AudioTranscriber
from context_analysis import StoryContextAnalyzer
import os
from datetime import datetime

class StorywritingPipeline:
    def __init__(self):
        """Initialize the pipeline components"""
        print("Initializing Story Writing Assistant...")
        print("Loading transcriber and language models...")
        self.transcriber = AudioTranscriber(model_size="medium")
        # Initialize StoryContextAnalyzer without model_size parameter
        self.context_analyzer = StoryContextAnalyzer()
        print("Pipeline ready!")

    def save_output(self, results: dict, output_dir: str = "story_outputs"):
        """Save the pipeline results to a file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"story_continuation_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("==== Original Transcription ====\n\n")
            f.write(results['transcription'])
            f.write("\n\n")
            
            f.write("==== Generated Suggestions ====\n\n")
            for i, suggestion in enumerate(results['continuations'], 1):
                f.write(f"Suggestion {i}:\n")
                f.write("-" * 50 + "\n")
                f.write(suggestion)
                f.write("\n" + "-" * 50 + "\n\n")
        
        return filepath

    def process_audio(self, audio_file_path: str, min_suggestions: int = 3):
        """Process audio file through the pipeline"""
        try:
            # Step 1: Transcribe audio
            print("\n1. Transcribing audio...")
            transcription = self.transcriber.transcribe(audio_file_path)
            
            if not transcription:
                print("Error: Audio transcription failed")
                return None
                
            print("\nTranscribed Text:")
            print("-" * 50)
            print(transcription)
            print("-" * 50)

            # Step 2: Analyze context
            print("\n2. Analyzing context...")
            context = self.context_analyzer.analyze_context(transcription)
            
            # Step 3: Generate continuations
            print("\n3. Generating story continuations...")
            continuations = []
            while len(continuations) < min_suggestions:
                new_continuations = self.context_analyzer.generate_continuations(
                    transcription,
                    num_suggestions=min_suggestions - len(continuations)
                )
                continuations.extend(new_continuations)
            
            print("\nGenerated Suggestions:")
            for i, continuation in enumerate(continuations, 1):
                print(f"\nSuggestion {i}:")
                print("-" * 50)
                print(continuation)
                print("-" * 50)

            results = {
                'transcription': transcription,
                'context': context,
                'continuations': continuations
            }

            output_file = self.save_output(results)
            print(f"\nResults saved to: {output_file}")

            return results

        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return None

def main():
    pipeline = StorywritingPipeline()
    
    audio_file = input("Enter the path to your audio file: ").strip()
    
    if not os.path.exists(audio_file):
        print(f"Error: File not found at {audio_file}")
        return
    
    results = pipeline.process_audio(audio_file, min_suggestions=3)
    
    if results:
        print("\nProcessing complete! Check the story_outputs directory for your results.")

if __name__ == "__main__":
    main()