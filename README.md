# ghost_writer - Your Writing Assistant

A comprehensive AI-powered tool that helps transform audio narratives into compelling story continuations. This pipeline integrates OpenAI's Whisper model for speech-to-text conversion and GPT-2 for generating creative story suggestions. Additionally, it analyzes the narrative context to enhance the generated content.

## Features

1. **Speech-to-Text Transcription**:
   - Utilizes OpenAI's Whisper model for accurate transcription of audio files.
   - Supports multiple model sizes (`tiny`, `base`, `small`, `medium`, `large`).

2. **Context Analysis**:
   - Analyzes transcribed text for named entities, emotional tone, and key actions.
   - Leverages spaCy for entity recognition and DistilBERT for sentiment analysis.

3. **Story Continuation Generation**:
   - Generates creative continuations for stories using GPT-2.
   - Provides multiple suggestions to inspire writers.

4. **Pipeline Integration**:
   - Combines transcription and context analysis seamlessly.
   - Saves results, including transcription and generated suggestions, to a text file.

## Project Structure

### `speech_to_text.py`
Handles audio transcription using OpenAI's Whisper model.

- **Classes**: `AudioTranscriber`
- **Functions**:
  - `transcribe(audio_file_path)`: Transcribes an audio file.
  - `transcribe_audio_file(audio_file_path, model_size)`: Helper function for quick transcription.

### `context_analysis.py`
Analyzes the context of the text and generates story continuations using GPT-2.

- **Classes**: `StoryContextAnalyzer`
- **Functions**:
  - `analyze_context(text)`: Extracts entities, emotional tone, and actions.
  - `generate_continuations(context, num_suggestions, max_length)`: Generates creative story continuations.

### `pipeline.py`
Integrates the transcription and context analysis components into a cohesive pipeline.

- **Classes**: `StorywritingPipeline`
- **Functions**:
  - `process_audio(audio_file_path, min_suggestions)`: Processes an audio file and generates story suggestions.
  - `save_output(results, output_dir)`: Saves transcription and suggestions to a file.

## Installation

Ensure you have Python 3.8 or higher installed. Install the required packages:

```bash
pip install torch transformers spacy openai-whisper
python -m spacy download en_core_web_sm
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Run the `pipeline.py` script:

   ```bash
   python pipeline.py
   ```

3. Provide the path to your audio file when prompted.
4. Results, including the transcription and generated story suggestions, will be saved in the `story_outputs` directory.

## Requirements

- Python 3.8+
- Packages:
  - `torch`
  - `transformers`
  - `spacy`
  - `openai-whisper`
  - `numpy`

## Example Output

1. **Original Transcription**:

   ```
   Sarah's hands trembled as she inserted the key into the rusty lock. The old mansion had been abandoned for twenty years, but tonight, something felt different. The air was thick with an unexplainable tension.
   ```

2. **Generated Suggestions**:

   **Suggestion 1**:
   ```
   As the door creaked open, a cold draft swept over Sarah, carrying with it the faint scent of roses. She stepped inside, her footsteps echoing in the vast emptiness, and froze when she saw a flicker of light down the hallway.
   ```

   **Suggestion 2**:
   ```
   The lock clicked, and the door gave way. Sarah hesitated, peering into the darkness. Suddenly, a voice whispered her name from within, sending a chill down her spine.
   ```

## Contributions

Feel free to fork this repository and submit pull requests. Suggestions and feature requests are welcome!

## License

This project is open-source and available under the MIT License.

---

Happy story writing!

