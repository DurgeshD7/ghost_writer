import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, pipeline
)
from typing import List, Dict
import spacy
import numpy as np

class StoryContextAnalyzer:
    def __init__(self):
        """Initialize the context analyzer with required models"""
        # Load spaCy for NER and basic text analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load sentiment analyzer for emotional context
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Initialize GPT-2 for story continuation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        self.model.eval()
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def analyze_context(self, text: str) -> Dict:
        """
        Analyze the provided text to extract contextual elements.
        """
        doc = self.nlp(text)
        
        # Extract named entities (characters, locations, etc.)
        entities = {
            'characters': [],
            'locations': [],
            'time_expressions': [],
            'misc': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON']:
                entities['characters'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ in ['TIME', 'DATE']:
                entities['time_expressions'].append(ent.text)
            else:
                entities['misc'].append((ent.text, ent.label_))
        
        # Analyze emotional context
        sentiment_result = self.sentiment_analyzer(text)[0]
        
        # Extract key verbs (actions) and their subjects
        actions = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = child.text
                        break
                if subject:
                    actions.append({
                        'subject': subject,
                        'action': token.text,
                        'sentence_id': token.sent.start
                    })
        
        return {
            'entities': entities,
            'sentiment': sentiment_result,
            'actions': actions,
            'last_sentence': list(doc.sents)[-1].text
        }

    def generate_continuations(self, 
                             context: str, 
                             num_suggestions: int = 3,
                             max_length: int = 200) -> List[str]:
        """
        Generate multiple possible story continuations based on the context.
        """
        inputs = self.tokenizer.encode(context, return_tensors='pt', padding=True)
        
        all_continuations = []
        
        for _ in range(num_suggestions):
            # Generate with some randomness for creativity
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.85,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
            continuation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the new content after the input context
            new_content = continuation[len(context):].strip()
            all_continuations.append(new_content)
            
        return all_continuations

def main():
    # Example usage
    analyzer = StoryContextAnalyzer()
    
    # Sample story fragment
    story_fragment = """
    Sarah's hands trembled as she inserted the key into the rusty lock. 
    The old mansion had been abandoned for twenty years, but tonight, 
    something felt different. The air was thick with an unexplainable tension.
    """
    
    # Analyze context
    context = analyzer.analyze_context(story_fragment)
    print("\nContext Analysis:")
    print(f"Characters: {context['entities']['characters']}")
    print(f"Locations: {context['entities']['locations']}")
    print(f"Emotional tone: {context['sentiment']}")
    
    # Generate continuations
    print("\nPossible Continuations:")
    continuations = analyzer.generate_continuations(story_fragment)
    for i, cont in enumerate(continuations, 1):
        print(f"\n{i}. {cont}")

if __name__ == "__main__":
    main()