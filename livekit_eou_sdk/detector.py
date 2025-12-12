"""
Arabic EOU Detector - Core detection class
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

class ArabicEOUDetector:
    """
    End-of-Utterance detector for Arabic conversational speech
    """
    
    def __init__(self, model_path="model/final_model", threshold=0.5, device=None):
        """
        Initialize the EOU detector
        
        Args:
            model_path: Path to fine-tuned model or HuggingFace model name
            threshold: Probability threshold for EOU detection (0-1)
            device: torch device ('cpu' or 'cuda')
        """
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Arabic EOU Detector...")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
    
    def predict(self, text):
        """
        Predict EOU probability for given text
        
        Args:
            text: Arabic text string
            
        Returns:
            float: EOU probability (0-1), higher means more likely to be end of utterance
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities for EOU class (label=1)
        probs = torch.softmax(logits, dim=-1)
        eou_probs = probs[:, :, 1].cpu().numpy()[0]
        
        # Take maximum EOU probability across all tokens
        # (if any token has high EOU probability, utterance might be ending)
        max_eou_prob = float(np.max(eou_probs))
        
        # Alternative: use last token's probability
        last_token_prob = float(eou_probs[-1])
        
        # Use weighted combination
        eou_score = 0.7 * last_token_prob + 0.3 * max_eou_prob
        
        return eou_score
    
    def is_eou(self, text):
        """
        Check if text represents end of utterance
        
        Args:
            text: Arabic text string
            
        Returns:
            bool: True if EOU detected, False otherwise
        """
        prob = self.predict(text)
        return prob >= self.threshold
    
    def predict_streaming(self, text_buffer):
        """
        Predict EOU for streaming text (accumulating buffer)
        
        Args:
            text_buffer: Current accumulated text from STT
            
        Returns:
            dict: {
                'eou_probability': float,
                'is_eou': bool,
                'confidence': str ('low'|'medium'|'high')
            }
        """
        prob = self.predict(text_buffer)
        is_eou = prob >= self.threshold
        
        # Determine confidence
        if prob < 0.3:
            confidence = 'low'
        elif prob < 0.7:
            confidence = 'medium'
        else:
            confidence = 'high'
        
        return {
            'eou_probability': prob,
            'is_eou': is_eou,
            'confidence': confidence,
            'text_length': len(text_buffer.split())
        }


if __name__ == "__main__":
    # Test the detector
    print("=" * 80)
    print("Testing Arabic EOU Detector")
    print("=" * 80)
    
    try:
        detector = ArabicEOUDetector()
        
        # Test cases
        test_texts = [
            "مرحبا",  # Hello (incomplete)
            "مرحبا كيف حالك؟",  # Hello how are you? (complete question)
            "أنا بخير",  # I'm fine (incomplete)
            "أنا بخير الحمد لله.",  # I'm fine thank God. (complete)
            "ممكن تساعدني",  # Can you help me (incomplete)
            "ممكن تساعدني في هذا الموضوع؟",  # Can you help me with this? (complete)
        ]
        
        print("\nTest Results:")
        print("-" * 80)
        for text in test_texts:
            result = detector.predict_streaming(text)
            print(f"\nText: {text}")
            print(f"  EOU Probability: {result['eou_probability']:.3f}")
            print(f"  Is EOU: {result['is_eou']}")
            print(f"  Confidence: {result['confidence']}")
    
    except Exception as e:
        print(f"\n⚠ Cannot test: {e}")
        print("Model needs to be trained first. Run model/train.py")
