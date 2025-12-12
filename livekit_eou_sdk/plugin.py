"""
LiveKit Plugin for Arabic EOU Detection
Integrates with LiveKit Agents framework
"""

from typing import Optional
from livekit import agents
from .detector import ArabicEOUDetector


class ArabicTurnDetectorModel:
    """
    LiveKit-compatible turn detector model for Arabic EOU detection
    """
    
    def __init__(
        self,
        model_path: str = "model/final_model",
        threshold: float = 0.5,
        min_confidence: float = 0.3
    ):
        """
        Initialize Arabic Turn Detector
        
        Args:
            model_path: Path to fine-tuned model
            threshold: EOU decision threshold
            min_confidence: Minimum confidence to report EOU
        """
        self.detector = ArabicEOUDetector(
            model_path=model_path,
            threshold=threshold
        )
        self.min_confidence = min_confidence
        self._text_buffer = ""
        
    def reset(self):
        """Reset the detector state"""
        self._text_buffer = ""
    
    def update(self, text: str) -> dict:
        """
        Update with new text from STT and check for EOU
        
        Args:
            text: Accumulated text from speech-to-text
            
        Returns:
            dict with EOU prediction info
        """
        self._text_buffer = text
        
        if not text or len(text.strip()) == 0:
            return {
                'is_eou': False,
                'probability': 0.0,
                'confidence': 'none'
            }
        
        result = self.detector.predict_streaming(text)
        
        # Only return True if confidence is above minimum
        if result['eou_probability'] < self.min_confidence:
            result['is_eou'] = False
        
        return result
    
    def predict_eou(self, text: str) -> float:
        """
        Predict EOU probability for given text
        
        Args:
            text: Text to analyze
            
        Returns:
            float: Probability of end-of-utterance (0-1)
        """
        return self.detector.predict(text)


class ArabicEOUPlugin:
    """
    LiveKit plugin wrapper for Arabic EOU detection
    Can be used in LiveKit Agent sessions
    """
    
    def __init__(
        self,
        model_path: str = "model/final_model",
        threshold: float = 0.5
    ):
        self.model = ArabicTurnDetectorModel(
            model_path=model_path,
            threshold=threshold
        )
    
    def get_model(self):
        """Get the turn detector model instance"""
        return self.model


# Example integration with LiveKit VoicePipelineAgent
def create_arabic_voice_agent(
    vad_plugin,
    stt_plugin,
    llm_plugin,
    tts_plugin,
    eou_model_path: str = "model/final_model"
):
    """
    Create a LiveKit VoicePipelineAgent with Arabic EOU detection
    
    Args:
        vad_plugin: Voice Activity Detection plugin
        stt_plugin: Speech-to-Text plugin
        llm_plugin: Language Model plugin
        tts_plugin: Text-to-Speech plugin
        eou_model_path: Path to Arabic EOU model
        
    Returns:
        Configured VoicePipelineAgent
    """
    # Note: This is a template. Actual implementation depends on LiveKit SDK version
    
    # Create turn detector
    turn_detector = ArabicTurnDetectorModel(model_path=eou_model_path)
    
    # Configure agent
    # (Actual LiveKit integration would go here)
    # agent = agents.VoicePipelineAgent(
    #     vad=vad_plugin,
    #     stt=stt_plugin,
    #     llm=llm_plugin,
    #     tts=tts_plugin,
    #     turn_detector=turn_detector
    # )
    
    return None  # Placeholder


if __name__ == "__main__":
    print("Testing Arabic EOU Plugin...")
    
    try:
        plugin = ArabicEOUPlugin()
        model = plugin.get_model()
        
        # Test
        test_text = "مرحبا كيف حالك؟"
        result = model.update(test_text)
        
        print(f"\nTest: {test_text}")
        print(f"Result: {result}")
        print("\n✓ Plugin initialized successfully")
    
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        print("Model needs to be trained first")
