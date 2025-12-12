"""
Demo LiveKit Agent with Arabic EOU Detection
UTF-8 enabled for proper Arabic display
"""

import asyncio
import sys
import os

# Set UTF-8 encoding for Windows terminal
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SDK
from livekit_eou_sdk import ArabicEOUDetector, ArabicTurnDetectorModel

# Import Arabic text fixers for Windows terminal
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    
    def fix_text(text):
        """Reshape and reorder Arabic text for correct terminal display"""
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except:
            return text
except ImportError:
    def fix_text(text): return text


class SimpleArabicAgent:
    """
    Simple demo agent with Arabic EOU detection
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            # Use absolute path relative to script location
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'final_model'))
        print("=" * 80)
        print(fix_text("تجربة كشف نهاية الجملة - اللهجة السعودية"))
        print("Arabic EOU Detection Demo - Saudi Dialect")
        print("=" * 80)
        
        self.eou_detector = ArabicTurnDetectorModel(
            model_path=model_path,
            threshold=0.7,
            min_confidence=0.5
        )
        print(f"\n✓ {fix_text('جاهز! النموذج تم تحميله بنجاح')}")
        print("✓ Ready! Model loaded successfully\n")
    
    def process_utterance(self, text):
        """Process a single utterance"""
        print(f"\n{'='*70}")
        print(f"Text: {fix_text(text)}")
        print(f"{'='*70}")
        
        # Check for EOU
        result = self.eou_detector.update(text)
        
        print(f"EOU Probability: {result['eou_probability']:.3f} | Confidence: {result['confidence']}")
        
        if result['is_eou']:
            print(f"{fix_text('نعم - كشف نهاية الجملة')}")
            print(f"YES - EOU Detected")
            return True
        else:
            print(f"{fix_text('لا - لم تكتشف نهاية الجملة')}")
            print(f"NO - No EOU")
            return False
    
    def run_demo(self):
        """Run demo with examples"""
        print("\n" + "=" * 80)
        print(fix_text("أمثلة توضيحية"))
        print("Example Demonstrations")
        print("=" * 80)
        
        # Examples that should detect EOU (complete sentences)
        print(f"\n--- {fix_text('أمثلة يجب أن تكشف نهاية الجملة')} (YES EOU) ---")
        print("--- Examples that SHOULD detect EOU ---\n")
        
        eou_examples = [
            ("مرحبا كيف حالك؟", "Hello, how are you?"),
            ("شكرا جزيلا على المساعدة", "Thank you very much for the help"),
        ]
        
        for arabic, english in eou_examples:
            print(f"\n({english})")
            self.process_utterance(arabic)
        
        # Examples that should NOT detect EOU (incomplete phrases)
        print(f"\n\n--- {fix_text('أمثلة لا يجب أن تكشف نهاية الجملة')} (NO EOU) ---")
        print("--- Examples that should NOT detect EOU ---\n")
        
        # NOTE: No threshold hacks needed anymore! Model v2 knows the difference.
        
        no_eou_examples = [
            ("أنا أريد", "I want..."),
            ("ممكن تساعدني", "Can you help me..."),
        ]
        
        for arabic, english in no_eou_examples:
            print(f"\n({english})")
            self.process_utterance(arabic)
        
        # Interactive mode
        print("\n\n" + "=" * 80)
        print(fix_text("وضع التجربة التفاعلية"))
        print("Interactive Testing Mode")
        print("=" * 80)
        print(f"\n{fix_text('تعليمات')} / Instructions:")
        print(f"  • {fix_text('اكتب نص بالعربي واضغط Enter')}")
        print("  • Write Arabic text and press Enter")
        print(f"  • {fix_text('اكتب خروج للخروج')}")
        print("  • Type 'quit' to exit")
        print("=" * 80)
        
        while True:
            try:
                print("\n" + "-" * 70)
                # Read input with UTF-8 handling if possible, though input() usually handles system encoding
                user_input = input(f"{fix_text('أدخل النص')} / Enter text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'خروج', 'q']:
                    print(f"\n👋 {fix_text('مع السلامة!')} / Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                self.process_utterance(user_input)
            
            except KeyboardInterrupt:
                print(f"\n\n👋 {fix_text('مع السلامة!')} / Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")


def main():
    """Main entry point"""
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul')
    
    try:
        agent = SimpleArabicAgent()
        agent.run_demo()
    except FileNotFoundError:
        print("\n" + "=" * 80)
        print(" MODEL NOT FOUND")
        print("=" * 80)
        print("\nThe trained model was not found.")
        print("Please ensure model is at: model/final_model")
        print("=" * 80)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
