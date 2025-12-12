import asyncio
import logging
import os
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, stt
from livekit.plugins import deepgram, silero

# Import our Arabic EOU SDK
from livekit_eou_sdk.plugin import ArabicTurnDetectorModel

# Visual fix for Windows Terminal
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def fix_text(text):
        try:
            return get_display(arabic_reshaper.reshape(text))
        except: return text
except:
    def fix_text(text): return text

load_dotenv()
logger = logging.getLogger("arabic-eou-agent")

async def entrypoint(ctx: JobContext):
    # Initialize EOU Model
    arabic_eou = ArabicTurnDetectorModel(
        model_path="model/final_model",
        threshold=0.7,
        min_confidence=0.5
    )
    
    print(fix_text("...جاري الاتصال"))
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print(fix_text("!تم الاتصال - ابدأ التحدث"))
    print("Connected! Start speaking...")

    # Initialize STT (Deepgram)
    # Using 'nova-2' model which is excellent for Arabic
    stt_provider = deepgram.STT(model="nova-2", language="ar")

    async def process_track(track: rtc.Track, video_track: rtc.Track = None):
        audio_stream = rtc.AudioStream(track)
        stt_stream = stt_provider.stream()

        async def audio_feeder():
            async for frame in audio_stream:
                stt_stream.push_frame(frame)
            stt_stream.end_input()
        
        asyncio.create_task(audio_feeder())

        async for event in stt_stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                # Get the text
                text = event.alternatives[0].text
                if not text: continue
                
                print(f"\nUser: {fix_text(text)}")
                
                # Check EOU
                # We update the model with the latest text
                # Ideally we want streaming partials, but generic STT events 
                # usually give 'INTERIM' and 'FINAL'.
                # EOU is most useful on INTERIM results (to interrupt).
                # But let's check both.
                
                result = arabic_eou.update(text)
                prob = result['eou_probability']
                
                if result['is_eou']:
                    print(f"   [EOU DETECTED] Prob: {prob:.2f} | {fix_text('نهاية جملة')}")
                else:
                    print(f"   [...Waiting]   Prob: {prob:.2f}")

            elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                # Real-time EOU check on partial text
                text = event.alternatives[0].text
                if not text: continue
                
                result = arabic_eou.update(text)
                prob = result['eou_probability']
                
                if result['is_eou']:
                    print(f" >> [INTERRUPT?] Prob: {prob:.2f} | Tx: {fix_text(text)}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(process_track(track))

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        worker_type=agents.WorkerType.ROOM, # Default, but good to be explicit
        agent_name="Arabic EOU Agent"
    ))
