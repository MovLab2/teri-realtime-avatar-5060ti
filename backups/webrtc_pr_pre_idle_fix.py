import asyncio
import os
from dotenv import load_dotenv
import numpy as np
import cv2
import time
from loguru import logger
import collections
import soundfile as sf

from pipecat.frames.frames import AudioRawFrame, StartFrame, CancelFrame, InterruptionTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import EndFrame
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


from livekit import api, rtc
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

import sys
sys.path.append(os.path.join(os.getcwd(), 'SoulX-FlashHead'))
from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

AUDIO_FILE = "examples/podcast_sichuan_16k.wav"

class WebRTCSyncPusher(FrameProcessor):
    def __init__(self, transport, model_pipeline, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport
        self.model_pipeline = model_pipeline
        
        infer_params = get_infer_params()
        self.width = infer_params['width']
        self.height = infer_params['height']
        
        self.video_source = rtc.VideoSource(self.width, self.height)
        self.video_track = rtc.LocalVideoTrack.create_video_track("bot-video", self.video_source)
        
        self.sample_rate = infer_params['sample_rate']
        self.tgt_fps = infer_params['tgt_fps']
        self.frame_num = infer_params['frame_num']
        self.motion_frames_num = infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num
        
        self.audio_source = rtc.AudioSource(self.sample_rate, 1)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("bot-audio", self.audio_source)
        
        self.cached_audio_length_sum = self.sample_rate * self.cached_audio_duration
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        
        self.audio_slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        
        # Load the WAV file for feeding
        wav_data, wav_sr = sf.read(AUDIO_FILE, dtype='float32')
        if wav_sr != self.sample_rate:
            raise ValueError(f"WAV sample rate {wav_sr} != expected {self.sample_rate}")
        if len(wav_data.shape) > 1:
            wav_data = wav_data.mean(axis=1)
        self.wav_data = wav_data
        self.wav_total = len(wav_data)
        self.wav_pos = 0
        logger.info(f"Loaded {AUDIO_FILE}: {self.wav_total} samples ({self.wav_total/self.sample_rate:.1f}s)")
        
        self.audio_buffer = bytearray()
        self.audio_float_buffer = []
        
        self.generation_queue = asyncio.Queue()
        self.playback_queue = collections.deque()
        self.is_publishing = False
        
        idle_bgr = cv2.imread("examples/omani_character.png")
        if idle_bgr is None:
            idle_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            idle_img[:] = (0, 0, 255)
        else:
            idle_img = cv2.resize(idle_bgr, (self.width, self.height))
        self.idle_rgba = cv2.cvtColor(idle_img, cv2.COLOR_BGR2RGBA)
        self._last_idle_frame = rtc.VideoFrame(self.idle_rgba.shape[1], self.idle_rgba.shape[0], rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
        
        self.silent_audio_frame = rtc.AudioFrame(
            data=np.zeros(int(self.sample_rate // self.tgt_fps), dtype=np.int16).tobytes(),
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=int(self.sample_rate // self.tgt_fps)
        )
        
    @property
    def cached_audio_duration(self):
        from flash_head.inference import get_infer_params
        return get_infer_params()['cached_audio_duration']
        
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if not self.is_publishing:
            logger.info('Forcing publish on first frame!')
            self.is_publishing = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            asyncio.create_task(self._audio_feed_loop())
            
        if isinstance(frame, StartFrame) and not self.is_publishing:
            self.is_publishing = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            
        if isinstance(frame, (CancelFrame, InterruptionTaskFrame)) or type(frame).__name__ == "InterruptionTaskFrame" or type(frame).__name__ == "CancelFrame":
            logger.info("Interruption detected! Clearing all video and audio buffers.")
            self.audio_buffer.clear()
            self.audio_float_buffer.clear()
            
            while not self.generation_queue.empty():
                try:
                    self.generation_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            self.playback_queue.clear()
            self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
            
        await self.push_frame(frame, direction)

    async def _audio_feed_loop(self):
        # Waits for room to connect, then feeds the WAV file at real-time pace
        logger.info("Waiting for room to connect before feeding audio...")
        while not hasattr(self.transport._client, "_room") or self.transport._client._room is None or not self.transport._client._room.isconnected():
            await asyncio.sleep(0.5)
        logger.info("Room connected. Starting file feed...")

        while True:
            end = self.wav_pos + self.audio_slice_samples
            if end > self.wav_total:
                self.wav_pos = 0
                end = self.audio_slice_samples
                logger.info("Looping audio file...")

            chunk = self.wav_data[self.wav_pos:end]
            self.wav_pos = end

            if len(chunk) < self.audio_slice_samples:
                chunk = np.pad(chunk, (0, self.audio_slice_samples - len(chunk)))

            chunk_bytes = (chunk * 32768.0).astype(np.int16).tobytes()
            await self.generation_queue.put((chunk, chunk_bytes))

            await asyncio.sleep(self.slice_len / self.tgt_fps)

    async def _video_loop(self):
        logger.info("Waiting for room to connect before publishing media...")
        while not hasattr(self.transport._client, "_room") or self.transport._client._room is None or not self.transport._client._room.isconnected():
            await asyncio.sleep(0.5)
            
        room = self.transport._client._room
        
        v_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await room.local_participant.publish_track(self.video_track, v_options)
        
        a_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(self.audio_track, a_options)
        
        logger.info("Starting perfectly synced playback loop...")
        
        while True:
            if not hasattr(self, '_started_playing'):
                self._started_playing = True
                next_frame_time = time.perf_counter()
            
            try:
                if len(self.playback_queue) > 1000:
                    logger.warning(f"Playback queue too large ({len(self.playback_queue)}). This is a severe lag bug.")
                        
                if len(self.playback_queue) > 0:
                    rgba, audio_bytes = self.playback_queue.popleft()
                    
                    i420 = cv2.cvtColor(rgba, cv2.COLOR_RGBA2YUV_I420)
                    self._last_frame_bytes = i420.tobytes()
                    self._last_frame = rtc.VideoFrame(i420.shape[1], int(i420.shape[0] * 2 / 3), 5, self._last_frame_bytes)
                    self.video_source.capture_frame(self._last_frame)
                    
                    af = rtc.AudioFrame(
                        data=bytes(audio_bytes),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        samples_per_channel=len(audio_bytes) // 2
                    )
                    await self.audio_source.capture_frame(af)
                else:
                    idle_i420 = cv2.cvtColor(self.idle_rgba, cv2.COLOR_RGBA2YUV_I420)
                    self._last_idle_bytes = idle_i420.tobytes()
                    self._last_idle_frame = rtc.VideoFrame(idle_i420.shape[1], int(idle_i420.shape[0] * 2 / 3), 5, self._last_idle_bytes)
                    self.video_source.capture_frame(self._last_idle_frame)
                    await self.audio_source.capture_frame(self.silent_audio_frame)
                    
            except Exception as e:
                logger.error(f"Error playing back frame: {e}")
            
            next_frame_time += (1.0 / self.tgt_fps)
            now = time.perf_counter()
            sleep_time = next_frame_time - now
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_frame_time = now
                
    async def _generation_loop(self):
        logger.info("Starting background GPU generation loop...")
        while True:
            chunk_floats, chunk_bytes = await self.generation_queue.get()
            
            self.audio_dq.extend(chunk_floats.tolist())
            audio_array = np.array(self.audio_dq)
            
            try:
                def run_infer():
                    torch.cuda.synchronize()
                    audio_embedding = get_audio_embedding(self.model_pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
                    video = run_pipeline(self.model_pipeline, audio_embedding)
                    torch.cuda.synchronize()
                    return video.float().cpu().numpy()
                
                t_start = time.perf_counter()
                video_np = await asyncio.to_thread(run_infer)
                t_end = time.perf_counter()
                logger.info(f"GPU rendered {video_np.shape[0]} frames in {t_end - t_start:.3f} seconds.")
                
                video_np = video_np[self.motion_frames_num:]
                num_frames = video_np.shape[0]
                bytes_per_video_frame = len(chunk_bytes) // num_frames
                
                for i in range(num_frames):
                    v_frame = video_np[i]

                    if v_frame.shape[0] == 3:
                        v_frame = np.transpose(v_frame, (1, 2, 0))
                    if v_frame.dtype != np.uint8:
                        v_frame = v_frame.astype(np.uint8)
                    rgba = cv2.cvtColor(v_frame, cv2.COLOR_RGB2RGBA)
                    
                    start = i * bytes_per_video_frame
                    end = start + bytes_per_video_frame
                    audio_slice = chunk_bytes[start:end]
                    
                    self.playback_queue.append((rgba, audio_slice))
                    
            except Exception as e:
                logger.error(f"Inference error: {e}")

async def main():
    try:
        url = os.environ.get('LIVEKIT_URL', 'wss://chatgptme-sp76gr03.livekit.cloud')
        api_key = os.environ.get('LIVEKIT_API_KEY')
        api_secret = os.environ.get('LIVEKIT_API_SECRET')
        room_name = os.environ.get('LIVEKIT_ROOM', 'soulx-flashhead-room')

        token = api.AccessToken(api_key, api_secret) \
            .with_identity("soulx-pr-bot") \
            .with_name("SoulX PR Avatar") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
            )).to_jwt()

        
        transport = LiveKitTransport(
            url=url, 
            room_name=room_name,
            token=token,
            params=LiveKitParams(
                audio_in_enabled=False,
                audio_out_enabled=False, 
                video_out_enabled=False, 
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000
            )
        )

        
        logger.info("Loading heavy SoulX Model into VRAM... This may take a minute.")
        
        model_pipeline = get_pipeline(
            world_size=1, 
            ckpt_dir="models/SoulX-FlashHead-1_3B", 
            wav2vec_dir="models/wav2vec2-base-960h", 
            model_type="lite"
        )
        
        logger.info("Pre-warming the GPU to build CUDA graphs. This will take ~30-40 seconds...")
        infer_params = get_infer_params()
        cached_audio_length_sum = infer_params['sample_rate'] * infer_params['cached_audio_duration']
        audio_end_idx = int(infer_params['cached_audio_duration'] * infer_params['tgt_fps'])
        audio_start_idx = audio_end_idx - infer_params['frame_num']
        
        dummy_audio = np.zeros(cached_audio_length_sum, dtype=np.float32)
        get_base_data(model_pipeline, cond_image_path_or_dir="examples/omani_character.png", base_seed=42, use_face_crop=False)
        torch.cuda.synchronize()
        dummy_embedding = get_audio_embedding(model_pipeline, dummy_audio, audio_start_idx, audio_end_idx)
        run_pipeline(model_pipeline, dummy_embedding)
        torch.cuda.synchronize()
        logger.info("SoulX Model fully loaded and GPU is pre-warmed.")

        pusher = WebRTCSyncPusher(transport, model_pipeline)

        pipeline = Pipeline([
            transport.input(),
            pusher,
        ])

        task = PipelineTask(pipeline, params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
        ))

        @transport.event_handler("on_participant_connected")
        async def on_participant_connected(transport, participant):
            logger.info(f"Participant connected: {participant}")
            
        @transport.event_handler("on_participant_disconnected")
        async def on_participant_disconnected(transport, participant):
            logger.info(f"Participant disconnected: {participant}")
        
        runner = PipelineRunner()
        
        logger.info(f'Starting CONTINUOUS WebRTC streaming bot in {room_name}...')
        await runner.run(task)

    except Exception as e:
        import traceback
        logger.error(f'Error: {e}\n{traceback.format_exc()}')

if __name__ == '__main__':
    load_dotenv()
    asyncio.run(main())
