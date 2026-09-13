import asyncio
import os
from dotenv import load_dotenv
import numpy as np
import cv2
import time
from loguru import logger
import collections

from pipecat.frames.frames import AudioRawFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from livekit import api, rtc
import torch
from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

class WebRTCSyncPusher(FrameProcessor):
    def __init__(self, transport, model_pipeline, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport
        self.model_pipeline = model_pipeline
        self.width = 512
        self.height = 512
        
        # Audio config
        infer_params = get_infer_params()
        self.sample_rate = infer_params['sample_rate']
        self.tgt_fps = infer_params['tgt_fps']
        self.cached_audio_duration = infer_params['cached_audio_duration']
        self.frame_num = infer_params['frame_num']
        self.motion_frames_num = infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num
        
        self.video_source = rtc.VideoSource(self.width, self.height)
        self.video_track = rtc.LocalVideoTrack.create_video_track("bot-video", self.video_source)
        
        self.audio_source = rtc.AudioSource(self.sample_rate, 1)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("bot-audio", self.audio_source)
        
        self.cached_audio_length_sum = self.sample_rate * self.cached_audio_duration
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)
        
        self.audio_slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        
        self.audio_buffer = bytearray()
        self.audio_float_buffer = []
        
        self.playback_queue = collections.deque()
        self.generation_queue = asyncio.Queue()
        
        idle_img = cv2.imread("examples/omani_character.png")
        if idle_img is None:
            idle_img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            idle_img = cv2.resize(idle_img, (512, 512))
        self.idle_rgba = cv2.cvtColor(idle_img, cv2.COLOR_BGR2RGBA)
        self.idle_vf = rtc.VideoFrame(self.idle_rgba.shape[1], self.idle_rgba.shape[0], rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
        
        self.silent_audio_frame = rtc.AudioFrame(
            data=np.zeros(int(self.sample_rate // self.tgt_fps), dtype=np.int16).tobytes(),
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=int(self.sample_rate // self.tgt_fps)
        )
        
        self.is_publishing = False

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
                    
                    # DYNAMIC ANIMATION MULTIPLIER
                    # Increase this to > 1.0 for more dramatic mouth/head movements
                    animation_multiplier = 1.5 
                    audio_embedding = audio_embedding * animation_multiplier
                    
                    video = run_pipeline(self.model_pipeline, audio_embedding)
                    torch.cuda.synchronize()
                    return video.cpu().numpy()
                
                video_np = await asyncio.to_thread(run_infer)
                
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

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if not self.is_publishing:
            logger.info('Forcing publish on first frame!')
            self.is_publishing = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            
        if isinstance(frame, StartFrame) and not self.is_publishing:
            self.is_publishing = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, AudioRawFrame):
            self.audio_buffer.extend(frame.audio)
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_float_buffer.extend(audio_data.tolist())
            
            while len(self.audio_float_buffer) >= self.audio_slice_samples:
                chunk_floats = np.array(self.audio_float_buffer[:self.audio_slice_samples])
                self.audio_float_buffer = self.audio_float_buffer[self.audio_slice_samples:]
                
                chunk_bytes_len = self.audio_slice_samples * 2
                chunk_bytes = self.audio_buffer[:chunk_bytes_len]
                self.audio_buffer = self.audio_buffer[chunk_bytes_len:]
                
                await self.generation_queue.put((chunk_floats, chunk_bytes))

        else:
            await self.push_frame(frame, direction)

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
            start_time = time.time()
            
            try:
                if len(self.playback_queue) > 10:
                    logger.warning(f"Playback queue too large ({len(self.playback_queue)}). Catching up to real-time...")
                    while len(self.playback_queue) > 5:
                        self.playback_queue.popleft()
                        
                if len(self.playback_queue) > 0:
                    rgba, audio_bytes = self.playback_queue.popleft()
                    
                    vf = rtc.VideoFrame(rgba.shape[1], rgba.shape[0], rtc.VideoBufferType.RGBA, rgba.tobytes())
                    self.video_source.capture_frame(vf)
                    
                    af = rtc.AudioFrame(
                        data=bytes(audio_bytes),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        samples_per_channel=len(audio_bytes) // 2
                    )
                    await self.audio_source.capture_frame(af)
                else:
                    vf = rtc.VideoFrame(self.idle_rgba.shape[1], self.idle_rgba.shape[0], rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
                    self.video_source.capture_frame(vf)
                    await asyncio.sleep(0.04)
                    continue
                    
            except Exception as e:
                logger.error(f"Error playing back frame: {e}")
            
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/self.tgt_fps) - elapsed)
            await asyncio.sleep(sleep_time)

async def main():
    try:
        url = os.environ['LIVEKIT_URL']
        api_key = os.environ['LIVEKIT_API_KEY']
        api_secret = os.environ['LIVEKIT_API_SECRET']
        room_name = 'soulx-flashhead-room'

        token = api.AccessToken(api_key, api_secret) \
            .with_identity('soulx-video-bot') \
            .with_name('SoulX Avatar') \
            .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True)) \
            .to_jwt()

        transport = LiveKitTransport(
            url=url, 
            room_name=room_name,
            token=token,
            params=LiveKitParams(
                audio_in_enabled=True,
                audio_out_enabled=False, 
                video_out_enabled=False, 
                vad_enabled=False,
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
        
        get_base_data(model_pipeline, cond_image_path_or_dir="examples/omani_character.png", base_seed=42, use_face_crop=False)
        
        logger.info("Pre-warming the GPU to build CUDA graphs. This will take ~30-40 seconds...")
        infer_params = get_infer_params()
        sample_rate = infer_params['sample_rate']
        tgt_fps = infer_params['tgt_fps']
        cached_audio_duration = infer_params['cached_audio_duration']
        frame_num = infer_params['frame_num']
        motion_frames_num = infer_params['motion_frames_num']
        
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        
        dummy_audio = np.zeros(cached_audio_length_sum, dtype=np.float32)
        torch.cuda.synchronize()
        dummy_embedding = get_audio_embedding(model_pipeline, dummy_audio, audio_start_idx, audio_end_idx)
        run_pipeline(model_pipeline, dummy_embedding)
        torch.cuda.synchronize()
        logger.info("SoulX Model fully loaded and GPU is pre-warmed.")

        pusher = WebRTCSyncPusher(transport, model_pipeline)

        pipeline = Pipeline([
            transport.input(),
            pusher
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False), idle_timeout_secs=None)
        
        runner = PipelineRunner()
        
        logger.info(f'Starting CONTINUOUS WebRTC streaming bot in {room_name}...')
        await runner.run(task)

    except Exception as e:
        logger.error(f'Error: {e}')

if __name__ == '__main__':
    load_dotenv()
    asyncio.run(main())