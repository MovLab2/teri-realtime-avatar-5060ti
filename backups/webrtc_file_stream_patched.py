import asyncio
import os
import numpy as np
import cv2
import time
import collections
from loguru import logger

from dotenv import load_dotenv
from pipecat.frames.frames import StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from livekit import api, rtc
import torch
import soundfile as sf
from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline


AUDIO_FILE = "examples/podcast_sichuan_16k.wav"


class FileStreamPusher(FrameProcessor):
    def __init__(self, transport, model_pipeline, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport
        self.model_pipeline = model_pipeline
        self.width = 512
        self.height = 512

        infer_params = get_infer_params()
        self.sample_rate = infer_params['sample_rate']
        self.tgt_fps = infer_params['tgt_fps']
        self.cached_audio_duration = infer_params['cached_audio_duration']
        self.frame_num = infer_params['frame_num']
        self.motion_frames_num = infer_params['motion_frames_num']
        self.slice_len = self.frame_num - self.motion_frames_num

        # Load the whole WAV once
        audio_data, file_sr = sf.read(AUDIO_FILE, dtype='float32')
        if file_sr != self.sample_rate:
            raise ValueError(f"WAV sample rate {file_sr} != expected {self.sample_rate}")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        self.audio_data = audio_data
        self.total_samples = len(audio_data)
        self.audio_pos = 0

        logger.info(f"Loaded {AUDIO_FILE}: {self.total_samples} samples ({self.total_samples/self.sample_rate:.1f}s)")

        # Audio context deque: keeps the last cached_audio_duration seconds for the model's rolling window
        self.cached_audio_length_sum = self.sample_rate * self.cached_audio_duration
        self.audio_dq = collections.deque([0.0] * self.cached_audio_length_sum, maxlen=self.cached_audio_length_sum)

        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num

        self.audio_slice_samples = self.slice_len * self.sample_rate // self.tgt_fps

        # Buffers for feeding chunks into the generation loop
        self.chunk_float_buffer = []
        self.chunk_byte_buffer = bytearray()

        self.playback_queue = collections.deque()
        self.generation_queue = asyncio.Queue()

        # Idle frame
        idle_img = cv2.imread("examples/omani_character.png")
        if idle_img is None:
            idle_img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            idle_img = cv2.resize(idle_img, (512, 512))
        self.idle_rgba = cv2.cvtColor(idle_img, cv2.COLOR_BGR2RGBA)

        # Video source
        self.video_source = rtc.VideoSource(self.width, self.height)
        self.video_track = rtc.LocalVideoTrack.create_video_track("bot-video", self.video_source)

        # Audio source (for publishing file audio to the room)
        self.audio_source = rtc.AudioSource(self.sample_rate, 1)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("bot-audio", self.audio_source)

        self.is_publishing = False
        self._started = False
        self.audio_playback_pos = 0

    async def _generation_loop(self):
        logger.info("Starting file-driven generation loop...")
        while True:
            chunk_floats, chunk_bytes = await self.generation_queue.get()
            self.audio_dq.extend(chunk_floats.tolist())
            audio_array = np.array(self.audio_dq)

            try:
                def run_infer():
                    torch.cuda.synchronize()
                    audio_embedding = get_audio_embedding(
                        self.model_pipeline, audio_array,
                        self.audio_start_idx, self.audio_end_idx
                    )
                    audio_embedding = audio_embedding * 1.5
                    video = run_pipeline(self.model_pipeline, audio_embedding)
                    torch.cuda.synchronize()
                    return video.cpu().numpy()

                video_np = await asyncio.to_thread(run_infer)
                num_frames = video_np.shape[0]
                for i in range(num_frames):
                    v_frame = video_np[i]
                    if v_frame.shape[0] == 3:
                        v_frame = np.transpose(v_frame, (1, 2, 0))
                    if v_frame.dtype != np.uint8:
                        v_frame = v_frame.astype(np.uint8)
                    rgba = cv2.cvtColor(v_frame, cv2.COLOR_RGB2RGBA)
                    self.playback_queue.append(rgba)
            except Exception as e:
                logger.error(f"Inference error: {e}")

    async def _audio_feed_loop(self):
        # Waits for room connect, then feeds the file at a paced rate
        logger.info("Waiting for room to connect before feeding audio...")
        while not hasattr(self.transport._client, "_room") or self.transport._client._room is None or not self.transport._client._room.isconnected():
            await asyncio.sleep(0.5)
        logger.info("Room connected. Starting audio feed...")

        while True:
            # Pull next slice from the file
            end = self.audio_pos + self.audio_slice_samples
            if end > self.total_samples:
                # loop the file
                self.audio_pos = 0
                end = self.audio_slice_samples
                logger.info("Looping audio file...")

            chunk = self.audio_data[self.audio_pos:end]
            self.audio_pos = end

            # Pad if short
            if len(chunk) < self.audio_slice_samples:
                chunk = np.pad(chunk, (0, self.audio_slice_samples - len(chunk)))

            # Convert to int16 bytes for the byte buffer (matching original behavior)
            int16 = (chunk * 32767).astype(np.int16)
            chunk_bytes = int16.tobytes()

            await self.generation_queue.put((chunk, chunk_bytes))

            # Pace: real-time playback rate. slice_len frames at tgt_fps = slice_len/tgt_fps seconds
            await asyncio.sleep(self.slice_len / self.tgt_fps)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if not self.is_publishing:
            logger.info('First frame received — starting loops.')
            self.is_publishing = True
            asyncio.create_task(self._video_loop())
            asyncio.create_task(self._generation_loop())
            asyncio.create_task(self._audio_feed_loop())

        await self.push_frame(frame, direction)

    async def _video_loop(self):
        logger.info("Waiting for room to connect before publishing...")
        while not hasattr(self.transport._client, "_room") or self.transport._client._room is None or not self.transport._client._room.isconnected():
            await asyncio.sleep(0.5)

        room = self.transport._client._room
        v_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await room.local_participant.publish_track(self.video_track, v_options)
        a_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(self.audio_track, a_options)
        logger.info("Video and audio tracks published. Starting playback loop...")

        # Pre-buffer: wait until we have enough chunks queued to give generation a head start
        BUFFER_CHUNKS = 20
        logger.info(f"Pre-buffering {BUFFER_CHUNKS} chunks before playback starts...")
        while len(self.playback_queue) < BUFFER_CHUNKS:
            await asyncio.sleep(0.5)
        logger.info(f"Buffer ready ({len(self.playback_queue)} chunks). Starting real-time playback.")

        while True:
            start_time = time.time()

            try:
                if len(self.playback_queue) > 120:
                    # Too much backlog — drop oldest frames to keep memory and latency sane
                    while len(self.playback_queue) > 60:
                        self.playback_queue.popleft()

                if len(self.playback_queue) > 0:
                    rgba = self.playback_queue.popleft()
                    vf = rtc.VideoFrame(rgba.shape[1], rgba.shape[0], rtc.VideoBufferType.RGBA, rgba.tobytes())
                    self.video_source.capture_frame(vf)

                    # Bundled audio: 640 samples per video frame (16000 / 25), no await
                    samples_per_frame = self.sample_rate // self.tgt_fps
                    audio_start = self.audio_playback_pos
                    audio_end = audio_start + samples_per_frame
                    if audio_end <= self.total_samples:
                        audio_chunk = self.audio_data[audio_start:audio_end]
                    else:
                        audio_chunk = np.pad(self.audio_data[audio_start:], (0, audio_end - self.total_samples))
                    self.audio_playback_pos = audio_end
                    if self.audio_playback_pos >= self.total_samples:
                        self.audio_playback_pos = 0

                    int16 = (audio_chunk * 32767).astype(np.int16)
                    af = rtc.AudioFrame(
                        data=int16.tobytes(),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        samples_per_channel=samples_per_frame
                    )
                    asyncio.create_task(self.audio_source.capture_frame(af))
                else:
                    vf = rtc.VideoFrame(self.idle_rgba.shape[1], self.idle_rgba.shape[0], rtc.VideoBufferType.RGBA, self.idle_rgba.tobytes())
                    self.video_source.capture_frame(vf)
            except Exception as e:
                logger.error(f"Error playing back frame: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/self.tgt_fps) - elapsed)
            await asyncio.sleep(sleep_time)

    async def _audio_publish_loop(self):
        # Waits for room connect, then publishes audio frames at exactly the sample rate
        while not hasattr(self.transport._client, "_room") or self.transport._client._room is None or not self.transport._client._room.isconnected():
            await asyncio.sleep(0.5)

        logger.info("Audio publish loop started.")
        samples_per_frame = self.sample_rate // 50  # 20ms frames = 50 fps audio
        frame_period = 1.0 / 50

        while True:
            start_time = time.time()

            audio_start = self.audio_playback_pos
            audio_end = audio_start + samples_per_frame
            if audio_end <= self.total_samples:
                audio_chunk = self.audio_data[audio_start:audio_end]
            else:
                audio_chunk = np.pad(self.audio_data[audio_start:], (0, audio_end - self.total_samples))
            self.audio_playback_pos = audio_end
            if self.audio_playback_pos >= self.total_samples:
                self.audio_playback_pos = 0

            int16 = (audio_chunk * 32767).astype(np.int16)
            af = rtc.AudioFrame(
                data=int16.tobytes(),
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_frame
            )

            try:
                await self.audio_source.capture_frame(af)
            except Exception as e:
                logger.error(f"Audio capture error: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, frame_period - elapsed)
            await asyncio.sleep(sleep_time)


async def main():
    try:
        url = os.environ['LIVEKIT_URL']
        api_key = os.environ['LIVEKIT_API_KEY']
        api_secret = os.environ['LIVEKIT_API_SECRET']
        room_name = 'soulx-flashhead-room'

        token = api.AccessToken(api_key, api_secret) \
            .with_identity('soulx-file-bot') \
            .with_name('SoulX File Avatar') \
            .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True)) \
            .to_jwt()

        transport = LiveKitTransport(
            url=url,
            room_name=room_name,
            token=token,
            params=LiveKitParams(
                audio_in_enabled=False,
                audio_out_enabled=False,
                video_out_enabled=False,
                vad_enabled=False,
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

        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        dummy_audio = np.zeros(cached_audio_length_sum, dtype=np.float32)
        torch.cuda.synchronize()
        dummy_embedding = get_audio_embedding(model_pipeline, dummy_audio, audio_start_idx, audio_end_idx)
        run_pipeline(model_pipeline, dummy_embedding)
        torch.cuda.synchronize()
        logger.info("SoulX Model fully loaded and GPU is pre-warmed.")

        pusher = FileStreamPusher(transport, model_pipeline)

        pipeline = Pipeline([
            transport.input(),
            pusher
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False), idle_timeout_secs=None)
        runner = PipelineRunner()

        logger.info(f'Starting FILE STREAMING bot in {room_name}...')
        await runner.run(task)

    except Exception as e:
        logger.error(f'Error: {e}')


if __name__ == '__main__':
    load_dotenv()
    asyncio.run(main())
