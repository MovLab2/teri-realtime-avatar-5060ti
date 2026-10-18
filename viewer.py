import asyncio
import os
import numpy as np
import sounddevice as sd
import tkinter as tk
from PIL import Image, ImageTk
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

URL = os.environ["LIVEKIT_URL"]
API_KEY = os.environ["LIVEKIT_API_KEY"]
API_SECRET = os.environ["LIVEKIT_API_SECRET"]
ROOM = "soulx-flashhead-room"

async def main():
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity("desktop-viewer")
        .with_name("Desktop Viewer")
        .with_grants(api.VideoGrants(room_join=True, room=ROOM))
        .to_jwt()
    )

    room = rtc.Room()
    audio_stream = sd.RawOutputStream(samplerate=48000, channels=1, dtype="int16", blocksize=4800)
    audio_stream.start()

    root = tk.Tk()
    root.title("Teri Avatar")
    root.geometry("512x512")
    label = tk.Label(root, text="Waiting for video...", bg="black", fg="white")
    label.pack(expand=True, fill="both")

    frame_count = [0]

    async def video_loop(track):
        print("video_loop started")
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
        async for event in stream:
            frame_count[0] += 1
            if frame_count[0] % 30 == 1:
                print(f"Frame {frame_count[0]}: {event.frame.width}x{event.frame.height}")
            frame = event.frame
            img = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            pil_img = Image.fromarray(img)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            label.config(image=tk_img)
            label.image = tk_img

    async def audio_loop(track):
        print("audio_loop started")
        stream = rtc.AudioStream(track)
        async for event in stream:
            audio_stream.write(event.frame.data)

    def on_track_subscribed(track, publication, participant):
        print(f"Track subscribed: {track.kind} from {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(video_loop(track))
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(audio_loop(track))

    room.on("track_subscribed", on_track_subscribed)

    await room.connect(URL, token)
    print("Connected. Subscribing to existing tracks...")

    for participant in room.remote_participants.values():
        print(f"Remote participant: {participant.identity}")
        for publication in participant.track_publications.values():
            print(f"  Track: {publication.kind}, subscribed={publication.subscribed}, has_track={publication.track is not None}")
            publication.set_subscribed(True)

    print("Waiting for tracks to arrive...")
    subscribed_sids = set()
    for _ in range(50):
        await asyncio.sleep(0.1)
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None and publication.track.sid not in subscribed_sids:
                    print(f"Track arrived: {publication.kind} from {participant.identity}")
                    subscribed_sids.add(publication.track.sid)
                    on_track_subscribed(publication.track, publication, participant)

    root.mainloop()

asyncio.run(main())
