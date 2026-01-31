import os

video_path = "finaloutputvideo.mp4"

if not os.path.exists(video_path):
    print("❌ Video file not found")
else:
    os.startfile(video_path)
    print("🎬 Video opened in default player")
