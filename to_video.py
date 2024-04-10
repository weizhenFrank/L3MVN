from moviepy.editor import ImageSequenceClip
import os

# Directory containing the episode folders
episode_dir = "/home/wliu25/projects/L3MVN/tmp/dump/yolo_thr0.7/episodes/thread_0/"

# Output directory for videos
output_dir = "/home/wliu25/video"
os.makedirs(output_dir, exist_ok=True)

# Custom fps for the output video
custom_fps = 5

# Iterate over episode folders
for episode_folder in os.listdir(episode_dir):
    episode_path = os.path.join(episode_dir, episode_folder)
    if os.path.isdir(episode_path):
        # Get list of image files in the episode folder
        image_files = [os.path.join(episode_path, f) for f in os.listdir(episode_path) if f.endswith(".png")]
        
        # Sort image files by name numerically
        image_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))

        # Generate video from images using moviepy
        clip = ImageSequenceClip(image_files, fps=custom_fps)
        output_video = os.path.join(output_dir, f"{episode_folder}.mp4")
        clip.write_videofile(output_video, codec="libx264", fps=custom_fps)
        print(f"Video generated for episode {episode_folder} with custom fps: {custom_fps}")
