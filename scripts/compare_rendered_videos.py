from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
import argparse


def merge_videos(video1_path, video2_path, output_path):
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    # Make both videos of the same duration (use the shorter duration)
    min_duration = min(video1.duration, video2.duration)
    video1 = video1.subclip(0, min_duration)
    video2 = video2.subclip(0, min_duration)

    # Crop videos to half width and horizontally concatenate them
    # half_width = video1.size[0] // 2
    # video1_cropped = video1.crop(x1=0, x2=half_width)
    # video2_cropped = video2.crop(x1=half_width, x2=half_width * 2)

    merged_video = clips_array([[video1, video2]])

    # Set the final video dimensions and frame rate
    final_video = merged_video.set_duration(min_duration)
    final_video = final_video.resize(width=video1.size[0] * 2, height=video1.size[1])

    # Write the final video to the specified output path
    final_video.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge method and baseline videos.')
    parser.add_argument('--method', type=str, help='Path to the method video file')
    parser.add_argument('--baseline', type=str, help='Path to the baseline video file')
    parser.add_argument('--output', type=str, default="output.mp4", help='Path to the output merged video file')

    args = parser.parse_args()

    merge_videos(args.method, args.baseline, args.output)

    print('Merged video created successfully.')