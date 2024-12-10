import cv2
import os
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import imageio

def create_zoomup(frame, x, y):
    # Crop and resize the frame for the zoomed video
    height, width, layers = frame.shape
    xy_zoom = (x, y)
    zoomed_frame = cv2.resize(frame[xy_zoom[1]: (xy_zoom[1] + height//8), xy_zoom[0]: (xy_zoom[0] + width//8), :], (int(width / 4), int(height / 4)))
    # Place the zoomed frame on the top right corner of the canvas
    frame[0:int(height / 4), -int(width / 4):] = zoomed_frame
    return frame

def create_video(input_folder, output_video, frame_rate, method_name, frame_num, x, y):
    # Get a list of all *ep*_rgb.png files in the folder, e.g., "ep0001_0005_rgb.png", "ep00010_0005_rgb.png", ... etc
    log_file = os.path.join(input_folder, "log_ngp.txt")

    image_files = glob.glob(os.path.join(input_folder, "validation", f'*ep*_{frame_num:04d}_*_rgb.png'))
    # Get a list of all error files in the folder, e.g., "ep0001_error_23.56.png", "ep00010_error_25.55.png", ... etc
    error_files = glob.glob(os.path.join(input_folder,"validation", f'*{frame_num:04d}_*_error_*.png'))
    image_files.sort()  # Sort the files to ensure the correct order
    error_files.sort()

    # Extract the error numbers from error files using regular expressions
    error_numbers = [float(re.search(r'error_(\d+\.\d+)', error_file).group(1)) for error_file in error_files]

    # Now, you can use the error_numbers list as needed

    if not image_files:
        print(f"No *ep*_rgb.png files found in the input folder: {input_folder}")
        return

    # Read the first image to get its dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # Create a VideoWriter object to write the main video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    still_img = None
    
        
    for i, image_file in enumerate(image_files):    
        frame = cv2.imread(image_file) # , cv2.IMREAD_GRAYSCALE)
        
        error_number = error_numbers[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        font_color = (255, 255, 255)  # White color

        cv2.putText(frame, f'{method_name} / PSNR: {error_number:.2f}', (10, height - 20),
                   font, font_scale, font_color, font_thickness, cv2.LINE_AA) # / GPU MEM: {(memory_usage[i]/10**9):.1f}

        cv2.putText(frame, f'Training Time: {(i+1)} min', (10, 50),
                    font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        # Crop and resize the frame for the zoomed video
        xy_zoom = (x, y)
        zoomed_frame = cv2.resize(frame[xy_zoom[1]: (xy_zoom[1] + height//8), xy_zoom[0]: (xy_zoom[0] + width//8), :], (int(width / 4), int(height / 4)))
        # Place the zoomed frame on the top right corner of the canvas
        frame[0:int(height / 4), -int(width / 4):] = zoomed_frame
        # Write the result frame to the main video
        if i==1:
            still_img=frame
        
        video_writer.write(frame)

    
    #plt.imshow(still_img)
    #plt.show()    
    # Release the VideoWriter and close all OpenCV windows
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video '{output_video}' has been created.")
    return still_img







def main():
    parser = argparse.ArgumentParser(description="Create two videos from two folders and display them side by side.")
    parser.add_argument("input_folder1", help="Path to the first folder containing *rgb.png files")
    parser.add_argument("input_folder2", help="Path to the second folder containing *rgb.png files")
    parser.add_argument("output_video", help="Output video file name (e.g., output_video.mp4)")
    parser.add_argument("--output_gif", default="output.gif", help="Output gif file name (e.g., output.gif)")
    parser.add_argument("--frame_rate", type=float, default=1, help="Frame rate of the output video (default: 30)")
    parser.add_argument("--frame_num", type=int, default=1)
    parser.add_argument("--x", type=int, default=-1)
    parser.add_argument("--y", type=int, default=-1)

    args = parser.parse_args()

    scene_name=args.input_folder1.split("/")[-2].split('_')[-1]

    frame1 = create_video(args.input_folder1, 'temp1.mp4', args.frame_rate, method_name="Baseline",
                  frame_num=args.frame_num, x=args.x, y=args.y)
    frame2 = create_video(args.input_folder2, 'temp2.mp4', args.frame_rate, method_name="Hard Sample Mining",
                 frame_num=args.frame_num, x=args.x, y=args.y)

    gt_img_path = os.path.join(args.input_folder1, "validation", f'ngp_{args.frame_num:04d}_gt_rgb.png')
    
    gt_img = cv2.imread(gt_img_path)
    height, width, channels = frame1.shape
    gt_img = create_zoomup(gt_img, x=args.x, y=args.y)

    combined_frame = cv2.hconcat([frame1[:,width//2:], frame2[:,width//2:], gt_img[:height,width//2:]])
    cv2.imwrite(f"videos/still_{scene_name}.png", combined_frame)

    
    
    # Read the two video files
    video1 = cv2.VideoCapture('temp1.mp4')
    video2 = cv2.VideoCapture('temp2.mp4')

    # Get the video frame width and height
    width = int(video1.get(3))
    height = int(video1.get(4))

    # Create an output video
    output_video = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), args.frame_rate, (2 * width, height))
    
    i=1
    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()


        if not ret1 or not ret2:
            break

        # Create a frame that combines both videos side by side
        combined_frame = cv2.hconcat([frame1, frame2])

        if i == 2:
            cv2.imwrite(f"videos/sidebyside_{scene_name}.png", combined_frame)

        i += 1

        # Write the combined frame to the output video
        output_video.write(combined_frame)

    # Release the videos and the output video
    video1.release()
    video2.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

