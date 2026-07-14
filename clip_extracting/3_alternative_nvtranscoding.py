
#!/usr/bin/env python3
"""
FFmpeg implementation of video clip extraction tool - High performance version
Replaces the original NVIDIA encoder implementation to solve initialization errors
"""
import argparse
import gc
import logging
import os
import subprocess
import tempfile
import time
from tqdm import tqdm
import concurrent.futures
import shutil

def extract_clip(video_filename, output_filename, start_frame, end_frame, fps, width, height, hw_accel=True):
    """Extract video clips using FFmpeg and encode in HEVC format"""
    # Calculate time in seconds
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.hevc', delete=False) as tmp_file:
        temp_filename = tmp_file.name
    
    try:
        # Choose encoder - use NVENC if hardware acceleration is available
        encoder = 'hevc_nvenc' if hw_accel else 'libx265'
        preset = 'p4' if hw_accel else 'ultrafast'
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-ss', f'{start_time}',
            '-i', video_filename,
            '-t', f'{duration}',
            '-c:v', encoder,
            '-preset', preset,
            '-b:v', '2M',
            '-vf', f'scale={width}:{height}',
            '-an',  # No audio
            '-f', 'hevc',
            temp_filename
        ]
        
        # Execute command, hide output
        process = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        if process.returncode != 0:
            if hw_accel:
                # If hardware acceleration fails, try software encoding
                logging.warning(f"Hardware acceleration encoding failed, switching to software encoding")
                return extract_clip(video_filename, output_filename, start_frame, end_frame, fps, width, height, False)
            else:
                logging.error(f"FFmpeg error: {process.stderr.decode()[:200]}...")
                return None
        
        # Copy temporary file to final location
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        shutil.move(temp_filename, output_filename)
        return output_filename
    
    except Exception as e:
        logging.error(f"Error extracting clip: {e}")
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        return None

def process_clip_task(args):
    """Task function for parallel processing of a single clip"""
    video_path, output_format, s, e, fps, width, height, hw_accel = args
    s, e = int(s), int(e)
    output_file = output_format.format(s, e)
    
    try:
        result = extract_clip(video_path, output_file, s, e, fps, width, height, hw_accel)
        return (result is not None, output_file)
    except Exception as e:
        logging.error(f"Error processing clip {s}-{e}: {e}")
        return (False, None)

def process_one_video(video_filename, vstream_filename_format, clips, args):
    """Process all clips in one video file using parallel processing"""
    files = []
    if len(clips) == 0:
        return files
    
    logging.info(f"Video file: {video_filename}")
    logging.info(f"Number of clips: {len(clips)}")
    
    # Get video information
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'csv=p=0',
        video_filename
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode != 0:
            logging.error(f"Unable to read video information: {probe_result.stderr}")
            return files
            
        width, height, frame_rate = probe_result.stdout.strip().split(',')
        fps = eval(frame_rate)  # Handle fractional frame rates
        logging.info(f"Video info: {width}x{height}, {fps}fps")
    except Exception as e:
        logging.error(f"Error parsing video information: {e}")
        # Use default parameters
        fps = args.fps
    
    # Check if NVENC is available
    hw_accel = args.use_hw_accel
    if hw_accel:
        check_cmd = ['ffmpeg', '-encoders']
        result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if 'hevc_nvenc' not in result.stdout:
            logging.warning("NVENC encoder not detected, will use software encoding")
            hw_accel = False
    
    # Prepare task list
    tasks = [(video_filename, vstream_filename_format, s, e, fps, args.width, args.height, hw_accel) 
             for s, e in clips]
    
    # Process clips in parallel
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = list(tqdm(executor.map(process_clip_task, tasks), total=len(tasks), desc="Processing clips"))
    
    # Collect results
    success_count = 0
    for success, file_path in futures:
        if success and file_path:
            files.append(file_path)
            success_count += 1
    
    # Calculate processing speed
    elapsed = time.time() - start_time
    if elapsed > 0 and success_count > 0:
        speed = success_count / elapsed
        logging.info(f"Processing speed: {speed:.2f} clips/sec")
    
    gc.collect()
    return files

def process_video_task(args_tuple):
    """Task function for parallel processing of multiple videos"""
    idx, vid, total, args = args_tuple
    
    try:
        # Read clip information
        clip_file = os.path.join(args.input_clip_dir, f"{vid}.txt")
        if not os.path.exists(clip_file):
            logging.error(f"Clip file does not exist: {clip_file}")
            return (vid, 0, 0)
            
        with open(clip_file, "r") as f:
            clips = [tuple(line.strip().split(" ")) for line in f]

        logging.info(f"[{idx}/{total}] Starting to process '{vid}' (contains {len(clips)} clips)")

        # Ensure output directory exists
        os.makedirs(os.path.join(args.output_dir, vid), exist_ok=True)
        
        # Check video file
        video_path = os.path.join(args.input_video_dir, f"{vid}.mkv")
        if not os.path.exists(video_path):
            # Try other common formats
            for ext in ['.mp4', '.webm', '.avi']:
                alt_path = os.path.join(args.input_video_dir, f"{vid}{ext}")
                if os.path.exists(alt_path):
                    video_path = alt_path
                    break
                    
            if not os.path.exists(video_path):
                logging.error(f"Video file does not exist: {vid}")
                return (vid, 0, len(clips))
        
        # Process video
        files = process_one_video(
            video_path,
            os.path.join(args.output_dir, vid, f"{vid}_{{:07d}}_{{:07d}}.hevc"),
            clips,
            args
        )
        
        if len(files) == len(clips):
            logging.info(f"Successfully processed {len(files)} clips for '{vid}'")
            return (vid, len(files), 0)
        else:
            logging.warning(f"Partially processed '{vid}': expected {len(clips)} clips, actually processed {len(files)}")
            return (vid, len(files), len(clips) - len(files))
                
    except Exception as e:
        logging.error(f"Error processing video '{vid}': {e}")
        return (vid, 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_clip_dir", type=str, help="Directory containing clip information")
    parser.add_argument("--input_video_dir", type=str, help="Directory containing source videos")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--width", type=int, default=1280, help="Output video width")
    parser.add_argument("--height", type=int, default=720, help="Output video height")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Reserved parameter for compatibility")
    parser.add_argument("--device_id", type=int, default=0, help="Reserved parameter for compatibility")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel threads for single video processing")
    parser.add_argument("--video_workers", type=int, default=4, help="Number of videos to process in parallel")
    parser.add_argument("--use_hw_accel", action="store_true", help="Use hardware acceleration (NVENC)",default=True)
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_clip_dir + "_vstreams"

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("FFmpeg is available, continuing processing")
    except FileNotFoundError:
        logging.error("FFmpeg not found. Please ensure FFmpeg is installed and in PATH")
        return 1

    # Get video file list
    vids = sorted([os.path.splitext(vid)[0] for vid in os.listdir(args.input_clip_dir)])
    logging.info(f"Total {len(vids)} video files to process")
    
    # Process multiple videos in parallel
    tasks = [(i+1, vid, len(vids), args) for i, vid in enumerate(vids)]
    
    success_count = 0
    failed_count = 0
    
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.video_workers) as executor:
        results = list(tqdm(executor.map(process_video_task, tasks), total=len(tasks), desc="Processing videos"))
    
    # Collect statistics
    for vid, success_clips, failed_clips in results:
        if failed_clips == 0:
            success_count += 1
        else:
            failed_count += 1
    
    total_time = time.time() - start_time
    logging.info(f"Processing completed! Success: {success_count}, Failed: {failed_count}")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    exit(main())