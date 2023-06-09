import subprocess

input_file = "downloads/recording-cc1971ee-c2ed-4336-8307-2dd18dad5a4c.m4a"
output_file = "download_wav/myout.wav"

# Run FFmpeg command
subprocess.run(['ffmpeg', '-i', input_file, output_file])