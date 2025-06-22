import os
from webp import load_images
from PIL import Image

# Directory containing the .webp files
input_folder = 'filtered_data'
output_folder = 'filtered_data'

# Make sure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.webp'):
        input_path = os.path.join(input_folder, filename)
        
        # Load the .webp image sequence
        anim = load_images(input_path)
        
        # Create the output .gif filename
        gif_filename = os.path.splitext(filename)[0] + '.gif'
        output_path = os.path.join(output_folder, gif_filename)
        
        # Save the frames as a GIF with the specified duration and loop settings
        anim[0].save(output_path, save_all=True, append_images=anim[1:], duration=80, loop=0)

        print(f"Converted {filename} to {gif_filename}")
