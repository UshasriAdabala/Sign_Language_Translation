import os
from PIL import Image
def convert_webp_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".webp"):
            try:
                img = Image.open(os.path.join(input_folder, filename))
                img.save(os.path.join(output_folder, filename.replace(".webp", ".png")), 'PNG')
            except Exception as e:
                print(f"Error converting {filename}: {e}")
convert_webp_to_png('filtered_data', 'processed_data')     
         