import pyiqa
import torch
from tqdm.auto import tqdm
import pathlib
import pandas as pd
from pathlib import Path
import traceback

metrics = ['topiq_fr', 'dists', 'lpips-vgg', 'lpips', 'wadiqam_fr', 'pieapp', 'vif', 'fsim', 'ssim', 'ms-ssim', 'psnr', 'psnry']

# Step 1: Set up the directory path where your images are located
folder_path = pathlib.Path('./datasets/kadid10k\images')
base_path = pathlib.Path('./metrics_values\KADID-10k_pyiqa_metrics')
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
# Step 2: Get all image files from the directory (filter for common image extensions)
image_files = []
for ext in image_extensions:
    image_files.extend(folder_path.glob(ext))
        
ref_images = [str(image).split('.')[0] for image in image_files if '_' not in image.resolve().name]

images_dict = {}
for image in image_files:
    
    image_name = image.resolve().name
    
    if '_' in str(image_name):
        base_name = f'{str(image_name).split("_")[0]}.png'
        images_dict[base_name].append(image)
    else:
        images_dict[image_name] = []

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

for metric in metrics:

    csv_path = base_path / f'KADID-10k_{metric}.csv'
    
    # Step 3: Check if CSV exists. If not, create it with headers
    if not csv_path.exists():
        df = pd.DataFrame(columns=['name', metric])
        df.to_csv(csv_path, index=False)

    # list all available metrics
    #print(pyiqa.list_models())

    # create metric with default setting
    iqa_metric = pyiqa.create_metric(metric, device=device)
    iqa_metric.eval()

    # check if lower better or higher better
    print(iqa_metric.lower_better)

    # example for iqa score inference
    # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
    #score_fr = iqa_metric(img_tensor_x, img_tensor_y)


    # Step 4: Loop through the images with tqdm and append to the CSV in each iteration
    for idx, image in enumerate(tqdm(list(images_dict.keys()), desc="Processing images")):
        absolute_path = folder_path / image  # Get absolute path
        
        for dist_image in images_dict[image]:
            # img path as inputs.
            try:
                score_fr = iqa_metric(str(absolute_path), str(dist_image))
                output_score = score_fr.cpu().item()
            except Exception as e:
                output_score = float('NaN')
                print("Last line of traceback:", traceback.format_exc().strip().splitlines()[-1])
    
    
            new_row = pd.DataFrame({
                'name': [dist_image.name],
                metric: [output_score]
            })

            # Append new row to the CSV file
            new_row.to_csv(csv_path, mode='a', header=False, index=False)

    print(f"CSV saved to {csv_path}")

