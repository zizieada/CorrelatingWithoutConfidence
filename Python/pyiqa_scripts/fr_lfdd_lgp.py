import pyiqa
import torch
from tqdm.auto import tqdm
import pathlib
import pandas as pd
from pathlib import Path
import traceback

metrics = ['topiq_fr', 'dists', 'lpips-vgg', 'lpips', 'wadiqam_fr', 'pieapp', 'vif', 'fsim', 'ssim', 'ms_ssim', 'psnr', 'psnry']

# Step 1: Set up the directory path where your images are located
base_path = pathlib.Path('./metrics_values/LFDD_LGP_pyiqa_metrics')

folder_path = pathlib.Path('./datasets/LFDD_LGP')
subfolders = [f for f in pathlib.Path(folder_path).iterdir() if f.is_dir()]

image_files = []
for subfolder in subfolders:
    subfolder_scenes = [f for f in pathlib.Path(subfolder).iterdir() if f.is_dir()]
    image_files.extend(subfolder_scenes[1:])

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
image_names = []
for ext in image_extensions:
    image_names.extend(image_files[0].glob(ext))
image_names = [name.name for name in image_names]
    
print(len(image_files))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

for metric in metrics:

    csv_path = base_path / f'LFDD_LGP_{metric}.csv'
    
    # Step 3: Check if CSV exists. If not, create it with headers
    if not csv_path.exists():
        df = pd.DataFrame(columns=[*range(0,len(image_names),1)]+['name', metric])
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
    for idx, scene in enumerate(tqdm(image_files, desc="Processing images")):
        
        
        scores = []
        for image in image_names:
            
            dist_image = scene / image   # Get absolute path
            ref_image = folder_path / scene.parent.parts[-1]
            ref_image = ref_image / scene.parent.parts[-1]
            ref_image = ref_image / image
        
            # img path as inputs.
            try:
                score_fr = iqa_metric(str(ref_image), str(dist_image))
                output_score = score_fr.cpu().item()
            except Exception as e:
                output_score = float('NaN')
                print("Last line of traceback:", traceback.format_exc().strip().splitlines()[-1])
            
            scores.append(output_score)
        
        my_dict = dict(zip([*range(0,len(image_names),1)], scores))
        my_dict['name'] = [scene.name]
        my_dict[metric] = [output_score]
    
        new_row = pd.DataFrame(my_dict)
        
        # Append new row to the CSV file
        new_row.to_csv(csv_path, mode='a', header=False, index=False)

    print(f"CSV saved to {csv_path}")

