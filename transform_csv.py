# import pandas as pd
# import os

# def transform_csv(input_dir,splits):

#   for split in splits:
#       input_file = os.path.join(input_dir, f"{split}.csv")
#       output_file = os.path.join(input_dir, f"{split}_reformatted.csv")

#       print(f"Processing {input_file} -> {output_file}")

#       df = pd.read_csv(input_file)
#       new_df = pd.DataFrame(columns=["image", "label", "points"])

#       for index, row in df.iterrows():
#           image_path = row["image_path"]
#           mask_path = row["mask_path"]

#           points = [
#               [int(row["point1_x"]), int(row["point1_y"])],
#               [int(row["point2_x"]), int(row["point2_y"])],
#               [int(row["point3_x"]), int(row["point3_y"])]
#           ]

#           new_df.loc[index] = [image_path, mask_path, str(points)]

#       new_df.to_csv(output_file, index=False)
#       print(f"Saved reformatted CSV: {output_file}\n")



# transform_csv.py

import pandas as pd
import os
import json
import ast # Import the ast module

def transform_csv(input_dir,splits):

    for split in splits:
        input_file = os.path.join(input_dir, f"{split}.csv")
        output_file = os.path.join(input_dir, f"{split}_reformatted.csv")

        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}. Skipping split '{split}'.")
            continue

        print(f"Processing {input_file}...")
        df = pd.read_csv(input_file)

        reformatted_data = []
        for index, row in df.iterrows():
            # Assuming the input CSV has 'image' and 'label' columns for paths
            # and 'points' column for point data
            image_path = row['image']
            label_path = row['label']
            points_data = row['points']
            reformatted_data.append({
                'image': image_path,
                'label': label_path,
                'points': points_data # Keep the original points data
            })

        reformatted_df = pd.DataFrame(reformatted_data)
        reformatted_df.to_csv(output_file, index=False)
        print(f"Reformatted data saved to {output_file}")
