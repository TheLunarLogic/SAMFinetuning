import pandas as pd
import ast
import os

# Load the original CSV
input_file = "processed_dataset/test.csv"
output_file = "test_reformatted.csv"

# Read the original CSV
df = pd.read_csv(input_file)

# Create a new DataFrame with the desired structure
new_df = pd.DataFrame(columns=["image", "label", "points"])

# Process each row
for index, row in df.iterrows():
    image_path = row["image_path"]
    mask_path = row["mask_path"]
    
    # Extract points and format them as a list of [x, y] coordinates
    points = [
        [int(row["point1_x"]), int(row["point1_y"])],
        [int(row["point2_x"]), int(row["point2_y"])],
        [int(row["point3_x"]), int(row["point3_y"])]
    ]
    
    # Add to the new DataFrame
    new_df.loc[index] = [image_path, mask_path, str(points)]

# Save the reformatted CSV
new_df.to_csv(output_file)
print(f"Converted CSV saved to {output_file}")