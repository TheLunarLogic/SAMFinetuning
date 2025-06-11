import pandas as pd
import os

def reformated_csv(input,splits):

  for split in splits:
      input_file = os.path.join(input, f"{split}.csv")
      output_file = os.path.join(input, f"{split}_reformatted.csv")

      print(f"Processing {input_file} -> {output_file}")

      df = pd.read_csv(input_file)
      new_df = pd.DataFrame(columns=["image", "label", "points"])

      for index, row in df.iterrows():
          image_path = row["image_path"]
          mask_path = row["mask_path"]

          points = [
              [int(row["point1_x"]), int(row["point1_y"])],
              [int(row["point2_x"]), int(row["point2_y"])],
              [int(row["point3_x"]), int(row["point3_y"])]
          ]

          new_df.loc[index] = [image_path, mask_path, str(points)]

      new_df.to_csv(output_file, index=False)
      print(f"Saved reformatted CSV: {output_file}\n")




