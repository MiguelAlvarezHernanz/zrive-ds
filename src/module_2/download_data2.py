import boto3
import os

bucket_name = "zrive-ds-data"
object_key = "groceries/box_builder_dataset/feature_frame.csv"
local_dir = "data/groceries"
local_file_path = os.path.join(local_dir, "sampled_box_builder_df.csv")

s3 = boto3.client('s3')
s3.download_file(bucket_name, object_key, local_file_path)

print(f"File downloaded successfully to: {local_file_path}")
