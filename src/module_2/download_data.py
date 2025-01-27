import boto3
import os

s3 = boto3.client('s3')

bucket_name = "zrive-ds-data"
prefix = "groceries/sampled-datasets/"
local_dir = "data/groceries"
os.makedirs(local_dir, exist_ok=True)

response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

for obj in response['Contents']:
    file_key = obj['Key']
    file_size = obj['Size'] / (1024 * 1024)  # Tamaño en MB
    print(f"Archivo: {file_key}, Tamaño: {file_size:.2f} MB")

if 'Contents' in response:
    for obj in response['Contents']:
        file_key = obj['Key']
        file_name = os.path.basename(file_key)
        if file_name:  # Evita directorios vacíos
            local_file_path = os.path.join(local_dir, file_name)
            print(f"Descargando {file_name}...")
            s3.download_file(bucket_name, file_key, local_file_path)
            print(f"{file_name} guardado en {local_file_path}")

