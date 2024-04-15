import os
import argparse
import pandas as pd
import pyarrow as pa
import lance
import time
from tqdm import tqdm

def process_images(images_folder, data_type):
    # Define schema for RecordBatch
    schema = pa.schema([('image', pa.binary()), 
                        ('filename', pa.string()), 
                        ('category', pa.string()), 
                        ('data_type', pa.string())])

    # Iterate over the categories within each data type
    category_folder = os.path.join(images_folder, data_type)
    for category in os.listdir(category_folder):
        category_folder = os.path.join(images_folder, data_type, category)
        
        # Iterate over the images within each category
        for filename in tqdm(os.listdir(category_folder), desc=f"Processing {data_type} - {category}"):
            # Construct the full path to the image
            image_path = os.path.join(category_folder, filename)

            # Read and convert the image to a binary format
            with open(image_path, 'rb') as f:
                binary_data = f.read()

            image_array = pa.array([binary_data], type=pa.binary())
            filename_array = pa.array([filename], type=pa.string())
            category_array = pa.array([category], type=pa.string())
            data_type_array = pa.array([data_type], type=pa.string())

            # Yield RecordBatch for each image
            yield pa.RecordBatch.from_arrays(
                [image_array, filename_array, category_array, data_type_array],
                schema=schema
            )

# Function to write PyArrow Table to Lance dataset
def write_to_lance(images_folder):
    # Create an empty RecordBatchIterator
    schema = pa.schema([
        pa.field("image", pa.binary()),
        pa.field("filename", pa.string()),
        pa.field("category", pa.string()),
        pa.field("data_type", pa.string())
    ])

    for data_type in ['train', 'test', 'val']:
        lance_file_path = os.path.join(images_folder, f"mini_imagenet_{data_type}.lance")
        
        reader = pa.RecordBatchReader.from_batches(schema, process_images(images_folder, data_type))
        lance.write_dataset(
            reader,
            lance_file_path,
            schema,
        )

def loading_into_pandas(images_folder):
    data_frames = {}  # Dictionary to store DataFrames for each data type
    
    for data_type in ['test', 'train', 'val']:
        uri = os.path.join(images_folder, f"mini_imagenet_{data_type}.lance")

        ds = lance.dataset(uri)

        # Accumulate data from batches into a list
        data = []
        for batch in tqdm(ds.to_batches(columns=["image", "filename", "category", "data_type"], batch_size=10), desc=f"Loading {data_type} batches"):
            tbl = batch.to_pandas()
            data.append(tbl)

        # Concatenate all DataFrames into a single DataFrame
        df = pd.concat(data, ignore_index=True)
        
        # Store the DataFrame in the dictionary
        data_frames[data_type] = df
        
        print(f"Pandas DataFrame for {data_type} is ready")
        print("Total Rows: ", df.shape[0])
    
    return data_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process image dataset.')
    parser.add_argument('--dataset', type=str, help='Path to the image dataset folder')

    try:
        args = parser.parse_args()
        dataset_path = args.dataset
        if dataset_path is None:
            raise ValueError("Please provide the path to the image dataset folder using the --dataset argument.")

        start = time.time()
        write_to_lance(dataset_path)
        data_frames = loading_into_pandas(dataset_path)
        end = time.time()
        print(f"Time(sec): {end - start}")

    except ValueError as e:
        print(e)
        print("Example:")
        print("python script_name.py --dataset /path/to/your/image_dataset_folder")
        exit(1)