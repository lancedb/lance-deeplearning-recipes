import io
import os
import time
import warnings
import requests
import tarfile
from tqdm import tqdm
from PIL import Image

import pyarrow as pa
import lance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import torchvision.models as models
import torchvision.transforms as transforms

import wandb

# Ignore warnings
warnings.simplefilter('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU instead")


def process_images(images_folder, split, schema):

    # Iterate over the categories within each data type
    label_folder = os.path.join(images_folder, split)
    for label in os.listdir(label_folder):
        label_folder = os.path.join(images_folder, split, label)
        
        # Iterate over the images within each label
        for filename in tqdm(os.listdir(label_folder), desc=f"Processing {split} - {label}"):
            # Construct the full path to the image
            image_path = os.path.join(label_folder, filename)

            # Read and convert the image to a binary format
            with open(image_path, 'rb') as f:
                binary_data = f.read()

            image_array = pa.array([binary_data], type=pa.binary())
            filename_array = pa.array([filename], type=pa.string())
            label_array = pa.array([label], type=pa.string())
            split_array = pa.array([split], type=pa.string())

            # Yield RecordBatch for each image
            yield pa.RecordBatch.from_arrays(
                [image_array, filename_array, label_array, split_array],
                schema=schema
            )

# Function to write PyArrow Table to Lance dataset
def write_to_lance(images_folder, dataset_name, schema):
    for split in ['test', 'train', 'valid']:
        lance_file_path = os.path.join(images_folder, f"{dataset_name}_{split}.lance")
        
        reader = pa.RecordBatchReader.from_batches(schema, process_images(images_folder, split, schema))
        lance.write_dataset(
            reader,
            lance_file_path,
            schema,
        )

# check for the cinic-10 data folder
if os.path.exists("community-examples/cnn-model-with-cinic-10/cinic-10-data"):
    print("CINIC-10 data folder already exists")
else:
    # Define the URL for the dataset file
    data_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y"

    # Create the data directory if it doesn't exist
    data_dir = "community-examples/cnn-model-with-cinic-10/cinic-10-data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the dataset file
    print("Downloading CINIC-10 dataset...")
    data_file = os.path.join(data_dir, "CINIC-10.tar.gz")

    response = requests.get(data_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    start_time = time.time()
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(data_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    end_time = time.time()
    download_time = end_time - start_time
    progress_bar.close()

    print(f"\nDownload time: {download_time:.2f} seconds")

    # Extract the dataset files
    print("Extracting dataset files...")
    with tarfile.open(data_file, 'r:gz') as tar:
        tar.extractall(path=data_dir)

    print("Dataset downloaded and extracted successfully!")

    dataset_path = "community-examples/cnn-model-with-cinic-10/cinic-10-data"
    dataset_name = os.path.basename(dataset_path)

    start = time.time()
    schema = pa.schema([
        pa.field("image", pa.binary()),
        pa.field("filename", pa.string()),
        pa.field("label", pa.string()),
        pa.field("split", pa.string())
    ])

    start = time.time()
    write_to_lance(dataset_path, dataset_name, schema)
    end = time.time()
    print(f"Time(sec): {end - start:.2f}")


# Define the image classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# transformation function 
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Define the custom dataset class
class CustomImageDataset(data.Dataset):
    def __init__(self, classes, lance_dataset, transform=None):
        self.classes = classes
        self.ds = lance.dataset(lance_dataset)
        self.transform = transform

    def __len__(self):
        return self.ds.count_rows()

    def __getitem__(self, idx):
        raw_data = self.ds.take([idx], columns=['image', 'label']).to_pydict()
        img_data, label = raw_data['image'][0], raw_data['label'][0]

        img = Image.open(io.BytesIO(img_data))

        # Convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.classes.index(label)
        return img, label
    
lr = 1e-3
momentum = 0.9
number_of_epochs = 3
cinic_root = "community-examples/cnn-model-with-cinic-10/cinic-10-data"
train_dataset_path = "community-examples/cnn-model-with-cinic-10/cinic-10-data/cinic-10-data_train.lance/"
test_dataset_path = "community-examples/cnn-model-with-cinic-10/cinic-10-data/cinic-10-data_test.lance/"
validation_dataset_path = "community-examples/cnn-model-with-cinic-10/cinic-10-data/cinic-10-data_valid.lance/"
model_batch_size = 64
batches_to_train = 256
batches_to_val = 128


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs, batch_to_train, batch_to_val, run_name):
    wandb.init(project="cinic-10", name = run_name)
    
    model.train()
    total_start = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batch_start = time.time()

        with tqdm(enumerate(train_loader), total=batch_to_train, desc=f"Epoch {epoch+1}") as pbar_epoch:
            for i, data in pbar_epoch:
                if i >= batch_to_train:
                    break

                optimizer.zero_grad()
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 10 == 0:
                    pbar_epoch.set_postfix({'Loss': loss.item()})
                    pbar_epoch.update(10)

        per_epoch_time = (time.time() - total_batch_start)
        avg_loss = running_loss / batch_to_train
        print(f'Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Time: {per_epoch_time:.4f} secs')
        wandb.log({"Loss": loss.item()})
        wandb.log({"Epoch Duration": per_epoch_time / 60})

    total_training_time = (time.time() - total_start) / 60
    print(f"Total Training Time: {total_training_time:.4f} mins")


    # Validation
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= batch_to_val:
                break
            images_val, labels_val = data[0].to(device), data[1].to(device)
            outputs_val = model(images_val)
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    wandb.log({"Validation Accuracy": val_accuracy})
    print('Finished Training')
    return model


lance_train_dataset = CustomImageDataset(classes, train_dataset_path, transform=transform_train)
lance_test_dataset = CustomImageDataset(classes, test_dataset_path, transform=transform_test)
lance_val_dataset = CustomImageDataset(classes, validation_dataset_path, transform=transform_val)

lance_train_loader = torch.utils.data.DataLoader(lance_train_dataset, batch_size=model_batch_size, shuffle=True)
lance_test_loader = torch.utils.data.DataLoader(lance_test_dataset, batch_size=model_batch_size, shuffle=True)
lance_val_loader = torch.utils.data.DataLoader(lance_val_dataset, batch_size=model_batch_size, shuffle=True)


net = Net(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

lance_trained_model = train_model(lance_train_loader, lance_val_loader, net, criterion, optimizer, device, number_of_epochs, batches_to_train, batches_to_val, run_name = "lance_run_testing")

DIR_PATH = 'community-examples/cnn-model-with-cinic-10/'

# Define the file paths
PATH_LANCE_MODEL = os.path.join(DIR_PATH, 'cinic_resnet_lance_model.pth')

# Check if the directory exists or not, if not create it
if not os.path.isdir(DIR_PATH):
    os.mkdir(DIR_PATH)

# Save the model
torch.save(lance_trained_model.state_dict(), PATH_LANCE_MODEL)

def test_model(test_loader, model, device, type):
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing {type}"):
            images_test, labels_test = data[0].to(device), data[1].to(device)
            outputs_test = model(images_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total_test += labels_test.size(0)
            correct_test += (predicted_test == labels_test).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Test Accuracy: {test_accuracy:.2f}% for {type} dataloader')

# Assuming 'device' is defined, e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_model(lance_test_loader, lance_trained_model, device, "lance")