import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import tqdm
# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Ensure dataset directory exists
data_dir = "PlantVillage-Dataset/raw/color"  # Adjust this if needed
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

# Define transforms (Now applied inside Dataset class)
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Add vertical flip
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])  # Sorted class names
        self.image_paths = []
        self.labels = []
        for label, class_dir in enumerate(self.classes):
            class_dir_path = os.path.join(root_dir, class_dir)
            for fname in os.listdir(class_dir_path):
                file_path = os.path.join(class_dir_path, fname)
                if os.path.isfile(file_path):  # Ensure it's a file
                    self.image_paths.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Create dataset and split into training and testing sets
dataset = PlantVillageDataset(data_dir, transform=None)
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please check the dataset path and contents.")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply transforms to the datasets
train_dataset.dataset.transform = train_transforms
test_dataset.dataset.transform = test_transforms

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a more complex model (e.g., ResNet50)
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers except the final layer

model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout
    nn.Linear(model.fc.in_features, len(dataset.classes))
)
for param in model.fc.parameters():
    param.requires_grad = True  # Unfreeze the final layer

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only optimize the final layer
model_path = os.path.join(output_dir, 'model.pth')
# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        torch.save(model.state_dict(), model_path)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Save the model

torch.save(model.state_dict(), model_path)

print(f'Model saved to {model_path}')