import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import glob
import shutil

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FireDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class FireDetectionModel(nn.Module):
    def __init__(self):
        super(FireDetectionModel, self).__init__()
        # Use ResNet50 with pre-trained weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers except the last few
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        # Replace the final layer with a simple classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=30):
    model.train()
    best_loss = float('inf')
    patience = 7
    patience_counter = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0

            torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fire_detection_model.pth'))
            print(f"Saved best model with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def setup_dataset_directory():
    """Create dataset directory structure if it doesn't exist."""
    base_dir = r'C:\Users\OMEN\Desktop\PROJECT 2025\fire_detection\dataset'
    fire_dir = os.path.join(base_dir, 'fire')
    no_fire_dir = os.path.join(base_dir, 'no_fire')
    
    os.makedirs(fire_dir, exist_ok=True)
    os.makedirs(no_fire_dir, exist_ok=True)
    
    return base_dir

def load_dataset(data_dir):
    """Load and prepare the dataset from the given directory."""
    fire_dir = os.path.join(data_dir, 'fire')
    no_fire_dir = os.path.join(data_dir, 'no_fire')
    
    if not os.path.exists(fire_dir) or not os.path.exists(no_fire_dir):
        raise ValueError(f"Dataset directories not found. Please create {fire_dir} and {no_fire_dir} directories")
    
    fire_images = glob.glob(os.path.join(fire_dir, '*.jpg')) + \
                 glob.glob(os.path.join(fire_dir, '*.jpeg')) + \
                 glob.glob(os.path.join(fire_dir, '*.png'))
    
    no_fire_images = glob.glob(os.path.join(no_fire_dir, '*.jpg')) + \
                    glob.glob(os.path.join(no_fire_dir, '*.jpeg')) + \
                    glob.glob(os.path.join(no_fire_dir, '*.png'))
    
    if not fire_images and not no_fire_images:
        raise ValueError("No images found in dataset directories. Please add some images to train the model.")
    
    image_paths = fire_images + no_fire_images
    labels = [1] * len(fire_images) + [0] * len(no_fire_images)
    
    print(f"Found {len(fire_images)} fire images and {len(no_fire_images)} non-fire images")
    return image_paths, labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    try:
        data_dir = setup_dataset_directory()
        print(f"Dataset directory structure created at: {data_dir}")
        print("Please add your images to the following directories:")
        print(f"- {os.path.join(data_dir, 'fire')} for images containing fire")
        print(f"- {os.path.join(data_dir, 'no_fire')} for images without fire")
        
        image_paths, labels = load_dataset(data_dir)
        
        model = FireDetectionModel().to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        dataset = FireDataset(image_paths, labels, transform=transform)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        print("Starting training...")
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=30)
        
        print("Training completed!")
        
        test_dir = os.path.join(data_dir, 'test_images')
        os.makedirs(test_dir, exist_ok=True)
        print(f"\nCreated test directory at: {test_dir}")
        print("Please add images you want to test in this directory")
        
        input("\nPress Enter after you have added images to the test directory...")
        
        model.eval()
        for image_path in glob.glob(os.path.join(test_dir, '*.jpg')) + \
                         glob.glob(os.path.join(test_dir, '*.jpeg')) + \
                         glob.glob(os.path.join(test_dir, '*.png')):
            try:
                image = Image.open(image_path).convert('RGB')
                image = test_transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(image)
                    confidence = output.item()
                    prediction = confidence > 0.5
                
                image_name = os.path.basename(image_path)
                result = "FIRE DETECTED" if prediction else "No fire"
                print(f"Image: {image_name:<30} Result: {result:<15} Confidence: {confidence*100:.1f}%")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease follow these steps:")
        print("1. Create a 'dataset' folder in the same directory as this script")
        print("2. Inside 'dataset', create two folders: 'fire' and 'no_fire'")
        print("3. Add your training images:")
        print("   - Put images containing fire in the 'fire' folder")
        print("   - Put images without fire in the 'no_fire' folder")
        print("4. Run the script again")

if __name__ == "__main__":
    main() 