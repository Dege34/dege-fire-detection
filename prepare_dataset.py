import os
from torch.utils.data import DataLoader
from fire_detection import FireDataset, transform

def prepare_dataset(data_dir, batch_size=32):

    image_paths = []
    labels = []
    
    fire_dir = os.path.join(data_dir, 'fire')
    for img_name in os.listdir(fire_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(fire_dir, img_name))
            labels.append(1) 
    
    no_fire_dir = os.path.join(data_dir, 'no_fire')
    for img_name in os.listdir(no_fire_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(no_fire_dir, img_name))
            labels.append(0) 
    
    dataset = FireDataset(image_paths, labels, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader

if __name__ == "__main__":
    data_dir = "dataset" 
    train_loader = prepare_dataset(data_dir)
    print(f"Dataset prepared with {len(train_loader.dataset)} images") 