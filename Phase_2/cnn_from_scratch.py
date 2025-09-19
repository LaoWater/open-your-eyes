import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class RestaurantSafetyCNN(nn.Module):
    """
    CNN built from scratch for restaurant safety classification
    
    Real-world application: Classify image patches as:
    - 0: No hands visible
    - 1: Hands with gloves (SAFE)
    - 2: Hands without gloves (VIOLATION)
    """
    
    def __init__(self, num_classes=3):
        super(RestaurantSafetyCNN, self).__init__()
        
        # LAYER 1: Low-level feature extraction
        # Detects edges, textures, basic shapes
        # In restaurants: detects skin texture, fabric texture, food textures
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB input
            out_channels=32,    # 32 different filters
            kernel_size=7,      # 7x7 filter (larger for initial features)
            stride=2,           # Step size 2 (reduces image size)
            padding=3           # Keep spatial dimensions reasonable
        )
        self.bn1 = nn.BatchNorm2d(32)  # Stabilizes training
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # LAYER 2: Mid-level features  
        # Combines edges into shapes, patterns
        # In restaurants: hand shapes, glove patterns, food shapes
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # LAYER 3: Higher-level features
        # More complex patterns and object parts
        # In restaurants: fingers, palm, glove seams, food items
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # LAYER 4: High-level features
        # Object parts and semantic concepts
        # In restaurants: hand poses, glove types, safety compliance indicators
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ADAPTIVE POOLING: Handle varying input sizes
        # Real-world: Restaurant cameras have different resolutions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # FULLY CONNECTED LAYERS: Classification head
        # Combines all features to make final decision
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(0.5)  # Prevents overfitting
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        # OUTPUT LAYER: Final classification
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights properly (important for training stability)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization - crucial for training success"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal initialization for FC layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass - let's trace through with a restaurant image
        Input: [batch_size, 3, 224, 224] - RGB restaurant image
        """
        batch_size = x.size(0)
        
        # LAYER 1: Extract basic features (edges, textures)
        # What happens: Filters detect skin texture, fabric patterns, metal surfaces
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Shape: [batch_size, 32, 56, 56]
        
        # LAYER 2: Combine into shapes and patterns  
        # What happens: Hand outlines, glove shapes, food item boundaries
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Shape: [batch_size, 64, 28, 28]
        
        # LAYER 3: More complex patterns
        # What happens: Finger positions, glove seams, safety equipment
        x = F.relu(self.bn3(self.conv3(x)))
        # Shape: [batch_size, 128, 28, 28]
        
        # LAYER 4: High-level semantic features
        # What happens: "This looks like a gloved hand", "This is unsafe"
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # Shape: [batch_size, 256, 14, 14]
        
        # ADAPTIVE POOLING: Standardize feature map size
        # Real-world benefit: Works with any input image size
        x = self.adaptive_pool(x)
        # Shape: [batch_size, 256, 7, 7]
        
        # FLATTEN: Convert to 1D for classification
        x = x.view(batch_size, -1)
        # Shape: [batch_size, 256*7*7] = [batch_size, 12544]
        
        # CLASSIFICATION HEAD: Make final decision
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # OUTPUT: Class probabilities
        x = self.fc3(x)
        # Shape: [batch_size, 3] - probabilities for each safety class
        
        return x
    
    def get_feature_maps(self, x, layer='conv1'):
        """
        Extract intermediate feature maps for visualization
        Useful for debugging and understanding what the model learned
        """
        if layer == 'conv1':
            return F.relu(self.bn1(self.conv1(x)))
        elif layer == 'conv2':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            return F.relu(self.bn2(self.conv2(x)))
        elif layer == 'conv3':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return F.relu(self.bn3(self.conv3(x)))
        # Add more layers as needed

# Custom Dataset for Restaurant Safety
class RestaurantSafetyDataset(Dataset):
    """
    Custom dataset for restaurant safety images
    Real-world: This is how you'd load your annotated restaurant data
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Class names for interpretation
        self.class_names = ['no_hands', 'hands_with_gloves', 'hands_without_gloves']
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Training function with real-world considerations
class SafetyModelTrainer:
    """
    Complete training pipeline for restaurant safety CNN
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with different learning rates for different layers
        # Real-world tip: Lower LR for pretrained features, higher for new layers
        self.optimizer = torch.optim.Adam([
            {'params': self.model.conv1.parameters(), 'lr': 1e-4},
            {'params': self.model.conv2.parameters(), 'lr': 1e-4},
            {'params': self.model.conv3.parameters(), 'lr': 5e-4},
            {'params': self.model.conv4.parameters(), 'lr': 5e-4},
            {'params': self.model.fc1.parameters(), 'lr': 1e-3},
            {'params': self.model.fc2.parameters(), 'lr': 1e-3},
            {'params': self.model.fc3.parameters(), 'lr': 1e-3},
        ])
        
        # Learning rate scheduler - reduces LR when training plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping - prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item():.4f}, '
                      f'Acc {100.*correct/total:.2f}%')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validation with detailed per-class metrics"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Per-class metrics
        class_correct = [0, 0, 0]  # no_hands, gloves_on, gloves_off
        class_total = [0, 0, 0]
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Per-class accuracy
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i] == targets[i]:
                        class_correct[label] += 1
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Print per-class accuracy (important for safety applications!)
        class_names = ['No Hands', 'Hands w/ Gloves', 'Hands w/o Gloves']
        for i in range(3):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                print(f'{class_names[i]} Accuracy: {class_acc:.2f}%')
        
        return epoch_loss, epoch_acc
    
    def train_model(self, train_loader, val_loader, epochs=50):
        """Complete training loop with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10  # Early stopping patience
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_safety_model.pth')
                print('✓ New best model saved!')
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    def plot_training_curves(self):
        """Plot training curves for analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

# Feature Visualization - Understanding what the CNN learned
class FeatureVisualizer:
    """
    Visualize what different layers of the CNN are learning
    Critical for debugging and understanding model behavior
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def visualize_filters(self, layer_name='conv1'):
        """Visualize the learned filters"""
        if layer_name == 'conv1':
            filters = self.model.conv1.weight.data.clone()
        elif layer_name == 'conv2':
            filters = self.model.conv2.weight.data.clone()
        
        # Normalize filters for visualization
        filters = filters - filters.min()
        filters = filters / filters.max()
        
        # Plot first 16 filters
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(min(16, filters.shape[0])):
            ax = axes[i//4, i%4]
            
            if filters.shape[1] == 3:  # RGB filters
                filter_img = filters[i].permute(1, 2, 0)
            else:  # Grayscale or single channel
                filter_img = filters[i, 0]
            
            ax.imshow(filter_img, cmap='gray')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        plt.suptitle(f'{layer_name} Learned Filters')
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_maps(self, image, layer='conv1'):
        """Visualize feature maps for a specific image"""
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # Convert numpy to tensor
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0)
            else:
                image_tensor = image.unsqueeze(0)
            
            # Get feature maps
            feature_maps = self.model.get_feature_maps(image_tensor, layer)
            feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
            
            # Plot first 16 feature maps
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            for i in range(min(16, feature_maps.shape[0])):
                ax = axes[i//4, i%4]
                fmap = feature_maps[i].cpu().numpy()
                
                ax.imshow(fmap, cmap='viridis')
                ax.set_title(f'Feature Map {i+1}')
                ax.axis('off')
            
            plt.suptitle(f'{layer} Feature Maps')
            plt.tight_layout()
            plt.show()

# Real-world usage example
def demonstrate_cnn_training():
    """
    Demonstrate complete CNN training pipeline
    """
    print("🏗️  Building CNN from scratch for restaurant safety...")
    
    # Initialize model
    model = RestaurantSafetyCNN(num_classes=3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data transforms (real-world preprocessing)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # In real-world, you'd load actual restaurant images:
    # train_dataset = RestaurantSafetyDataset(train_paths, train_labels, train_transform)
    # val_dataset = RestaurantSafetyDataset(val_paths, val_labels, val_transform)
    
    print("\n📊 Model Architecture:")
    print(model)
    
    print("\n🎯 Key CNN Concepts Demonstrated:")
    print("✓ Hierarchical feature learning (edges → shapes → objects)")
    print("✓ Translation invariance through pooling")
    print("✓ Parameter sharing through convolution")
    print("✓ Proper weight initialization")
    print("✓ Batch normalization for stable training")
    print("✓ Dropout for regularization")

if __name__ == "__main__":
    demonstrate_cnn_training()