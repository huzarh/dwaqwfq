import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import torchvision.models as models
import logging

class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weight factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SpeakerClassifier(nn.Module):
    """ResNet-18 based model for speaker classification."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of speakers to classify
            pretrained: Whether to use pretrained weights
        """
        super(SpeakerClassifier, self).__init__()
        
        # Load pretrained ResNet-18 model
        self.resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        # Modify first layer to accept single-channel input (grayscale)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Modify the final fully connected layer for speaker classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final FC layer
        
        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights for the modified layers
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        logging.info(f"Initialized SpeakerClassifier with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input spectrogram of shape (batch_size, n_mels, time)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Get features from ResNet backbone
        features = self.resnet(x)
        
        # Apply classifier head
        logits = self.classifier(features)
        
        return logits

class SpectralAttention(nn.Module):
    """Spectral attention module for enhanced feature learning."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        """
        Initialize the attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio for attention
        """
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature map of shape (batch_size, channels, height, width)
            
        Returns:
            Attended feature map
        """
        # Average pool features
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        
        # Max pool features
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        
        # Combine attention weights
        out = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        return x * out

class SpeakerClassifierWithAttention(SpeakerClassifier):
    """Extended SpeakerClassifier with spectral attention mechanism."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        Initialize the model with attention.
        
        Args:
            num_classes: Number of speakers to classify
            pretrained: Whether to use pretrained weights
        """
        super(SpeakerClassifierWithAttention, self).__init__(num_classes, pretrained)
        
        # Add spectral attention modules at key points in the network
        self.attention1 = SpectralAttention(64)   # After layer1
        self.attention2 = SpectralAttention(128)  # After layer2
        self.attention3 = SpectralAttention(256)  # After layer3
        self.attention4 = SpectralAttention(512)  # After layer4
        
        logging.info(f"Initialized SpeakerClassifierWithAttention with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention modules.
        
        Args:
            x: Input spectrogram of shape (batch_size, n_mels, time)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Extract features with attention
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.attention1(x)
        
        x = self.resnet.layer2(x)
        x = self.attention2(x)
        
        x = self.resnet.layer3(x)
        x = self.attention3(x)
        
        x = self.resnet.layer4(x)
        x = self.attention4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply classifier head
        logits = self.classifier(x)
        
        return logits 