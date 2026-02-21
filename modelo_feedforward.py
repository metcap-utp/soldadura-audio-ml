"""
FeedForward classifier para embeddings VGGish agregados.

Este modelo agrega los embeddings VGGish de múltiples frames
usando estadísticas (media y desviación estándar) y luego
aplica un clasificador FeedForward simple.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGishAggregator(nn.Module):
    """Agrega embeddings VGGish de múltiples frames."""
    
    def __init__(self, use_std: bool = True):
        super().__init__()
        self.use_std = use_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, features) o (batch, features, time)
        Returns:
            Aggregated features: (batch, features) o (batch, features*2)
        """
        # Handle different input shapes
        if x.dim() == 3:
            if x.size(2) == 128:  # (batch, time, 128)
                mean = x.mean(dim=1)  # (batch, 128)
                if self.use_std:
                    std = x.std(dim=1)
                    return torch.cat([mean, std], dim=1)  # (batch, 256)
                return mean
            else:  # (batch, 128, time)
                mean = x.mean(dim=2)  # (batch, 128)
                if self.use_std:
                    std = x.std(dim=2)
                    return torch.cat([mean, std], dim=1)  # (batch, 256)
                return mean
        else:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")


class FeedForwardClassifier(nn.Module):
    """FeedForward classifier para features agregados."""

    def __init__(
        self,
        input_size: int = 256,  # 128*2 con mean+std
        hidden_sizes = None,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.feature_extractor(x)

        if return_embedding:
            return embedding

        return self.classifier(embedding)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)


class FeedForwardMultiTask(nn.Module):
    """FeedForward con múltiples cabezas de clasificación."""
    
    def __init__(
        self,
        input_size: int = 256,  # 128*2 con mean+std
        hidden_sizes = None,
        num_classes_espesor: int = 3,
        num_classes_electrodo: int = 4,
        num_classes_corriente: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        
        self.aggregator = VGGishAggregator(use_std=True)
        
        self.feature_extractor = FeedForwardClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=3,  # Dummy, replaced below
            dropout=dropout,
        )
        
        # Remove dummy classifier
        del self.feature_extractor.classifier
        
        embedding_dim = hidden_sizes[-1]
        self.classifier_espesor = nn.Linear(embedding_dim, num_classes_espesor)
        self.classifier_electrodo = nn.Linear(embedding_dim, num_classes_electrodo)
        self.classifier_corriente = nn.Linear(embedding_dim, num_classes_corriente)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x: VGGish embeddings (batch, time, 128) or pre-aggregated (batch, 256)
        Returns:
            Dict with logits or embedding
        """
        # Aggregate first (skip if already aggregated)
        if x.dim() == 3:
            aggregated = self.aggregator(x)  # (batch, 256)
        elif x.dim() == 2 and x.size(1) == 256:
            aggregated = x  # Already aggregated
        else:
            raise ValueError(f"Expected 3D (batch, time, 128) or 2D (batch, 256), got {x.shape}")
        
        embedding = self.feature_extractor(aggregated, return_embedding=True)
        
        if return_embedding:
            return embedding
        
        return {
            'logits_espesor': self.classifier_espesor(embedding),
            'logits_electrodo': self.classifier_electrodo(embedding),
            'logits_corriente': self.classifier_corriente(embedding),
        }
    
    def get_embedding(self, x: torch.Tensor):
        return self.forward(x, return_embedding=True)


def test_feedforward():
    """Test the FeedForward models."""
    batch_size = 2
    time_steps = 19
    
    # Test aggregator
    aggregator = VGGishAggregator(use_std=True)
    x = torch.randn(batch_size, time_steps, 128)
    agg = aggregator(x)
    print(f"Input shape: {x.shape}")
    print(f"Aggregated shape: {agg.shape}")
    assert agg.shape == (batch_size, 256)
    
    # Test classifier
    classifier = FeedForwardClassifier(input_size=256, num_classes=3)
    out = classifier(agg)
    print(f"\nClassifier output shape: {out.shape}")
    assert out.shape == (batch_size, 3)
    
    # Test multi-task
    multi_task = FeedForwardMultiTask(input_size=256)
    out = multi_task(x)
    print(f"\nMulti-task output:")
    print(f"  Espesor: {out['logits_espesor'].shape}")
    print(f"  Electrodo: {out['logits_electrodo'].shape}")
    print(f"  Corriente: {out['logits_corriente'].shape}")
    assert out['logits_espesor'].shape == (batch_size, 3)
    assert out['logits_electrodo'].shape == (batch_size, 4)
    assert out['logits_corriente'].shape == (batch_size, 2)
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_feedforward()
