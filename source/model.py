import torch
from torch import nn
from torchvision import models


class Identity(nn.Module):
    """
    Identity layer.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


class Dense(nn.Module):
    """
    Fully-connected layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        use_batchnorm=False,
    ):
        super().__init__()

        self.use_bias = use_bias
        self.use_batchnorm = use_batchnorm

        self.dense = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=self.use_bias and not self.use_batchnorm,
        )

        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class ProjectionHead(nn.Module):
    """
    Projection head;
    converts extracted features to the embedding space.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        head_type="nonlinear",
    ):
        """
        Initial;
        :param in_features: number of input feature;
        :param hidden_features: number of hidden features;
        :param out_features: number of output features;
        :param head_type: linear -- one dense layer,
        non-linear -- two dense layers with ReLU activation function;
        """
        super().__init__()

        if head_type == "linear":
            self.layers = Dense(in_features, out_features, False, True)
        elif head_type == "nonlinear":
            self.layers = nn.Sequential(
                Dense(in_features, hidden_features, True, True),
                nn.ReLU(),
                Dense(hidden_features, out_features, False, True),
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class SimCLR(nn.Module):
    """
    Contrastive model.
    """

    def __init__(self):
        super().__init__()

        # Configure base model
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.maxpool = Identity()
        self.encoder.fc = Identity()

        # Unfreeze parameters
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(512, 256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        out = self.projector(out)
        return out
