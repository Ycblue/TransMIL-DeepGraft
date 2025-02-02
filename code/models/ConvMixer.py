import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list


class ConvMixerLayer(Module):
    """
    <a id="ConvMixerLayer"></a>
    ## ConvMixer layer
    This is a single ConvMixer layer. The model will have a series of these.
    """

    def __init__(self, d_model: int, kernel_size: int):
        """
        * `d_model` is the number of channels in patch embeddings, $h$
        * `kernel_size` is the size of the kernel of spatial convolution, $k$
        """
        super().__init__()
        # Depth-wise convolution is separate convolution for each channel.
        # We do this with a convolution layer with the number of groups equal to the number of channels.
        # So that each channel is it's own group.
        self.depth_wise_conv = nn.Conv2d(d_model, d_model,
                                         kernel_size=kernel_size,
                                         groups=d_model,
                                         padding=(kernel_size - 1) // 2)
        # Activation after depth-wise convolution
        self.act1 = nn.GELU()
        # Normalization after depth-wise convolution
        self.norm1 = nn.BatchNorm2d(d_model)

        # Point-wise convolution is a $1 \times 1$ convolution.
        # i.e. a linear transformation of patch embeddings
        self.point_wise_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        # Activation after point-wise convolution
        self.act2 = nn.GELU()
        # Normalization after point-wise convolution
        self.norm2 = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor):
        # For the residual connection around the depth-wise convolution
        residual = x

        # Depth-wise convolution, activation and normalization
        x = self.depth_wise_conv(x)
        x = self.act1(x)
        x = self.norm1(x)

        # Add residual connection
        x += residual

        # Point-wise convolution, activation and normalization
        x = self.point_wise_conv(x)
        x = self.act2(x)
        x = self.norm2(x)

        #
        return x


class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings"></a>
    ## Get patch embeddings
    This splits the image into patches of size $p \times p$ and gives an embedding for each patch.
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the number of channels in patch embeddings $h$
        * `patch_size` is the size of the patch, $p$
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        # Activation function
        self.act = nn.GELU()
        # Batch normalization
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Apply convolution layer
        x = self.conv(x)
        # Activation and normalization
        x = self.act(x)
        x = self.norm(x)

        #
        return x


class ClassificationHead(Module):
    """
    <a id="ClassificationHead"></a>
    ## Classification Head
    They do average pooling (taking the mean of all patch embeddings) and a final linear transformation
    to predict the log-probabilities of the image classes.
    """

    def __init__(self, d_model: int, n_classes: int):
        """
        * `d_model` is the number of channels in patch embeddings, $h$
        * `n_classes` is the number of classes in the classification task
        """
        super().__init__()
        # Average Pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor):
        # Average pooling
        x = self.pool(x)
        # Get the embedding, `x` will have shape `[batch_size, d_model, 1, 1]`
        x = x[:, :, 0, 0]
        # Linear layer
        x = self.linear(x)

        #
        return x


class ConvMixer(Module):
    """
    ## ConvMixer
    This combines the patch embeddings block, a number of ConvMixer layers and a classification head.
    """

    def __init__(self, conv_mixer_layer: ConvMixerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings,
                 classification: ClassificationHead):
        """
        * `conv_mixer_layer` is a copy of a single [ConvMixer layer](#ConvMixerLayer).
         We make copies of it to make ConvMixer with `n_layers`.
        * `n_layers` is the number of ConvMixer layers (or depth), $d$.
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        # Classification head
        self.classification = classification
        # Make copies of the [ConvMixer layer](#ConvMixerLayer)
        self.conv_mixer_layers = clone_module_list(conv_mixer_layer, n_layers)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[batch_size, d_model, height / patch_size, width / patch_size]`.
        x = self.patch_emb(x)
        print(x.shape)
        # Pass through [ConvMixer layers](#ConvMixerLayer)
        for layer in self.conv_mixer_layers:
            x = layer(x)

        # Classification head, to get logits
        x = self.classification(x)

        #
        return x

if __name__ == '__main__':


    convmix = ConvMixerLayer(d_model=20, kernel_size=9)
    patch_emb = PatchEmbeddings(d_model=20, patch_size=7, in_channels=3)
    classification = ClassificationHead(d_model=20, n_classes=2)
    model = ConvMixer(conv_mixer_layer=convmix, n_layers=20, patch_emb = patch_emb, classification=classification )
    x = torch.randn([1,3,224,224])
    
    y = model(x)
    print(y.shape)