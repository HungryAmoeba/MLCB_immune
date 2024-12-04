import torch

import torch.nn as nn

class VirchowST(nn.Module):
    # essentially a wrapper that uses Hibou model to extract features and then applies an MLP on top of it
    def __init__(self, virchow_model, output_dim=460, linear_config=[1024]):
        super(VirchowST, self).__init__()
        self.virchow_model = virchow_model

        mlp = torch.nn.Sequential()
        mlp.add_module("linear_0", torch.nn.Linear(virchow_model.config.hidden_size, linear_config[0]))
        mlp.add_module("relu_0", torch.nn.ReLU())

        for i in range(len(linear_config)-1):
            mlp.add_module(f"linear_{i+1}", torch.nn.Linear(linear_config[i], linear_config[i+1]))
            mlp.add_module(f"relu_{i+1}", torch.nn.ReLU())
        
        mlp.add_module("linear_out", torch.nn.Linear(linear_config[-1], output_dim))
        self.mlp = mlp

        # Freeze the hibou model weights
        for param in self.hibou_model.parameters():
            param.requires_grad = False

        self.nonlin = nn.SiLU()
            
    def forward(self, x):
        # Pass the input through the hibou model
        virchow_output = self.virchow_model(**x).pooler_output

        # Apply the linear layers and ReLU activation
        # x = self.nonlin(self.linear1(hibou_output))
        # x = self.linear2(x)
        x = self.mlp(virchow_output)
        return x

if __name__ == '__main__':
    import timm
    import torch

    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked
    from PIL import Image

    # need to specify MLP layer and activation function for proper init
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.eval()

    transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    image1 = Image.open("notebooks/image_crop/1.tif")
    image2 = Image.open("notebooks/image_crop/1.tif")

    image1 = transforms(image1).unsqueeze(0)  # size: 1 x 3 x 224 x 224
    image2 = transforms(image2).unsqueeze(0)  # size: 1 x 3 x 224 x 224

    images = torch.cat([image1, image2], dim=0)  # concatenate images along the batch dimension
    output = model(images)  # size: 2 x 261 x 1280

    class_token = output[:, 0]    # size: 1 x 1280
    patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

    # concatenate class token and average pool of patch tokens
    embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

