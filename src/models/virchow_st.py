import torch

import torch.nn as nn

class VirchowST(nn.Module):
    # essentially a wrapper that uses Hibou model to extract features and then applies an MLP on top of it
    def __init__(self, virchow_model, output_dim=460, linear_config=[1024]):
        super(VirchowST, self).__init__()
        self.virchow_model = virchow_model

        mlp = torch.nn.Sequential()
        mlp.add_module("linear_0", torch.nn.Linear(2560, linear_config[0]))
        mlp.add_module("relu_0", torch.nn.ReLU())

        for i in range(len(linear_config)-1):
            mlp.add_module(f"linear_{i+1}", torch.nn.Linear(linear_config[i], linear_config[i+1]))
            mlp.add_module(f"relu_{i+1}", torch.nn.ReLU())
        
        mlp.add_module("linear_out", torch.nn.Linear(linear_config[-1], output_dim))
        self.mlp = mlp

        # Freeze the Virchow model weights
        for param in self.virchow_model.parameters():
            param.requires_grad = False

        self.nonlin = nn.SiLU()
            
    def forward(self, x):
        # Pass the input through the Virchow model
        output = self.virchow_model(x)  # size: 1 x 261 x 1280

        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        virchow_output = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

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
    processor = transforms

    VirchowWrapper = VirchowST(model, linear_config = [1024, 512, 256])
    print(VirchowWrapper)
