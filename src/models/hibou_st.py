import torch

import torch.nn as nn

class HibouST(nn.Module):
    # essentially a wrapper that uses Hibou model to extract features and then applies an MLP on top of it
    def __init__(self, hibou_model, output_dim=460, linear_config=[1024]):
        super(HibouST, self).__init__()
        self.hibou_model = hibou_model
        mlp = torch.nn.Sequential()
        mlp.add_module("linear_0", torch.nn.Linear(hibou_model.config.hidden_size, linear_config[0]))
        mlp.add_module("relu_0", torch.nn.ReLU())
        for i in range(len(linear_config)-1):
            mlp.add_module(f"linear_{i}", torch.nn.Linear(linear_config[i], linear_config[i+1]))
            mlp.add_module(f"relu_{i}", torch.nn.ReLU())
        
        mlp.add_module("linear_out", torch.nn.Linear(linear_config[-1], output_dim))

        # Freeze the hibou model weights
        for param in self.hibou_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Pass the input through the hibou model
        hibou_output = self.hibou_model(**x).pooler_output

        # Apply the linear layers and ReLU activation
        x = self.relu(self.linear1(hibou_output))
        x = self.linear2(x)
        return x
    
if __name__ == '__main__':
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
    model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
    hibou_wrapper = HibouST(model, linear_config = [1024, 512, 256])
    print(hibou_wrapper)
    hibou_wrapper_2 = HibouST(model)
    
