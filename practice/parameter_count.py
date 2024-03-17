import torchvision.models as models

def count_parameters(model):
    total_num = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            total_num += parameter.numel() 
    return total_num

if __name__ == "__main__":
    
    model = models.resnet18(pretrained=True)  # Set to False if you don't want to use the pre-trained weights
    
    
    num = count_parameters(model)
    print(num)