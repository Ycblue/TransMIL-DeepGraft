## Choose Model and extract features from (augmented) image patches and save as .pt file

# from datasets.custom_dataloader import HDF5MILDataloader
from datasets import JPGMILDataloader
from torchvision import models

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract_features(input_dir, output_dir, model, batch_size):

    

    dataset = JPGMILDataloader(data_root, label_path=label_path, mode='train', load_data=False, n_classes=n_classes)
    model =  models.resnet50(pretrained=True)    
    for param in self.model_ft.parameters():
        param.requires_grad = False
    self.model_ft.fc = nn.Linear(2048, self.out_features)

    model = model.to(device)
    model.eval()

    




if __name__ == '__main__':

    # input_dir, output_dir
    # initiate data loader
    # use data loader to load and augment images 
    # prediction from model
    # choose save as bag or not (needed?)
    
    # features = torch.from_numpy(features)
    # torch.save(features, output_path + '.pt') 

