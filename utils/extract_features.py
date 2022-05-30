## Choose Model and extract features from (augmented) image patches and save as .pt file

from datasets.custom_dataloader import HDF5MILDataloader


def extract_features(input_dir, output_dir, model, batch_size):


    dataset = HDF5MILDataloader(data_root, label_path=label_path, mode='train', load_data=False, n_classes=n_classes)
    if model == 'resnet50':
        model = Resnet50_baseline(pretrained = True)       
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

