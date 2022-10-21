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



    for bag_candidate_idx in range(total):
        bag_candidate = bags_dataset[bag_candidate_idx]
        bag_name = os.path.basename(os.path.normpath(bag_candidate))
        print(bag_name)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        bag_base = bag_name.split('\\')[-1]
        
        if not os.path.exists(os.path.join(feat_dir, bag_base + '.pt')):
            
            print(bag_name)
            
            output_path = os.path.join(feat_dir, bag_name)
            file_path = bag_candidate
            print(file_path)
            output_file_path = Compute_w_loader(file_path, output_path, 
    												model = model, batch_size = batch_size, 
    												verbose = 1, print_every = 20,
    												target_patch_size = target_patch_size)
                        
            if os.path.exists (output_file_path):
                file = h5py.File(output_file_path, "r")
                features = file['features'][:]
                
                print('features size: ', features.shape)
                print('coordinates size: ', file['coords'].shape)
                
                features = torch.from_numpy(features)
                torch.save(features, os.path.join(feat_dir, bag_base+'.pt'))
                file.close()

    




if __name__ == '__main__':

    # input_dir, output_dir
    # initiate data loader
    # use data loader to load and augment images 
    # prediction from model
    # choose save as bag or not (needed?)
    
    # features = torch.from_numpy(features)
    # torch.save(features, output_path + '.pt') 

