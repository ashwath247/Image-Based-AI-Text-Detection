# Importing Required Libraries
import argparse

from model import Model
from data_pos import Data as POSData
from data_use import Data as USEData
from  inference_pos import Inference as POSInference
from inference_use import Inference as USEInference
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI-Text Detection')
    parser.add_argument('--method', type = str, default = 'USE', help = 'Parts of Speech or Universal Sentence Encoder')
    parser.add_argument('--train', action = 'store_true', default = False, help = 'Whether to train the model')
    parser.add_argument('--num_epochs', type = int, default = 100, help = 'Number of epochs to train.')
    parser.add_argument('--model_path', type = str, default = './model_store/best_use_model.pth', help = 'Trained model file path')
    parser.add_argument('--infer', action = 'store_true', help = 'Whether to infer from the trained model.')

    args = parser.parse_args()

    use_data = args.method

    # Initialize the model
    if args.method == 'USE':
        model = Model('zigzag_textnet')
    else:
        model = Model('zigzag_resnet')

    if args.train:
        if args.method == 'USE':
            # Initialize Data object with the CSV file name
            data_obj = USEData(csv_name='./data/HC3.csv')

            # Get train, test, and validation datasets
            train_set, test_set, val_set = data_obj.get_train_test_val_data()
        
        else:
            # Initialize Data object with the CSV file name
            data_obj = POSData(csv_name='./data/HC3.csv')

            # Process HC3 Data -> POS Tagged Images -> POS Tagged Tensors
            data_obj.save_pos_tagged_images('ai', images_dir='./data/numsent_3/')
            data_obj.save_pos_tagged_images('human', images_dir='./data/numsent_3/')
            data_obj.save_torch_data_batches(folder_path='./data/numsent_3/batches/')

            # Get train, test, and validation datasets
            train_set, test_set, val_set = data_obj.get_train_test_val_data()

        # Create a DataLoader for Training, Validation and Testing Sets with batch size as 32
        batch_size = 32
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
        valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=False)

        # Train the model for the specified number of epochs using the training and validation loaders
        model.train(num_epochs=args.num_epochs, train_loader=train_loader, val_loader=valid_loader)

        # Test the trained model using the test loader
        model.test(dataloader=test_loader)

    else:
        # Loading trained model
        model.model.load_state_dict(torch.load(args.model_path))

    if args.infer:

        # Initialize Inference Object
        if args.method == 'USE':
            infer_obj = USEInference(model.model)
        else:
            infer_obj = POSInference(model.model)

        # Model Inference
        infer_obj.infer()


