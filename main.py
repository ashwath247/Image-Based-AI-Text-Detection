# Importing Required Libraries
from model import Model
from data_pos import Data as POSData
from data_use import Data as USEData
from torch.utils.data import DataLoader

if __name__ == '__main__':
    use_data = 'use'

    if use_data == 'pos':
        # Initialize Data object with the CSV file name
        data_obj = POSData(csv_name='./data/HC3.csv')

        # Save POS tagged images for the 'ai' category
        data_obj.save_pos_tagged_images('ai', images_dir='./data/numsent_3/')

        # Save POS tagged images for the 'human' category
        data_obj.save_pos_tagged_images('human', images_dir='./data/numsent_3/')

        # Save torch data batches to the specified folder path
        data_obj.save_torch_data_batches(folder_path='./data/numsent_3/batches/')

        # Get train, test, and validation datasets
        train_set, test_set, val_set = data_obj.get_train_test_val_data()

        # Initialize the model
        model = Model('zigzag_resnet')

    elif use_data == 'use':
        # Initialize Data object with the CSV file name
        data_obj = USEData(csv_name='./data/HC3.csv')

        # Get train, test, and validation datasets
        train_set, test_set, val_set = data_obj.get_train_test_val_data()

        # Initialize the model
        model = Model('text_resnet')

    # Set the batch size
    batch_size = 32

    # Create a DataLoader for the training set to handle batch loading and shuffling
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False)

    # Create a DataLoader for the validation set
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=False)

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=False)

    # Train the model for the specified number of epochs using the training and validation loaders
    model.train(num_epochs=100, train_loader=train_loader, val_loader=valid_loader)

    # Test the trained model using the test loader
    model.test(dataloader=test_loader)