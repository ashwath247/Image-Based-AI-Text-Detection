import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model():
    """
    Model Class to handle training and testing the POS and USE versions of ZigZag ResNet

    Methods
    -------
    train(num_epochs, train_loader, val_loader)
        Train the model for multiple epochs

    train_val(train_loader, val_loader)
        Train and validate the model for a single epoch

     test(dataloader)
        Test the provided test dataloader on the trained model
    """

    def __init__(self, model_name, num_gpus = 1):
        """
        Model Class init

        Parameters
        ----------
        model_name: str
            'zigzag_resnet' or 'zigzag_textnet'
        num_gpus: int
            Number of GPUs to utliize for training
        """

        # Check if CUDA is available, set the device accordingly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_name == 'zigzag_resnet':
            # Create a ZigZag_ResNet model 
            self.model = ZigZag_ResNet(BasicBlock, [2, 2, 2, 2, 2, 1, 1], num_classes=2).to(self.device)
            
            # Define the CrossEntropyLoss criterion for classification
            self.criterion = nn.CrossEntropyLoss()
            
            # Create an SGD optimizer with learning rate 0.001, momentum 0.8, weight decay 0.0005, and Nesterov momentum
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001, momentum=0.8, weight_decay=0.0005, nesterov=True)
            
            # Create a ZigZagLROnPlateauRestarts scheduler with mode 'max', initial LR 0.001,
            # up factor 0.3, down factor 0.5, up patience 1, down patience 1, restart after 30 epochs, and verbose output
            self.scheduler = ZigZagLROnPlateauRestarts(self.optimizer, mode='max', lr=0.001,
                                                    up_factor=0.3, down_factor=0.5,
                                                    up_patience=1, down_patience=1,
                                                    restart_after=30, verbose=True)

        elif model_name == 'zigzag_textnet':
            # Create a ZigZag_TextResNet model
            self.model = ZigZag_TextResNet(BasicBlock, [2, 2, 2, 2, 2, 1, 1], num_classes = 2).to(self.device)

            # Define the CrossEntropyLoss criterion for classification
            self.criterion = nn.CrossEntropyLoss()
            
            # Create an SGD optimizer with learning rate 0.001, momentum 0.8, weight decay 0.0005, and Nesterov momentum
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001, momentum=0.8, weight_decay=0.0005, nesterov=True)
            
            # Create a ZigZagLROnPlateauRestarts scheduler with mode 'max', initial LR 0.001,
            # up factor 0.3, down factor 0.5, up patience 1, down patience 1, restart after 30 epochs, and verbose output
            self.scheduler = ZigZagLROnPlateauRestarts(self.optimizer, mode='max', lr=0.001,
                                                    up_factor=0.3, down_factor=0.5,
                                                    up_patience=1, down_patience=1,
                                                    restart_after=30, verbose=True)

        # For Utilizing Multiple GPUs
        if num_gpus != 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpus)))
    
    def train(self, num_epochs, train_loader, val_loader):
        """
        Function to train and validate the model for multiple epochs.

        Parameters
        ----------
        num_epochs: int
            Number of epochs to train
        train_loader: torch.utils.data.DataLoader
            Training Set Data Loader
        val_loader: torch.utils.data.DataLoader
            Validation Set Data Loader
        """

        # Initializing corresponding train and validation arrays for storing accuracries and losses
        train_losses_ = []  
        train_accuracies_ = []  
        valid_losses_ = []  
        valid_accuracies_ = []  
        
        # Iterating for num_epochs times
        for epoch in range(num_epochs):
            print(f"\n\tEpoch: {epoch+1}/{num_epochs}")
            
            # Perform training and validation for the current epoch
            train_loss, train_accuracy, val_loss, val_accuracy = self.train_val(train_loader, val_loader)
            
            # Append the results to the respective lists
            train_losses_.append(train_loss)
            train_accuracies_.append(train_accuracy)
            valid_losses_.append(val_loss)
            valid_accuracies_.append(val_accuracy)
            
            # Print the training and validation metrics for the current epoch
            print(f"\tTraining Loss: {round(train_loss, 4)}; Training Accuracy: {round(train_accuracy*100, 4)}%")
            print(f"\tValidation Loss: {round(val_loss, 4)}; Validation Accuracy: {round(val_accuracy*100, 4)}%")

    def train_val(self, train_loader, val_loader):
        """
        Function to train and validate the model for multiple epochs.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            Training Set Data Loader
        val_loader: torch.utils.data.DataLoader
            Validation Set Data Loader

        Returns
        -------
        train_loss: int
            Training Loss
        train_accuracy: int
            Training Accuracy
        valid_loss: int
            Validation Loss
        valid_accuracy: int
            Validation Accuracy
        """

        # Set the model in training mode
        self.model.train()  
        
        # Initializing variables to compute loss and accuracy
        train_loss = 0  
        correct = 0  
        total = 0 
        
        # Iterating over the train data loader
        for i, data in enumerate(train_loader, 0):
            image, label = data

            # Send images and labels to GPU
            image = image.to(self.device)
            label = label.to(self.device)
            
            # Clear gradients from the previous iteration
            self.optimizer.zero_grad()  
            
            # Forward pass
            output = self.model(image)  

            # Calculate and accumulate the loss
            loss = self.criterion(output, label) 
            train_loss += loss.item()  
            
            # Get the predicted labels and count number of correct predictions
            pred = torch.max(output.data, 1)[1]
            cur_correct = (pred == label).sum().item() 
            
            # Backpropagation; Update the model parameters based on the gradients
            loss.backward()  
            self.optimizer.step()  
            
            # Update total and correct variables 
            total += label.size(0)  
            correct += cur_correct 

        # Compute Training Accuracy and Loss for the epoch
        train_accuracy = correct / total  
        train_loss = train_loss / len(train_loader) 
        
        # Perform validation
        valid_loss, valid_accuracy = self.test(val_loader)  
        
        # Adjust the learning rate based on validation accuracy
        self.scheduler.step(valid_accuracy)  
        
        return train_loss, train_accuracy, valid_loss, valid_accuracy

    def test(self, dataloader):
        """
        Function to test the trained model on the given data loader

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            Testing data loader.

        Returns
        -------
        test_loss: int
            Testing Loss
        test_accuracy: int
            Testing Accuracy
        """

        # Set the model in evaluation mode
        self.model.eval()  

        # Initializing variables to compute loss and accuracy
        test_loss = 0  
        correct = 0  
        total = 0  
        
        # Iterating over the data loader
        for i, data in enumerate(dataloader, 0):
            image, label = data

            # Send images and labels to GPU
            image = image.to(self.device)
            label = label.to(self.device)
            
            # Forward pass
            output = self.model(image)  

            # Calculate and update the loss
            loss = self.criterion(output, label)  
            cur_loss = loss.item()  
            test_loss += cur_loss 
            
            # Get the predicted labels and count number of correct predictions
            pred = torch.max(output.data, 1)[1] 
            cur_correct = (pred == label).sum().item()  
            total += label.size(0)  # Accumulate the total number of samples
            correct += cur_correct  # Accumulate the number of correct predictions
            
        # Compute Testing Accuracy and Loss for the epoch
        accuracy = correct / total  
        test_loss = test_loss / len(dataloader)  
        return test_loss, accuracy

class BasicBlock(nn.Module):
    """
    BasicBlock Class for handling basic blocks of the ResNet along with skip connections.

    Methods
    -------
    forward(x)
        Forward Propagate the given input x through the network.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        BasicBlock init

        Parameters
        ----------
        in_planes: int
            Number of input planes.
        planes: int
            Number of output planes.
        stride: int
            Stride.
        """

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """
        BasicBlock forward

        Parameters
        ----------
        x: torch.Tensor
            Input

        Returns
        -------
        out: torch.Tensor
            Output
        """

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ZigZag_TextResNet(nn.Module):
    """
    Universal Sentence Encoder based ZigZag ResNet Class

    Methods
    -------
    forward(x)
        Forward Propagate the given input x through the network.
    """

    def __init__(self, block, num_blocks, num_classes=2):
        """
        ZigZag_TextResNet init

        Parameters
        ----------
        block: BasicBlock
            BasicBlock
        num_block: list(int)
            Number of repetitions for each block.
        num_classes: int
            Number of classes.
        """

        super(ZigZag_TextResNet, self).__init__()
        self.in_planes = 64
        self.fc = nn.Linear(512, 768)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 64, num_blocks[4], stride=2)
        self.layer6 = self._make_layer(block, 128, num_blocks[5], stride=2)
        self.layer7 = self._make_layer(block, 256, num_blocks[6], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Function to create the sub-layers within a particular block.

        Parameters
        ----------
        block: BasicBlock
            BasicBlock
        planes: int
            Number of output planes.
        num_blocks: int
            Number of repetitions.
        stride: int
            Stride

        Returns
        -------
        nn.Sequential(*layers)
        """

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        ZigZag_TextResNet forward

        Parameters
        ----------
        x: torch.Tensor
            Input

        Returns
        -------
        out: torch.Tensor
            Output
        """

        out = x.to(self.fc.weight.dtype)

        # Passing the input to a fully connected layer
        out = self.fc(out)

        # Reshaping the FC Layer output to 3 16x16 matrices stacked
        out = out.view(-1, 3, 16, 16)

        # Regular ResNet Architecture with modified block ordering
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ZigZag_ResNet(nn.Module):
    """
    Parts of Speech based ZigZag ResNet Class

    Methods
    -------
    forward(x)
        Forward Propagate the given input x through the network.
    """

    def __init__(self, block, num_blocks, num_classes=2):
        """
        ZigZag_ResNet init

        Parameters
        ----------
        block: BasicBlock
            BasicBlock
        num_block: list(int)
            Number of repetitions for each block.
        num_classes: int
            Number of classes.
        """

        super(ZigZag_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 64, num_blocks[4], stride=2)
        self.layer6 = self._make_layer(block, 128, num_blocks[5], stride=2)
        self.layer7 = self._make_layer(block, 256, num_blocks[6], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Function to create the sub-layers within a particular block.

        Parameters
        ----------
        block: BasicBlock
            BasicBlock
        planes: int
            Number of output planes.
        num_blocks: int
            Number of repetitions.
        stride: int
            Stride

        Returns
        -------
        nn.Sequential(*layers)
        """

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        ZigZag_ResNet forward

        Parameters
        ----------
        x: torch.Tensor
            Input

        Returns
        -------
        out: torch.Tensor
            Output
        """

        # Regular ResNet Architecture with modified block ordering
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ZigZagLROnPlateauRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    ZigZagLROnPlateauRestarts Class to define our custom-defined Learning Rate Scheduler based on ReduceLROnPlateau.

    Methods
    -------
    step(metric)
        Return updated learning rate.
    """

    def __init__(self, optimizer, mode='min', lr=0.01, up_factor=1.1, down_factor=0.8, up_patience=10, down_patience=10, restart_after=30, verbose=True):
        """
        ZigZagLROnPlateauRestarts Class init

        Parameters
        ----------
        optimizer: torch.optim
            Learning Rate Optimizer
        mode: str
            Whether to minimize or maximize the metric
        lr: float
            Learning Rate
        up_factor: float
            Factor by which the learning rate will be scaled up.
        down_factor: float
            Factor by which the learning rate will be scaled down.
        up_patience: int
            Number of epochs to wait before scaling up learning rate.
        down_patience: int
            Number of epochs to wait before scaling down learning rate. 
        restart_after: int
            Number of epochs to wait before resetting to best learning rate
        verbose: bool
            Whether to display learning rate progress.
        """

        super(ZigZagLROnPlateauRestarts).__init__()
        self.optimizer = optimizer
        self.mode = mode
        self.up_factor = 1 + up_factor
        self.down_factor = 1 - down_factor
        self.up_patience = up_patience
        self.down_patience = down_patience
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.prev_metric = np.Inf if self.mode == 'min' else -np.Inf
        self.best_lr = lr
        self.restart_after = restart_after
        self.verbose = verbose
        self.num_epochs = 0
        
    def step(self, metric):
        """
        Function which will analyze the given metric and update learning rate accordingly.

        Parameters
        ----------
        metric: float
            Performance Metric
        """

        self.num_epochs += 1

        # If the metric is to be minimized
        if self.mode == 'min':

            # If the current metric is lower than the previous metric
            if metric < self.prev_metric:

                # Setting current learning rate as the best learning rate
                self.best_lr = self.optimizer.param_groups[0]['lr']

                # Updating number of good and bad epochs
                self.num_bad_epochs = 0
                self.num_good_epochs += 1

                # If number of good epochs is greater than up patience
                if self.num_good_epochs > self.up_patience:

                    # Scale up the learning rate
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.up_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"increasing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset number of good epochs.
                    self.num_good_epochs = 0

            # If the current metric is greater than the previous metric
            else:

                # Updating number of good and bad epochs
                self.num_bad_epochs += 1
                self.num_good_epochs = 0

                # If number of bad epochs is greater than down patience
                if self.num_bad_epochs > self.down_patience:

                    # Scale down the learning rate
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.down_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"reducing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset number of bad epochs.
                    self.num_bad_epochs = 0

        # If the metric is to be maximized
        else:

            # If the current metric is greater than the previous metric
            if metric > self.prev_metric:

                # Setting current learning rate as the best learning rate
                self.best_lr = self.optimizer.param_groups[0]['lr']

                # Updating number of good and bad epochs
                self.num_bad_epochs = 0
                self.num_good_epochs += 1

                # If number of good epochs is greater than up patience
                if self.num_good_epochs > self.up_patience:

                    # Scale up the learning rate
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.up_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"increasing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset number of good epochs.
                    self.num_good_epochs = 0

            # If the current metric is lower than the previous metric
            else:

                # Updating number of good and bad epochs
                self.num_bad_epochs += 1
                self.num_good_epochs = 0

                # If number of bad epochs is greater than down patience
                if self.num_bad_epochs > self.down_patience:

                    # Scale down the learning rate
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.down_factor
                    self.optimizer.param_groups[0]['lr'] = new_lr
                    if self.verbose:
                        print(f"reducing learning rate of group 0 to {new_lr:.4e}.")

                    # Reset number of bad epochs.
                    self.num_bad_epochs = 0

        # Set current metric as the previous metric for the next metric
        self.prev_metric = metric
        
        # If restart_after epochs has passed after the last (re)start
        if self.num_epochs % self.restart_after == 0:

            # Set the learning rate to the best learning rate registered
            self.optimizer.param_groups[0]['lr'] = self.best_lr
            if self.verbose:
                print(f"restart: setting learning rate of group 0 to best learning rate value: {self.best_lr:.4e}.")