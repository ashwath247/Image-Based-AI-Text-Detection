# Importing Required Libraries
import ast
import glob
import multiprocessing
import nltk
import os

import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm

# Argument to suppress Matplotlib Plotting to Console
import matplotlib
matplotlib.use('Agg')

class Data():
    """
    Data Class to facilitate conversion of text data into Parts of Speech tagged Image Data.

    Methods
    -------
    save_pos_tagged_images(name, images_dir)
        Save the POS-tagged image-based text embeddings.

    save_torch_data_batches(folder_path)
        Save the PyTorch Data Batch Files of generated images.

    get_train_test_val_data(batch_path, split)
        Get the train, test, and validation datasets from the image-based dataset.
    """
    
    def __init__(self, csv_name):
        """
        Data Class init

        Parameters
        ----------
        csv_name: str
            HC3 Dataset csv file path.
        """
       
        self.csv_name = csv_name

        # Calling the _read_csv function
        self._read_csv()

    def _read_csv(self):
        """
        Function to read the HC3 Dataset csv and process into human and chatgpt paragraphs.
        """
        
        # Reading the HC3 Dataset
        self.df = pd.read_csv(self.csv_name)

        # Create a list of lists of strings, where each inner list represents a paragraph of human/chatgpt answers.
        self.human_paragraphs = [''.join(ast.literal_eval(human_paragraph)).replace('\n', '').split('.') 
                                for human_paragraph in list(self.df['human_answers'])]
        self.chatgpt_paragraphs = [''.join(ast.literal_eval(chatgpt_paragraph)).replace('\n', '').split('.') 
                                for chatgpt_paragraph in list(self.df['chatgpt_answers'])]
        
    def _cpu_thread_worker(self, paragraph, itr, name):
        """
        Function to process a paragraph and save into corresponding image embedddings.

        Parameters
        ----------
        paragraph: list(str)
            String array consisting of several related sentences.

        itr: int
            Paragraph iteration ID.

        name: str
            'ai' or 'human'
        """
        
        # Ignore if the number of sentences in the paragraph is less than 3
        if len(paragraph) < 3:
            return
        
        # Initializing the arrays to store POS-Tags and its corresponding lengths.
        arrs = []
        arr_lens = []

        mini_itr = 1
        # Iterating over the paragraph
        for i in range(len(paragraph)):

            # Extracting the current sentence
            sentence = paragraph[i] + '.'

            # POS-Tagging the sentence
            arr, arr_len = self.pos_obj.get_tags(sentence)

            # Append metadata to corresponding arrays
            arrs.append(arr)
            arr_lens.append(arr_len)

        # Iterate over the paragraph in the following sequence
        # 1->2->3 then 2->3->4 then 3->4->5, etc
        for i in range(0, len(paragraph) - 2):
            try:
                # Computing the smallest sentence in the set of three sentences
                min_len = min(arr_lens[i:i+3])
                # Computing the largest sentence in the set of three sentences
                max_len = max(arr_lens[i:i+3])

                # Initializing array to store padded sentences
                arrs_ = []
                for j in range(3):
                    arr = arrs[i+j].copy()

                    # Padding all the 3 sentennces with 0s to make them equally long as the longest sentence.
                    arr.extend([0]*(max_len - arr_lens[i+j]))

                    # Append the padded embeddings.
                    arrs_.append(arr)

                # Stack the arrays vertically
                data = np.vstack(arrs_)
                
                # Create a contour plot
                plt.contourf(data, cmap='hsv')

                # Set Plot Axis as OFF
                plt.axis('off')
                plt.savefig(f"{self.images_dir}{name}/{name}_{itr}_{mini_itr}.png",bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
                
                # Update the mini-iteration
                mini_itr += 1

            except Exception as e:
                print(e)     

    def save_pos_tagged_images(self, name, images_dir):
        """
        Function to save the POS-Tagged Images for a particular label (AI or Human).

        Parameters
        ----------
        name: str
            'human' or 'ai'
        images_dir: str
            Directory to store the images.
        """
        
        # Set the data store path.
        self.images_dir = images_dir

        # If the name is "human", use the human paragraphs. Otherwise, use the chatgpt paragraphs.
        if name.lower() == "human":
            paragraphs = self.human_paragraphs
            name = name.lower()
        else:
            paragraphs = self.chatgpt_paragraphs
            name = "ai"

        # Create the Image Directory if not present
        try:
            os.mkdir(f"{self.images_dir}")
        except:
            pass
        try:
            os.mkdir(f"{self.images_dir}/{name}")
        except:
            pass

        # Create an instance of the POSTags class.
        self.pos_obj = POSTags()

        itr = 1

        # Create a multiprocessing pool.
        with multiprocessing.Pool() as pool:
            # Create a list to store the results of the CPU thread worker.
            results = []

            # Iterate over the paragraphs and save them to files.
            for paragraph in tqdm(paragraphs):
                # Apply the CPU thread worker to the paragraph.
                result = pool.apply_async(self._cpu_thread_worker, args=(paragraph, itr, name))
                results.append(result)
                itr += 1

            # Iterate over the results and get the images.
            for result in tqdm(results):
                result.get()

    def save_torch_data_batches(self, folder_path):
        """
        Function to convert the stored POS-Tagged Images into corresponding PyTorch Batch Tensors
        for easier data loading for training the model.

        Parameters
        ----------
        folder_path: str
            Directory to store the Data Batch Files.
        """

        # Set the data store path.
        self.data_store_path = folder_path

        # Create the Data Store Directory if not present
        try:
            os.mkdir(f"{self.data_store_path}")
        except:
            pass
        try:
            os.mkdir(f"{self.data_store_path}/batches/")
        except:
            pass

        # Define the transform to apply on the images.
        transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])

        # Create an instance of the custom dataset.
        dataset = POSImageDataset(self.data_store_path, transform=transform)

        # Create a DataLoader to handle batch loading.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=True)

        # Create a list to store batch class labels.
        batch_labels = []

        # Iterate over the DataLoader and save the batches to files.
        for batch_idx, (batch, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Save the batch to a file.
            torch.save(batch, f"{self.data_store_path}/batches/data_batch_{batch_idx+1}.pt")

            # Add the batch labels to the list.
            batch_labels.extend(labels.tolist())

        # Write the batch labels to a META file.
        with open(f"{self.data_store_path}/batches/batches.META", "w") as f:
            for label in batch_labels:
                f.write(f"{label}\n")


    def get_train_test_val_data(self, batch_path=None, split=[0.8, 0.1, 0.1]):
        """
        Function to convert the stored Data Batch Files into train, test and validation sets and return them.

        Parameters
        ----------
        batch_path: str
            Directory where the Data Batch Files are stored.
        split: list(int)
            Train, Test, and Validation ratios.

        Returns
        -------
        train_set: POSImageTensorDataset
            Training Set
        test_set: POSImageTensorDataset
            Testing Set
        val_set: POSImageTensorDataset
            Validation Set
        """

        # If the batch path is not given, use the default batch directory.
        if batch_path is None:
            batch_path = f"{self.data_store_path}/batches/"

        # Get the list of .pt files in the batch directory.
        file_paths = glob.glob(batch_path + '.pt')

        # Load the META file to get the labels.
        batch_labels = []
        with open(f"{batch_path}batches.META", "r") as f:
            for line in f:
                label = int(line.strip())
                batch_labels.append(label)

        # Load the tensors from the .pt files and concatenate them.
        all_tensors = []
        for file_path in file_paths:
            tensor = torch.load(file_path, map_location=torch.device('cpu'))
            all_tensors.append(tensor)
        all_tensors = torch.cat(all_tensors)

        # Create an instance of the custom dataset with data normalization applied.
        mean = [0.0028, 0.0024, 0.0006]
        std = [0.0014, 0.0010, 0.0011]
        transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

        # Create the training, testing, and validation datasets.
        dataset = POSImageTensorDataset(all_tensors, batch_labels, transform)
        train_set, test_set = torch.utils.data.random_split(dataset, [(split[0]+split[2])*len(dataset), split[1]*len(dataset)])
        train_set, val_set  = torch.utils.data.random_split(train_set, [split[0]*len(dataset), split[2]*len(dataset)])

        return (train_set, test_set, val_set)

class POSImageTensorDataset(Dataset):
    """
    Custom-defined PyTorch Dataset Class to handle POS-Tagged images.
    """

    def __init__(self, images, labels, transform=None):
        """
        POSImageTensorDataset Class init.

        Parameters
        ----------
        images: torch.Tensor
            List of POS-Tagged images converted to torch tensors.
        transform: torchvision.transforms.Compose
            List of transforms to apply on the data, else None
        """

        # The tensor of images.
        self.images = images

        # The tensor of labels.
        self.labels = labels

        # A transform that is applied to each image.
        self.transform = transform

    def __len__(self):
        """
        Returns
        -------
            Number of samples in the POSImageTensorDataset.

        """

        return len(self.images)

    def __getitem__(self, index):
        """
        Function to get the sample in POSImageTensorDataset at a particular index.

        Parameters
        ----------
        index: int
            Index of the sample desired.
        
        Returns
        -------
        image: torch.Tensor
            Image at the index.
        label: torch.Tensor
            Label of the Image at the index.
        """

        # The image and label at the given index.
        image = self.images[index]
        label = self.labels[index]

        # Move the image tensor to CPU and convert to NumPy ndarray.
        image = image.cpu().numpy()

        # Convert NumPy ndarray to PIL Image.
        image = Image.fromarray(np.uint8(image), mode='RGB')

        # Apply data transformation if available.
        if self.transform is not None:
            # Convert PIL Image to NumPy ndarray.
            image = np.array(image)
            # Convert ndarray to Tensor.
            image = torch.from_numpy(image)
            image = self.transform(image)

        # Return the image and its label.
        return image, label
    
class POSImageDataset(Dataset):
    """
    Custom-defined PyTorch Dataset Class to convert POS-Tagged images torch Dataset.
    """

    def __init__(self, root_dir, transform=None):
        """
        POSImageTensorDataset Class init.

        Parameters
        ----------
        root_dir: str
            Directory where the POS-tagged images are stored.
        transform: torchvision.transforms.Compose
            List of transforms to apply on the data, else None
        """

        self.root_dir = root_dir

        # A transform that is applied to each image.
        self.transform = transform

        # The list of image files.
        self.image_files = glob.glob(root_dir + '**/*.png', recursive=True)

        # The list of classes.
        self.classes = sorted(set([image_file.split(os.sep)[-2] for image_file in self.image_files]))

        # A dictionary that maps classes to their corresponding indices.
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        """
        Returns
        -------
            Number of samples in the POSImageDataset.

        """

        return len(self.image_files)

    def __getitem__(self, index):
        """
        Function to get the sample in POSImageDataset at a particular index.

        Parameters
        ----------
        index: int
            Index of the sample desired.
        
        Returns
        -------
        image: torch.Tensor
            Image at the index.
        label: torch.Tensor
            Label of the Image at the index.
        """

        # The index of the image file.
        image_file = self.image_files[index]

        image_path = image_file 
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[image_file.split(os.sep)[-2]]

        # If a transform is defined, apply it to the image.
        if self.transform is not None:
            image = self.transform(image)

        # Return the image and its label.
        return image, label
            
class POSTags:
    """
    Class to handle mapping of nltk-returned POS Tags to corresponding ids.
    """

    def __init__(self):
        """
        POSTags Class init.
        """

        self.pos_dict = {'CC': 1, 'CD': 2, 'DT': 3, 'EX': 4, 'FW': 5, 'IN': 6, 'JJ': 7, 'JJR': 8,
                         'JJS': 9, 'LS': 10, 'MD': 11, 'NN': 12, 'NNS': 13, 'NNP': 14, 'NNPS': 15,
                         'PDT': 16, 'POS': 17, 'PRP': 18, 'PRP$': 19, 'RB': 20, 'RBR': 21, 'RBS': 22,
                         'RP': 23, 'SYM': 24, 'TO': 25, 'UH': 26, 'VB': 27, 'VBD': 28, 'VBG': 29,
                         'VBN': 30, 'VBP': 31, 'VBZ': 32, 'WDT': 33, 'WP': 34, 'WP$': 35, 'WRB': 36}

    def get_tags(self, sentence):
        """
        Function to get the POS Tags for a particular sentence.

        Parameters
        ----------
        sentence: str
            Sentence string.

        Returns
        -------
        pos_tags: list(int)
            POS-tag embedding array for the given sentence.

        len_pos: int
            Length of POS Tag array.

        """

        # Tokenize the sentence into words and POS tags.
        words_and_tags = nltk.pos_tag(nltk.word_tokenize(sentence))

        # Get the POS tags for each word.
        pos_tags = [self.pos_dict.get(tag, 0) for word, tag in words_and_tags]

        # Remove any POS tags that are not defined.
        pos_tags = [pos_tag_ for pos_tag_ in pos_tags if pos_tag_ != 0]
        len_pos = len(pos_tags)

        return pos_tags, len_pos