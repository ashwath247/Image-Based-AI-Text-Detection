# Importing Required Libraries
import ast
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import tensorflow_hub as hub

class Data():
    """
    Data Class to facilitate conversion of text data into Parts of Speech tagged Image Data.

    Methods
    -------
    get_train_test_val_data(batch_path, split)
        Get the train, test, and validation datasets from the image-based dataset.
    """

    def __init__(self, csv_name =  None):
        """
        Data Class init

        Parameters
        ----------
        csv_name: str
            HC3 Dataset csv file path.
        """
        if csv_name != None:
            self.csv_name = csv_name
            self._read_csv()

        # Load the Tensorflow Universal Sentence Encoder embedding model.
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.use_embed_model = hub.load(module_url)

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

    def process_paragraph(self, paragraph, num_sentences):
        """
        Function to Universal Sentence encode the given paragraph in num_sentences granularity.

        Parameters
        ----------
        paragraph: list(str)
            List of sentences.
        num_sentences: int
            Granularity to extract the paragraph in.

        Returns
        -------
        embeddings: list(int)
            Universal Sentence Encoded embeddings of the given paragraph.
        """

        # Initializing the embedding array
        embeddings = []
        for i in range(0, len(paragraph)-num_sentences):
            mini_paragraph = '\n'.join(paragraph[i:i+num_sentences])

            # Use the Universal Sentence Encoder to get the embedding
            use_embedding = self.use_embed_model([mini_paragraph])

            # Store the embedding values to the embedding array
            embeddings.append(use_embedding[0].numpy().tolist())
        return(embeddings)
    
    def map_values_to_range(self, arr, max_val = 255):
        """
        Function to map the USE-embeddings to the range of 0 to max_val.

        Parameters
        ----------
        arr: list(int)
            USE embedding array.
        max_val: int
            Max Value of the range to map to.

        Returns
        -------
        mapped_values: list(int)
            Scaled USE Embeddings.

        """

        # Compute the minium and maximum element in the array
        minimum = np.min(arr)
        maximum = np.max(arr)

        # Compute the range of the embeddings
        value_range = maximum - minimum

        # Initialize array to store the scaled embeddings
        mapped_values = []

        # Iterating over the embeddings
        for value in arr:

            # Scale the embeddings
            value -= minimum
            value /= value_range
            value *= max_val

            # Update the scaled embeddings
            mapped_values.append(value)
        return mapped_values
        
    def get_embeddings(self, name, num_sentences, limit = 150000):
        """
        Function to get the USE Embeddings for a particular label (AI or Human).

        Parameters
        ----------
        name: str
            'ai' or 'human'
        num_sentences: int
            Sentence granularity.
        limit: int
            Maximum number of datapoints needed.

        Returns
        -------
        embeddings: list(int)
            Universal Sentence Encoded embeddings of the given class (AI or Human).
        """

        # Check if 'ai' or 'human' paragraph needs to be USE encoded
        if name.lower() == 'human':
            paragraphs = self.human_paragraphs
        else:
            paragraphs = self.chatgpt_paragraphs

        # Initializing embedding array
        embeddings = []

        # Iterating over the paragraphs
        for paragraph in tqdm(paragraphs):
            embeddings.extend(self.process_paragraph(paragraph, num_sentences))

            # Stop the embedding process once the limit is reached
            if len(embeddings) >= limit:
                break

        # Clip off the embeddings upto the limit
        embeddings = embeddings[:limit]
        return(embeddings)
    
    def get_train_test_val_data(self, limit_per_class = 150000):
        """
        Function to convert the stored Data Batch Files into train, test and validation sets and return them.

        Parameters
        ----------
        limit_per_class: int
            Maximum number of datapoints needed per class.

        Returns
        -------
        train_set: USEDataset
            Training Set
        test_set: USEDataset
            Testing Set
        val_set: USEDataset
            Validation Set
        """

        # Get embeddings for ChatGPT and Human Generated Paragraphs
        chatgpt_embeddings = self.get_embeddings(name = 'ai', num_sentences = 3)
        human_embeddings   = self.get_embeddings(name = 'human', num_sentences = 3)

        # Initializing and updating labels array
        labels = []
        labels.extend(['ai']*limit_per_class)
        labels.extend(['human']*limit_per_class)

        # Initializing and updating embeddings array
        embeddings = []
        embeddings.extend(chatgpt_embeddings)
        embeddings.extend(human_embeddings)

        # Scale the embeddings to 0 to 255
        embeddings_ = []
        for embedding in embeddings:
            embeddings_.append(self.map_values_to_range(embedding, max_val = 255))
        
        # Convert the embeddings to torch tensor
        tembeddings = torch.from_numpy(np.array(embeddings_))

        # ID the labels
        labels_ = []
        for label in labels:
            if label == 'ai':
                labels_.append(0)
            else:
                labels_.append(1)

        # Load the USE Dataset tensors into a custom-defined PyTorch Dataset
        dataset = USEDataset(tembeddings, labels_)

        # Split the USE Dataset to training, testing, and validation sets
        train_set, test_set = torch.utils.data.random_split(dataset, [250000, 50000])
        train_set, val_set  = torch.utils.data.random_split(train_set, [200000, 50000])

        return(train_set, test_set, val_set)

class USEDataset(Dataset):
    """
    Custom-defined PyTorch Dataset Class to handle USE-embedded tensors.
    """

    def __init__(self, tensors, labels, transform=None):
        """
        USEDataset Class init.

        Parameters
        ----------
        tensors: torch.Tensor
            List of USE-Embeddings converted to torch tensors.
        transform: torchvision.transforms.Compose
            List of transforms to apply on the data, else None
        """

        self.tensors = tensors
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns
        -------
            Number of samples in the USEDataset.

        """

        return len(self.tensors)

    def __getitem__(self, index):
        """
        Function to get the sample in USEDataset at a particular index.

        Parameters
        ----------
        index: int
            Index of the sample desired.
        
        Returns
        -------
        tensor: torch.Tensor
            Scaled USE-embedding tensor array at the index.
        label: torch.Tensor
            Label of the tensor at the index.
        """

        tensor = self.tensors[index]
        label = self.labels[index]
        
        # Apply data transformation if available
        if self.transform is not None:
            tensor = self.transform(tensor)
            
        label = torch.tensor(label)

        return tensor, label
    
class USEInferenceDataset(Dataset):
    """
    Universal Sentence Encoder Dataset for Inference
    """
    def __init__(self, data):
        """
        USEInferenceDataset init

        Parameters
        ----------
        data: torch.Tensor
            An array of user text converted to scaled USE Embeddings
        """
        self.data = data

    def __len__(self):
        """
        Returns
        -------
            Number of samples in the USEDataset.

        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Function to get the sample in USEInferenceDataset at a particular index.

        Parameters
        ----------
        index: int
            Index of the sample desired.
        
        Returns
        -------
        tensor: torch.Tensor
            Scaled USE-embedding tensor array at the index.
        """
        # Convert the sublist to a PyTorch tensor
        tensor = torch.tensor(self.data[index])
        return tensor