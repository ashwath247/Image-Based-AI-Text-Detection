# Importing required libraries
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from model import Model
from data_pos import POSTags, POSImageDataset

class Inference():
    """
    Inference Class to execute model inference on user input.

    Methods
    -------
    infer()
        Model Inference

    get_sentiment(user_input)
        Classify whether the given user text input is AI or Human generated.
    """

    def __init__(self, model):
        """
        Inference init.

        Parameters
        ----------
        model: ResNet Model
        """

        # Initializing Parts of Speech Tags object
        self.pos_obj = POSTags()

        self.model = model

    def infer(self):
        """
        Function to execute model inference
        """

        # MODEL INFERENCE; Enter a text to check if AI-Generated
        while True:

            # Question Prompt
            text = input("Enter a text to classify its sentiment (type 'quit' to exit): ")

            # If 'quit' then quit
            if text.lower() == 'quit':
                break

            # Get the prediction of whether it is AI-Generated or Human-Generated
            sentiment = self.get_sentiment(text)
            print(f"Sentiment: {sentiment}")

    def get_sentiment(self, user_input):
        """
        Function to get the sentiment of the User Prompt

        Parameters
        ----------
        user_input: str
            User Input prompt, minimum 3 sentences.

        Returns
        -------
            A string representing percentages of confidence of AI or Human content.
        """

        # Create the directory .user for storing the image-embeddings
        if not os.path.isdir('.user'):
            os.mkdir('.user')

        # Process the user input
        user_input = ''.join(user_input).replace('\n', '').split('.')
        try:
            user_input.remove('')
        except:
            pass

        itr = 1
        # If the user input has more than 3 sentences
        if len(user_input) >= 3:
            
            # Iterating over the user input paragraph in 3 sentences granularity.
            for i in range(len(user_input) - 2):

                # Get the POS Tags and corresponding lengths for user sentences i, i+1, and i+2
                arr1, arr1_len = self.pos_obj.get_tags(user_input[i]) 
                arr2, arr2_len = self.pos_obj.get_tags(user_input[i + 1]) 
                arr3, arr3_len = self.pos_obj.get_tags(user_input[i + 2]) 

                # Compute the maximum and minimum length of the POS-embeddings generated
                max_len = max([arr1_len, arr2_len, arr3_len])
                min_len = min([arr1_len, arr2_len, arr3_len])

                # Pad the POS-embeddings with 0 to match the 
                arr1.extend([0]*(max_len - arr1_len))
                arr2.extend([0]*(max_len - arr2_len))
                arr3.extend([0]*(max_len - arr3_len))

                # Stack the arrays vertically
                data = np.vstack([arr1, arr2, arr3])

                # Create a contour plot
                plt.contourf(data, cmap='hsv')

                # Set Plot Axis as OFF
                plt.axis('off')

                # Save inference embedding figure
                plt.savefig(f"./.user/user_{itr}.png",bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()

                itr += 1

            # Set the root directory of your images
            root_dir = "./.user/"

            # Check if CUDA is available and use GPU if possible
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Define the transform to apply on the images
            transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])

            # Create an instance of the custom POS Image dataset
            dataset = POSImageDataset(root_dir, transform=transform)

            # Creating the dataloader from the dataset with batch size as 1
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            # Evaluate the model
            self.model.to(device)
            self.model.eval()

            # Initializing variable counters
            count = 0
            ai_perc = 0
            human_perc = 0

            # Disabling torch gradient calculation
            with torch.no_grad():

                # Iterating over the user input dataloader
                for i, (image, _) in enumerate(dataloader, 0):
                    count += 1

                    # Transfering the images to CUDA
                    image = image.to(device)

                    # Getting output from the last layer of the model
                    output = self.model(image)

                    # Computing the overall and class probabilities
                    probabilities = torch.sigmoid(output)
                    ai_probability = probabilities.detach().cpu().numpy()[0][0]
                    human_probability = probabilities.detach().cpu().numpy()[0][1]
                    total_probability = ai_probability + human_probability

                    # Computing the percentage confidence
                    ai_percentage = ai_probability / total_probability
                    human_percentage = human_probability / total_probability
                    ai_perc += ai_percentage
                    human_perc += human_percentage

                    # Inferring the per-sample predicted class label
                    class_labels = {0: 'AI', 1: 'Human'}
                    predicted_class = torch.argmax(probabilities).item()

            
            # Removing the directory contatining images pertaining to user-entered input text
            shutil.rmtree(root_dir)
            return(f"AI: {round((ai_perc/count)*100, 3)}%; Human: {round((human_perc/count)*100, 3)}%")
        else:
            return('Please enter a larger paragraph (minimum: 3 sentences)')