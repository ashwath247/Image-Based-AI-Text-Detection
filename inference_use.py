# Importing required libraries
import torch
from torch.utils.data import Dataset, DataLoader

from model import Model
from data_use import Data

# Loading the Text-Based ZigZag ResNet Model
model = Model(model_name = 'zigzag_textnet')

# Run the below cell after loading the best model 
#model.load_state_dict(torch.load('./models/best_model.pth'))

# Loading the Data Class
data_obj = Data()

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
    
def get_sentiment(user_input):
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

    # Process the user input
    user_input = ''.join(user_input).replace('\n', '').split('.')

    # USE Encode the user input and scale them to 0 to 255 range
    user_input_embeddings = data_obj.process_paragraph(user_input, num_sentences = 3)
    scaled_user_embeddings = [data_obj.map_values_to_range(user_input_embedding, max_val = 255) for user_input_embedding in user_input_embeddings]

    # Load the embeddings into the USEInferenceDataset and generate the dataloader with batch size as 1
    dataset = USEInferenceDataset(scaled_user_embeddings)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate the model
    model.to(device)
    model.eval()
    
    # Initializing variable counters
    count = 0
    ai_perc = 0
    human_perc = 0

    # Disabling torch gradient calculation
    with torch.no_grad():

        # Iterating over the user input dataloader
        for i, image in enumerate(dataloader, 0):
            count += 1

            # Transfering the images to CUDA
            image = image.to(device)

            # Getting output from the last layer of the model
            output = model(image)

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

    return(f"AI: {round((ai_perc/count)*100, 3)}%; Human: {round((human_perc/count)*100, 3)}%")

# MODEL INFERENCE; Enter a text to check if AI-Generated
while True:

    # Question Prompt
    text = input("Enter a text to classify its sentiment (type 'quit' to exit): ")

    # If 'quit' then quit
    if text.lower() == 'quit':
        break

    # Get the prediction of whether it is AI-Generated or Human-Generated
    sentiment = get_sentiment(text)
    print(f"Sentiment: {sentiment}")