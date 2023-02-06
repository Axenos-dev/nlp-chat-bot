import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from nlp_model.model import NLPModel
from nlp_model.utils import tokenize_sentence, bag_of_words
import nltk
import json
import random
import torch

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

with open("intents/intents.json", "r") as f:
    intents = json.load(f)

DATA = torch.load("nlp_model/checkpoint/nlp_model.pth")
MODEL = NLPModel(DATA['input_size'], DATA['hidden_size'], DATA['output_size'])
MODEL.load_state_dict(DATA['model_state_dict'])
MODEL.eval()

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = get_response(text)
        output.append(response)

    return SimpleText(dict(text=output))

# function for getting response
def get_response(question: str) -> str:
    # proccessing data requires punkt, so if not found - installing
    try:
        preproccessed_question = preproccess_question(question)
    except LookupError:
        print("-----Punkt not found, installing-----")
        nltk.download("punkt")
        
        preproccessed_question = preproccess_question(question)
    
    # getting prediction to all classes
    with torch.inference_mode():
        predictions = MODEL(preproccessed_question)
    
    predictions = torch.softmax(predictions, dim=1)
    # recieving the maximum probabily from predictions and index of element
    probability, index = torch.max(predictions, dim=1)
    
    # if probaility too low, this mean that model didn`t understand the question well
    if probability.item() < 0.8:
        return "I dont know answer on this question"
    
    # picking the random answer from intents
    response = random.choice(intents['intents'][index.item()]['responses'])
    
    return response

# function for preproccessing a quesion to bag of words
def preproccess_question(question: str) -> torch.Tensor:
    bow = bag_of_words(tokenize_sentence(question), DATA['vocabulary'])
    bow = bow.reshape(1, bow.shape[0])
    
    return torch.from_numpy(bow)
    