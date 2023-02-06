from model import NLPModel
from dataset import get_data
import nltk

import torch
from torch import nn

# train function
def train(epochs: int, lr: float):
    #device-agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to {device}")
    
    # nltk requires punkt resource, if we run it at first - downloading
    try:
        dataloader, vocabulary, tags = get_data(batch_size=2)
    except LookupError:
        print("----- Resource punkt is not found, installing -----")
        nltk.download("punkt")
        
        dataloader, vocabulary, tags = get_data(batch_size=2)
    
    # initializing our model
    model = NLPModel(input_size=len(vocabulary), hidden_size=16, output_size=len(tags)).to(device)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for _, (sentence, label) in enumerate(dataloader):
            # setting our data to device
            sentence, label = sentence.to(device), label.to(device).long()
            
            prediction = model(sentence)
            
            loss = loss_func(prediction, label.long())
            
            # standart learning proccess
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # print out stat on each 100 epochs  
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")

    # collecting useful data
    data_to_save = {
        "model_state_dict": model.state_dict(),
        "input_size": len(vocabulary),
        "hidden_size": 16,
        "output_size": len(tags),
        "vocabulary": vocabulary,
        "tags": tags
    }
    
    torch.save(data_to_save, "nlp_model/checkpoint/nlp_model.pth")

if __name__ == "__main__":
    train(epochs=2000, lr=0.001)