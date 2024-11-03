
"""
To use this implementation, you simply have to implement `get_classifications` such that it returns classifications.
You can then let your agent compete on the server by calling

    python3 server_interaction.py path/to/your/config.json
"""
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import json
import logging
import requests
import time
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator



class TextClassificationModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super(TextClassificationModel, self).__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = torch.nn.Linear(embed_dim, num_labels)

    def forward(self, text, offsets=None):  # Offsets are optional now
        # If offsets are provided (i.e., batch of varying lengths), use them
        if offsets is not None:
            embedded = self.embedding(text, offsets)
        else:  # Otherwise, treat it as a batch of fixed-length sequences or single input
            embedded = self.embedding(text)
        return torch.sigmoid(self.fc(embedded))

def get_classifications(titles):
    """Given a list of titles, return the 5 most probable classifications for each title."""
    
    # Load device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model
    model = TextClassificationModel(21400, 128, 63) 
    model.load_state_dict(torch.load('text_classification_model.pth'))
    model.eval()
    model.to(device)
    
    # Load tokenizer and classification labels
    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data_iter):
        for title in data_iter['title']:
            yield tokenizer(title)

    # Build the vocabulary from the dataset
    df_train = pd.read_json("train2.json")
    vocab = build_vocab_from_iterator(yield_tokens(df_train), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    classifications = sorted(set([item for sublist in df_train['classifications'] for item in sublist]))
    
    # Preprocess each title (tokenization and tensor conversion)
    results = []
    for title in titles:
        # Tokenize the title
        tokens = tokenizer(title)
        tokens_tensor = torch.tensor(vocab(tokens), dtype=torch.int64).unsqueeze(0).to(device)  # 1D tensor for single sentence
        
        with torch.no_grad():
            output = model(tokens_tensor)
            predicted_probs = output.squeeze().cpu().numpy()  # Get probabilities for each classification
        
        # Get indices of the top 5 classifications
        top5_indices = predicted_probs.argsort()[-5:][::-1]  # Indices of top 5 classifications
        
        # Convert indices to classification labels
        top5_classifications = [classifications[idx] for idx in top5_indices]
        results.append(top5_classifications)
    
    return results


def run(config_file, action_function, parallel_runs=True):
    logger = logging.getLogger(__name__)

    with open(config_file, 'r') as fp:
        config = json.load(fp)

    actions = []
    for request_number in range(51):    # 50 runs are enough for full evaluation. Running much more puts unnecessary strain on the server's database.
        logger.info(f'Iteration {request_number} (sending {len(actions)} actions)')
        # send request
        response = requests.put(f'{config["url"]}/act/{config["env"]}', json={
            'agent': config['agent'],
            'pwd': config['pwd'],
            'actions': actions,
            'single_request': not parallel_runs,
        })
        if response.status_code == 200:
            response_json = response.json()
            for error in response_json['errors']:
                logger.error(f'Error message from server: {error}')
            for message in response_json['messages']:
                logger.info(f'Message from server: {message}')

            action_requests = response_json['action-requests']
            if not action_requests:
                logger.info('The server has no new action requests - waiting for 1 second.')
                time.sleep(1)  # wait a moment to avoid overloading the server and then try again
            # get actions for next request
            actions = []
            for action_request in action_requests:
                actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
        elif response.status_code == 503:
            logger.warning('Server is busy - retrying in 3 seconds')
            time.sleep(3)  # server is busy - wait a moment and then try again
        else:
            # other errors (e.g. authentication problems) do not benefit from a retry
            logger.error(f'Status code {response.status_code}. Stopping.')
            break

    print('Done - 50 runs are enough for full evaluation')


if __name__ == '__main__':
    import sys
    run(sys.argv[1], get_classifications)
