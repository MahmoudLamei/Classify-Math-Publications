**Topic:** SS24 Assignment 2.5: Classify Math Publications

**Description** This project is a text classification model built to suggest classifications for mathematical publications based on their titles. The model classifies titles into predefined categories which involves training a neural network using PyTorch for language processing, particiluarly for multi-label classifications. 

Background: Publications are selected from publicly accessible dataset (ZBMd) provided by the zipped file contains title of publications and their classifications, which are based on the Mathematics Subject Classification(MSC). Aim of the project is to train a model that predicts up to 5 classifications (as a list of 5 strings) for each title.

**Dependencies**
To install the required libraries, you can use the requirements.txt file that provided. Run the following command to install all dependencies:

`pip install -r requirements.txt `

Main libraries from the Script:

1. 'torch': Library used for build and train the neural network. Includes tensor operations, automatic differentiation and model optimization. 
2. 'torchdata': Used for creating, managing and transforming the dataset to make the data loading efficient.  
3. 'torchtext': This extention is used for processing and preparing the text data. Includes tokenization, vocabulary management and data loading purposes for natural language processing.
4. 'tqdm': Library used for progress tracking during training.

**How to Run the Code**

1. Ensure you have Python 3.8 or higher.
2. Install the dependencies via the command mentioned above.
3. Download the dataset from the assignment repository and place the JSON files in the root directory.
4. Run the training script: `python3 model_trainer2.py`
5. Run the server: `python3 server_interaction.py config.json`
6. If needed run the evaluator: `python3 evaluate_script.py my_test_classifications.json test-without-classifications.json test.json`

The model_trainer.py outputs a model that will be used in the server_interaction.py  to classify new titles. The server_interaction.py is a simple server that accepts new titles and returns the  classifications. The evaluate_script.py is used to evaluate the model's performance on the given test set.


**Repository Structure**
- '.vscode/' is the Visual Studio Code settings and configurations
- '__pycache__/' is the Python cache files for optimized module loading.
- '.gitignore' is specifies files and directories to be ignored by Git.
- 'README.md' is this file, provides information about the project.
- 'config.json' is the configuration file for the agent.
- 'evaluate_script.py' is the script to evaluate the performance of the trained model.
- 'model_trainer.py' is the main script used to train the text classification model using the training data.
- 'my_test_classifications.json' contains the model's predicted classifications for the test dataset.
- 'requirements.txt' defines the external libraries and their versions required to run the code including 'torch', 'torchtext', 'torchdata' and 'tqdm'.
- 'server_interaction.py' is the script to interact with the server for model evaluation.
- 'solution_validation.json' is the validation dataset with classifications.
- 'test-without-classifications.json' is the test dataset without classifications (input for the model).
- 'train2.json' is the training dataset.
- 'validation-without-classifications.json' is the validation dataset without classifications (input for validation).
- 'validation2.json' is the validation dataset with classifications.

Input file format: The training data that consists of 200000 titles. Stored in one file, where each line is a JSON object. Examplary input:
```
{"title": "An extension of Chevalley’s theorem to congruences modulo prime powers", "classifications": ["Number theory", "Group theory and generalizations"]}
```
Output
After training the model, returns the 5 classifications as a list. Such as:
```
[
    {
        "title": "Weakly (1, 3)-semiaffine planes",
        "classifications": [
            "Geometry",
            "Combinatorics",
            "Statistical mechanics, structure of matter",
            "Nonassociative rings and algebras",
            "Computer science"
        ]
    },
    {
        "title": "Superconformal field theory in three dimensions: correlation functions of conserved currents",
        "classifications": [
            "Quantum theory",
            "Several complex variables and analytic spaces",
            "Relativity and gravitational theory",
            "Dynamical systems and ergodic theory",
            "Global analysis, analysis on manifolds"
        ]
    },
```

**Solution Summary**
Model built for multi-label classification, where each publication can have multiple associated classifications. Implementation leverages tokenization, vocabulary building, embedded layers, and a feed-forward neural network to output the classification. The process could be defined as:

1. Data Loading: Titles are tokenized using the 'basic_english' tokenizer from TorchText library. Then, labels are transformed into tensor format to process by neural network.
2. Vocabulary Building: Constructed from the tokenized dataset via 'build_vocab_from_iterator' from TorchText. This allows model to handles the text efficiently. Unknown words are words mapped to special '<unk>' token.
3. Custom Dataset Class: Purpose of the dataset is to handle loading and tokenizing the data. Used for preparing batches during training via the DataLoader.
4. Model Architecture: Simple feed-forward model consists of embedding layer (EmbeddingBag) which processes the tokenized titles. Followed by fully connected layer to output the predicted labels. 
Embedding layer transforms the input tokens into dense vectors. In latter, fully connected layer outputs classifications with the help of sigmoid activation function to handle multi-label classification. 
5. Optimization and Loss: Since this is a multi-label classification problem, the model is trained by binary cross entropy loss(BCELoss). It is important to use BCELoss to get consistent classifications. 'Adam' is used as an optimizer with a learning rate of 0.009.
6. Training: Model trained over 7 epochs. After each epoch, loss is reported. Main idea is make optimizer to adjust parameters over time by minimizing the error which is based on the difference between the predicted and actual values. 'train' function handles the forward and backward propagation, optimizing the model at each batch step. 
7. Model Save: After training, model's parameters saved to the 'text_classification_model.pth' file for evaluation.