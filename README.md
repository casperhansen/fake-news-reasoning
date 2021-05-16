
# Automatic Fake News Detection: Are Models Learning to Reason

Casper Hansen, Christian Hansen, Lucas Chaves Lima (2021). Automatic Fake News Detection: Are Models Learning to Reason? ACL-IJCNLP 2021
 <br>
1. Make sure to download the dataset (https://www.dropbox.com/s/3v5oy3eddg3506j/multi_fc_publicdata.zip?dl=0), and place in the same directory as code-acl. 
2. Install a python environment with the required packages (requirements.txt)
3. replace the model_selection.py from the hypopt packages with model_selection.py provided in the source code. (The original hypopt code contains a bug).
4. When you run the code for the first time, it might take some time to download pretrained language models and GloVe word embeddings.
5. You can run the RF, LSTM, and BERT models using the examples below (it runs the models on Snopes using only claims as input)
	- RF: python main.py --dataset snes --inputtype CLAIM_ONLY --model bow
	- LSTM: python main.py --dataset snes --inputtype CLAIM_ONLY --model lstm --batchsize 16 --lr 0.0001 --lstm_hidden_dim 128 --lstm_layers 2 --lstm_dropout 0.1
	- BERT: python main.py --dataset snes --inputtype CLAIM_ONLY --model bert --batchsize 8 --lr 3e-6
6. You can create the table and plots from the paper based on your results by running: python analyze.py



