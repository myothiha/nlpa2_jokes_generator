from flask import Flask, render_template, request
import numpy as np
import pickle
import torch
from models.classes import LSTMLanguageModel
from library.utils import generate
import torchtext

app = Flask(__name__)

# Loading the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meta = pickle.load(open('models/meta.pkl', 'rb'))
args = meta['args']
vocab = meta['vocab']
model      = LSTMLanguageModel(**args).to(device)
model.load_state_dict(torch.load('models/best-val-lstm_lm.pt',  map_location=device))
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate_jokes():

    # get prompt from HTML form.
    prompt = request.form['query'].strip()

    max_seq_len = 30
    seed = 0

    #smaller the temperature, more diverse tokens but comes 
    #with a tradeoff of less-make-sense sentence
    temperature = 0.8 # generate response with slightly creativity.
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
                        vocab, device)

    generation = " ".join(generation)

    return render_template('index.html', result = generation, old_query = prompt)


port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)