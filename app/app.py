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
    # return '<h1>Hello from Flask & Docker</h2>'
    return render_template("index.html")

@app.route('/search', methods=['POST'])
def search():

    prompt = request.form['query'].strip()

    # first computer sentence vector for a given query.
    # qwords = query.split(" ")

    # qwords_embeds = np.array([model.get_embed(word) for word in qwords])
    
    # qsentence_embeds = np.mean(qwords_embeds, 0)

    max_seq_len = 30
    seed = 0

    #smaller the temperature, more diverse tokens but comes 
    #with a tradeoff of less-make-sense sentence
    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    temperature = 0.5
    # for temperature in temperatures:
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
                        vocab, device)
    print(" ".join(generation))
    generation = " ".join(generation)
    # print(str(temperature)+'\n'+' '.join(generation)+'\n')

    # corpus_embeds = []
    # for each_sent in corpus:
    #     words_embeds = np.array([model.get_embed(word) for word in each_sent])
    #     sentence_embeds = np.mean(words_embeds, 0)
    #     corpus_embeds.append(sentence_embeds)

    # corpus_embeds = np.array(corpus_embeds)

    # result_idxs = find_closest_indices_cosine(corpus_embeds, qsentence_embeds)

    # result = []
    # for idx in result_idxs:
    #     result.append(' '.join(corpus[idx]))

    return render_template('index.html', result = generation, old_query = prompt)


port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)