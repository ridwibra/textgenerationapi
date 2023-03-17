from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware


from keras.layers import Embedding, GRU, Dense
from keras.models import Sequential
import tensorflow as tf
import numpy as np
from keras.losses import sparse_categorical_crossentropy

app = FastAPI()

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    # "http://localhost",
    #"http://localhost:3000",
    "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text = open('shakespeare.txt', 'r').read()
vocab = sorted(set(text))
char_to_ind = {char: ind for ind, char in enumerate(vocab)}
ind_to_char = np.array(vocab)


def sparse_cat_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,
                        batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons, return_sequences=True, stateful=True,
                  recurrent_initializer='glorot_uniform', reset_after=True))
    model.add(Dense(vocab_size))
    model.compile('adam', loss=sparse_cat_loss)
    return model


model = create_model(vocab_size=84, embed_dim=64,
                     rnn_neurons=1026, batch_size=1)

model.load_weights('shakespeare_model_weights.h5')
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_seed, gen_size=500, temp=1.0):
    # number to generate
    num_generate = gen_size
    # evaluate the input text and convert the text to index
    input_eval = [char_to_ind[s] for s in start_seed]
    # expand it to meet the batch format shape
    input_eval = tf.expand_dims(input_eval, 0)
    # holds the generated text
    text_generated = []
    # how surprising you want the results to be
    temperature = temp
    # reset the state of the model
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch shape dimension
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(ind_to_char[predicted_id])
    return (start_seed + "".join(text_generated))



@app.get('/')
async def root():
    return "Welcome"


@app.get('/generate')
# async def receive(seed: str = Form(...), length: int = Form(...)):
async def receive(seed: str, length: int):
    # print(seed)
    # print(length)
    result = generate_text(model, seed, gen_size=length)
    # print(result)
    return result
