#Caleb Carr | Caleb.M.Carr-1@ou.edu | 2019
# GPL-3.0-only
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
import time
import random

#Splash screen
print('Greek Prose Trainer and Generator')
print('Caleb Carr | GPL v3 | 2019\n')
purpose = input('Train or Generate: ')
if purpose == 'Train':
    EPOCHS = int(input('Number of Training Epochs: '))
else:
    EPOCHS = 20
# Read, then decode for py2 compat.
text_name = input('File name (without extension): ')
path_to_file = './'+text_name+'.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
# Show how the first 13 characters from the text are mapped to integers
#print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//seq_length
# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
# Batch size
BATCH_SIZE = 64
# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# Length of the vocabulary in chars
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    '''This function creates the keras model.  For this CPU model,
       layers.LSTM is used.  model_gpu.py uses layers.CuDNNLSTM'''
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
vocab_size = len(vocab),
embedding_dim=embedding_dim,
rnn_units=rnn_units,
batch_size=BATCH_SIZE)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints/'+text_name
# Name of the checkpoint files
#uncomment the following and comment the following line if you want to save every single checkpoint model
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

if purpose == 'Train':
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

#Generate text after training!
def generate_text(model, start_string):
  '''Evaluation step (generating text using the learned model)'''
  # Number of characters to generate
  num_generate = 1000
  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  # Empty string to store our results
  text_generated = []
  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = .2
  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions,0)
      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

if purpose == 'Generate':
    generated_text = generate_text(model, start_string=u"Παῦλος ἀπόστολος Χριστοῦ Ἰησοῦ")
    print(generated_text)
    ext = str(random.randint(0,300))
    f = open('./training_checkpoints/'+text_name+'/generated'+ext+'.txt','w')
    f.write(generated_text)
    f.close()
    print('Text written to: ./training_checkpoints/'+text_name+'/generated'+ext+'.txt')
