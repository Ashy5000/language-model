# %%
!pip install numba

# %%
!pip install dill

# %%
# Imports
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import json
import random
import copy
import math
from numba import jit, cuda
import dill
import sys

# %%
# Function for CPU/GPU speed comparison
def test_gpu_speed():
  # normal function to run on cpu
  def func(a):
    for i in range(10000000):
      a[i]+= 1

  # function optimized to run on gpu
  @jit(target_backend='cuda', forceobj=True)
  def func2(a):
    for i in range(10000000):
        a[i]+= 1
  if __name__=="__main__":
    n = 10000000
    a = np.ones(n, dtype = np.float64)

    start = timer()
    func(a)
    print("without GPU:", timer()-start)

    start = timer()
    func2(a)
    print("with GPU:", timer()-start)

# %%
# Loads dataset
data_file = open("all_scripts_raw.json", "r")
data_string = data_file.read()
data = json.loads(data_string)

# %%
training_set = []
training_data_samples = 10
max_input_length = 10
series_list = list(data.keys())
for i in range(training_data_samples):
  series = series_list[random.randrange(len(series_list))]
  episodes_list = list(data[series].keys())
  episode_name = episodes_list[random.randrange(len(episodes_list))]
  episode_script = data[series][episode_name]
  snippet_start = random.randrange(len(episode_script))
  snippet_end = snippet_start + max_input_length
  episode_snippet = episode_script[snippet_start:snippet_end]
  snippet_known = episode_snippet[0:max_input_length - 2]
  snippet_unknown = episode_snippet[max_input_length - 2]
  training_set.append({"known": snippet_known, "unknown": snippet_unknown})
print(training_set)

# %%
chars = sorted(list(set(data_string)))
text_int_conversion_dict = {"\n": 91, "\r": 92}
int_text_conversion_dict = {91: "\n", 92: "\r"}
index = 0
for char in chars:
  text_int_conversion_dict[char] = index
  int_text_conversion_dict[index] = char
  index += 1
print(text_int_conversion_dict)
int_list = list(int_text_conversion_dict.keys())
max_int = max(int_list)
print("Highest integer: {}".format(max_int))

# %%
# Class for singular neuron in neural network
class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias
  def evaluate(self, inputs):
    index = 0
    inputs_copy = inputs.copy()
    inputs_length = len(inputs_copy)
    while index < inputs_length:
      inputs_copy[index] = inputs_copy[index] * self.weights[index]
      index += 1
    z = sum(inputs_copy) + self.bias
    return (1 / (1 + (math.e) ** -z)) * max_int

# %%
# Class for layer of neural network
class Layer:
  def __init__(self, neurons):
    self.neurons = neurons

  @jit(target_backend='cuda', forceobj=True)
  def evaluate(self, inputs):
    outputs = []
    for neuron in self.neurons:
      outputs.append(neuron.evaluate(inputs))
    return outputs

# %%
# Class for entire neural network
class FeedForwardNetwork:
  def __init__(self, layers):
    self.layers = layers
  def evaluate(self, inputs):
    active_data = inputs
    for layer in self.layers:
      active_data = layer.evaluate(active_data)
    return active_data

# %%
# Class for memory layer
class MemoryLayer:
  def __init__(self, width):
    self.width = width
    self.state = np.ones(width).tolist()
  def evaluate(self, inputs):
    prev_state = self.state
    if len(inputs) != self.width:
      raise Exception("MemoryLayer inputs list: length invalid. Expected: {} Actual: {}".format(self.width, len(inputs)))
    self.state = inputs
    return prev_state

# %%
# Feed forward neural net generator
def generate_neural_net(width, depth, inputs):
  neuron_types = [Neuron(np.zeros(inputs).tolist(), 0), Neuron(np.zeros(width).tolist(), 0)]
  # neuron_types[0] is a prototype neuron for all neurons in the first layer of the neural net. neuron_types[1] is for all other neurons.
  layer_types = [Layer([neuron_types[0]] * width), Layer([neuron_types[1]] * width)]
  # layer_types[0] is a prototype neuron for the first layer in the neural net. layer_types[1] is for all other layers.
  layers = [layer_types[0]] + ([layer_types[1]] * (depth - 1))
  return FeedForwardNetwork(layers)

# %%
# Class for full neural network
class NeuralNet:
  def __init__(self, sections):
    self.sections = sections
  def evaluate(self, inputs):
    active_data = inputs
    for section in self.sections:
      active_data = section.evaluate(active_data)
    return active_data

# %%
conversion_net = None
generation_net = None

# %%
# Creates neural networks
def generate_networks():
  conversion_net_feed_forward = generate_neural_net(max_input_length, 4, max_input_length)
  conversion_net = NeuralNet([conversion_net_feed_forward])

  generation_net_feed_forward_0 = generate_neural_net(20, 10, max_input_length)
  generation_net_feed_forward_1 = generate_neural_net(20, 10, 20)
  generation_net_feed_forward_2 = generate_neural_net(20, 10, 20)

  generation_net_memory_layer_0 = MemoryLayer(20)
  generation_net_memory_layer_1 = MemoryLayer(20)

  generation_net = NeuralNet([generation_net_feed_forward_0, generation_net_memory_layer_0, generation_net_feed_forward_1, generation_net_memory_layer_1, generation_net_feed_forward_2])
  return { "conversion": conversion_net, "generation": generation_net }

# %%
# Loads neural networks
def load_networks():
  with open("/inputs/language-model/model.bin", "br") as model_file:
    unpickler = dill.Unpickler(model_file)
    models_object = unpickler.load()
  return { "conversion": models_object["conversion"], "generation": models_object["generation"] }

# %%
# Functions for converting a character to an integer and vice versa
def char_to_int(char):
  char = text_int_conversion_dict[char]
  return char
def int_to_char(integer):
  if integer in int_text_conversion_dict:
    integer = int_text_conversion_dict[integer]
  else:
    integer = "[unknown character]"
  return integer

# %%
# Functions for converting text to a list of integers and vice versa
def text_to_int(text):
  lowercase = text.lower()
  output = np.zeros(max_input_length)
  for i in range(len(text)):
    output[i] = char_to_int(text[i])
  return output
def int_to_text(arr):
  return "".join(map(int_to_char, arr))
def float_to_text(arr):
  for i in range(len(arr)):
    arr[i] = round(arr[i])
  return int_to_text(arr)

# %%
def get_response(prompt_text):
  prompt_int = text_to_int(prompt_text)
  prompt_converted = conversion_net.evaluate(prompt_int)
  output_int = generation_net.evaluate(prompt_converted)
  output = float_to_text(output_int)
  return output

# %%
# Function for running the user interface
def run_interface():
  for i in range(5):
    prompt_text = input("> ")
    if prompt_text == "exit":
      return
    response = get_response(prompt_text)
    print(response)

# %%
# Run this cell to load the model from model.txt
networks = load_networks()
conversion_net = networks["conversion"]
generation_net = networks["generation"]

# %%
# Cost/loss function (Mean Squared Error)
def squared_error(pred, true):
  squared_error = (pred[0] - true) ** 2
  return squared_error

# %%
# Function for finding derivative of another function
def find_derivative(func, x):
  h = 0.0000001
  return (func(x + h)[0] - func(x)[0]) / h

# %%
def evaluate_inputs_list(model, inputs_list):
  outputs = []
  for inputs in inputs_list:
    outputs.append(model.evaluate(inputs))
  return outputs

# %%
def mean_squared_error(pred, true):
  errors = np.subtract(true, pred)
  squared_errors = np.square(errors)
  return squared_errors.mean()

# %%
# Function that returns a function that takes in inputs and the value for a specific weight in a neural network.
# The weight is set to the second paramter, and the neural net is run with the inputs as the first parameter.
def create_weight_function(model, net, layer, neuron, weight, inputs):
  def weight_function(x):
    model_copy = copy.deepcopy(model)
    model_copy.sections[net].layers[layer].neurons[neuron].weights[weight] = x
    return model_copy.evaluate(inputs)
  return weight_function

# %%
adjustment_limit = 5

# @jit(target_backend='cuda', forceobj=True)
def optimize_weight(model, net, layer, neuron, weight, inputs_list, desired):
  optimal_nudges = []
  index = 0
  for inputs in inputs_list:
    # Creates a weight function for the desired inputs and weight
    weight_function = create_weight_function(model, net, layer, neuron, weight, inputs)
    # Gets value of desired weight
    weight_value = model.sections[net].layers[layer].neurons[neuron].weights[weight]
    # Calculates the derivative of the weight function
    impact = find_derivative(weight_function, weight_value)
    if not impact:
      impact = 1.0
    # Calculates current mean squared error
    model_outputs = model.evaluate(inputs)
    error = squared_error(model_outputs, desired[index])
    # Calculates large nudge
    nudge_large = error / impact
    # Calculates optimal nudge
    nudge_optimal = nudge_large / 5
    if nudge_optimal > adjustment_limit:
      nudge_optimal = adjustment_limit
    elif nudge_optimal < -adjustment_limit:
      nudge_optimal = -adjustment_limit
    # Adds optimal nudge to array of optimal nudges
    optimal_nudges.append(nudge_optimal)
    index += 1
  # Nudges weight value by average of optimal nudges
  final_nudge = sum(optimal_nudges) / len(optimal_nudges)
  model.sections[net].layers[layer].neurons[neuron].weights[weight] += final_nudge

# %%
def run_epoch(data, model):
  print("Running epoch...")
  inputs_list = []
  desired = []
  for chunk in data:
    inputs_list.append(text_to_int(chunk["known"]))
    desired.append(char_to_int(chunk["unknown"]))
  print("Dataset compiled.")
  print("Optimizing model...")
  for i in range(len(model.sections)):
    print("Optimizing section...")
    section = model.sections[i]
    if not hasattr(section, "layers"):
      continue
    for j in range(len(section.layers)):
      print("Optimizing layer...")
      layer = section.layers[j]
      sys.stdout.write("Neurons optimized: ")
      for k in range(len(layer.neurons)):
        neuron = layer.neurons[k]
        for l in range(len(neuron.weights)):
          optimize_weight(model, i, j, k, l, inputs_list, desired)
        sys.stdout.write("|")
      print(" Layer optimized.")
    print("Section optimized.")
  print("Current loss: {}".format(mean_squared_error(desired, evaluate_inputs_list(model, inputs_list))))

# %%
def save_models():
  with open("/outputs/language-model/model.bin", "bw") as model_file:
    pickler = dill.Pickler(model_file)
    models_string = pickler.dump({ "conversion": conversion_net, "generation": generation_net })
    print("Models successfully saved.")

# %%
def train(data, model, epochs):
  for i in range(epochs):
    run_epoch(data, model)
    save_models()
    print("{}/{} epochs completed.".format(i + 1, epochs))

# %%
train(training_set, generation_net, 1)

# %%
save_models()