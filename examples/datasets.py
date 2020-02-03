# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets used in examples."""


import array
import collections
import gzip
import os
from os import path
import re
import struct
import string
import urllib.request

import jax.numpy as jnp
import numpy as np


_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, _DATA))
  else:
    print("file {} already exists".format(filename))


def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def large_movie_review_dataset(
    vocab_size,
    min_word_len=3,
    words_per_review=30,
    preprocess=False):
    """Downloads the Large Move Review dataset described here 
       https://ai.stanford.edu/~amaas/data/sentiment/
    
    Args:
        vocab_size: integer indicting the maximum words to store in vocabulary.
                    The used words will be the ones that are more frequent
        min_word_len: Minimum length to keep a word. Only takes effect if
                      preprocess == true
        words_per_review: Maximum words per review. Only takes effect if
                          preprocess == true
        preprocess: whether or not to preprocess the dataset.
                    If preprocess is false then the raw dataset (text) 
                    is returned, otherwise the texts are encoded using
                    a simple word2idx
    
    Examples:

        >>> from examples import datasets
        >>> # (vocab_size, min_word_len, words_per_review, preprocess)
        >>> ds = datasets.large_movie_review_dataset(100, 3, 30, True)
        >>> (x_train, y_train), (x_test, y_test) = ds
        >>> x_train.shape
        (25000, 30)
        >>> y_train.shape
        (25000,)

    Returns:
        (x_train, y_train), (x_test, y_test)
        In case preprocess == true returns text already parsed as integers and
        with numpy array format. Otherwise, the format of sets is a plain python
        list of texts
    
    """
    
    def download_and_decompress(dst_dir):
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        filename = 'aclImdb_v1.tar.gz'

        # If directory containing uncompressed files does not exists, then download
        # the tar.gz and uncompressed it
        if not path.exists(dst_dir):
            import tarfile
            _download(url, filename)
            tar_file = path.join(_DATA, filename)
            print("Decompressing file {}".format(tar_file))
            tar = tarfile.open(tar_file, "r:gz")
            tar.extractall(path=dst_dir)
            tar.close()
    
    def get_text(fname):
        with open(fname) as f:
            return f.read()
    
    def get_split(base_path, split):
        f_path = lambda f, label: path.join(base_path, split, label, f)

        pos_reviews = os.listdir(path.join(base_path, split, 'pos'))
        neg_reviews = os.listdir(path.join(base_path, split, 'neg'))

        pos_reviews = [get_text(f_path(fname, 'pos')) for fname in pos_reviews]
        neg_reviews = [get_text(f_path(fname, 'neg')) for fname in neg_reviews]

        reviews = pos_reviews + neg_reviews
        labels = [1] * len(pos_reviews) + [0] * len(neg_reviews)
        return reviews, labels

    def clean_texts(texts):
        # 1. Remove punctuation
        # 2. Remove special markups such as <br>
        # 3. Lowercase all texts
        replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        reviews = [t.translate(str.maketrans('', '', string.punctuation)) 
                   for t in texts]
        reviews = [replace_with_space.sub(" ", t).lower() for t in texts]
        return reviews

    base_dir = path.join(_DATA, 'large_movie_review', 'aclImdb')
    download_and_decompress(base_dir)

    x_train, y_train = get_split(base_dir, 'train')
    x_test, y_test = get_split(base_dir, 'test')

    x_train = clean_texts(x_train)
    x_test = clean_texts(x_test)

    perm = np.random.permutation(len(x_train))
    x_train, y_train = zip(*[(x_train[i], y_train[i]) for i in perm])

    if not preprocess:
        return (x_train, y_train), (x_test, y_test)
    
    bag_of_words = collections.Counter()
    for text in x_train:
        for word in text.split()[:words_per_review]:
            if len(word) > min_word_len:
                bag_of_words[word] += 1
     
    # Sort dict by value and pick the top most used words
    bag_of_words = collections.OrderedDict(
        sorted(bag_of_words.items(), key=lambda x: x[1], reverse=True))
    
    most_common_words = list(bag_of_words.keys())[:vocab_size]
    vocab = { '<pad>': 0, '<unk>': 1 }
    vocab = dict(**vocab, 
                 **{w: i for i, w in enumerate(most_common_words, start=2)})

    def encode_text(text):
        # Get word or unknown
        text = [vocab.get(word, 1) 
                for word in text.split()[:words_per_review] 
                if len(word) > min_word_len]
        if len(text) < words_per_review:
            offset = words_per_review - len(text)
            padding = [vocab['<pad>']] * offset
        else:
            padding = []
        return text + padding

    x_train = np.array([encode_text(t) for t in x_train])
    x_test = np.array([encode_text(t) for t in x_test])

    return (x_train, np.array(y_train)), (x_test, np.array(y_test))
