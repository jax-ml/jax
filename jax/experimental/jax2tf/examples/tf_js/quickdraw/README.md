# Quickdraw

This directory contains the code necessary to train a CNN on the Quickdraw
dataset using Flax, and subsequently convert it using jax2tf into a model that
can be loaded in TF.js.

This example is based on [a demo featured in the TensorFlow blog](https://blog.tensorflow.org/2018/07/train-model-in-tfkeras-with-colab-and-run-in-browser-tensorflowjs.html).
The `third_party` code is taken from
[zaidalyafeai.github.io](https://github.com/zaidalyafeai/zaidalyafeai.github.io/tree/master/sketcher),
and is a part of the code featured in the aforementioned blog post.

## Training the model

You can train and export the model yourself by running
```bash
$ python3 quickdraw.py
```

The training itself lasts roughly 30 minutes, assuming the dataset has already
been downloaded. You can also skip to the [next section](#interacting-with-the-model)
to see instructions for playing with a pre-trained model.

The dataset will be downloaded directly into a `data/` directory in the
`/tmp/jax2tf/tf_js_quickdraw` directory; by default, the model is configured to
classify inputs into 100 different classes, which corresponds to a dataset of
roughly 11 Gb. This can be tweaked by modifying the value of the `NB_CLASSES`
global variable in `quickdraw.py` (max number of classes: 100). Only the files
corresponding to the first `NB_CLASSES` classes in the dataset will be
downloaded.

The training loop runs for 5 epochs, and the model as well as its equivalent
TF.js-loadable model are subsequently saved into `/tmp/jax2tf/tf_js_quickdraw`.

## Interacting with the model

Here is a (third party) [demo](https://zaidalyafeai.github.io/sketcher/) of how
interacting with the trained model looks like.

If you want to try it yourself, the trained model along with third party front
end code can be downloaded [here](https://storage.googleapis.com/jax2tf-examples/tf_js/quickdraw/quickdraw_demo_files.tar.gz). The tarball contains the same
exact files as the website linked above, save for one line change to `main.js`
(tagged with a comment), along with the trained weights. The model has an
accuracy of roughly 78% on the used training set, and 76% on the used test set,
where we define an accurate prediction as one for which the class predicted
with the highest likelihood is the expected class.

Simply unpack the tarball, `cd` into the resulting directory and run

```bash
$ python3 -m http.server <port>
```

to interact with the model at `localhost:<port>`.
