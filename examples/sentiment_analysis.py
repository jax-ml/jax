import jax
import jax.numpy as np
import jax.experimental.optimizers as optim
import jax.experimental.stax as stax

from examples import datasets

VOCAB_SIZE = 10000
SEQUENCE_LEN = 256
EMBEDDING_DIM = 50

EPOCHS = 4
STEPS_PER_EPOCH = 256
BATCH_SIZE = 32

# Trining functions
def binary_cross_entropy(y_true, y_pred):
    y_true = y_true.reshape(-1).astype('int32')
    y_pred = y_pred.reshape(-1)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    return -np.log(pt)


def forward_and_loss(params, x, y):
    predictions = apply_fn(params, x)
    return np.mean(binary_cross_entropy(y, predictions))


# Metrics
def accuracy_score(y_true, y_pred, threshold=.5):
    y_pred = y_pred.reshape(-1)
    true_mask = y_pred > .5
    
    n_instances = y_true.shape[0]
    correct_mask = (y_true == true_mask).astype('float32')
    n_correct = np.sum(correct_mask)

    return n_correct / n_instances


key = jax.random.PRNGKey(0)

print('Preprocessing dataset...')
ds = datasets.large_movie_review_dataset(
    vocab_size=VOCAB_SIZE, 
    words_per_review=SEQUENCE_LEN,
    preprocess=True)

(x_train, y_train), (x_test, y_test) = ds
x_train = jax.device_put(x_train)
y_train = jax.device_put(y_train)

# Decalare the model
# The model is based on word embeddings and 1d Convolutions
init_fn, apply_fn = stax.serial(
    stax.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    stax.Conv1d(64, filter_shape=(3,)), stax.Relu,
    stax.MaxPool((2,), strides=(2,), spec='NWC'),
    stax.Flatten,
    stax.Dense(256), stax.Relu,
    stax.Dense(128), stax.Relu,
    stax.Dense(1), stax.Sigmoid)

key, subkey = jax.random.split(key)
_, params = init_fn(subkey, (-1, SEQUENCE_LEN))

backward_fn = jax.jit(jax.value_and_grad(forward_and_loss))

# Create the optimizer
optim_init_fn, optim_step, get_params = optim.adam(1e-3)
optim_state = optim_init_fn(params)
optim_step = jax.jit(optim_step)

print('Training...')
for epoch in range(EPOCHS):
    running_loss = 0.0
    print(f'Epoch [{epoch}]')

    for step in range(STEPS_PER_EPOCH):
        key, subkey = jax.random.split(key)
        batch_idx = jax.random.randint(subkey, 
                                       shape=(BATCH_SIZE,), 
                                       minval=0, maxval=x_train.shape[0])
        x = x_train[batch_idx]
        y = y_train[batch_idx]

        loss, grads = backward_fn(get_params(optim_state), x, y)
        running_loss += loss
        
        optim_state = optim_step(step * epoch, grads, optim_state)

        if (step + 1) % 20 == 0:
            mean_loss = running_loss / step
            print(f'Epoch [{epoch}] [{step}/{STEPS_PER_EPOCH}] '
                  f'loss: {mean_loss:.4f}')
    
    print('Evaluating...')
    y_test_pred = apply_fn(get_params(optim_state), x_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test accuracy: {accuracy:.4f}')