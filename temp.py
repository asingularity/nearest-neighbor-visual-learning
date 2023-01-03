



import tensorflow as tf

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Build the model
input_layer_size = 784
hidden_layer_sizes = [32]
output_layer_size = 10
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=[input_layer_size]),
  tf.keras.layers.Dense(hidden_layer_sizes[0], activation='relu'),
  tf.keras.layers.Dense(output_layer_size)
])

aux_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs + [model.layers[1].output])

# Build the computation graph
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, input_layer_size])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, output_layer_size])

# Get the final output and intermediate output
final_output, intermediate_output = aux_model(inputs)

loss = tf.reduce_mean(tf.square(labels - final_output))  # mean square error loss
optimizer = tf.compat.v1.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# Run the computation graph in a session
with tf.compat.v1.Session() as sess:
  sess.run(tf.compat.v1.global_variables_initializer())
  while True:
    # Continuously get a new sample and train on it
    sample_input, sample_label = ...
    final_output_val, intermediate_output_val, _ = sess.run([final_output, intermediate_output, train_op], feed_dict={inputs: sample_input, labels: sample_label})


'''

copy vision stuff, just what is needed, from NL repo

for IB:
https://github.com/erdewit/ib_insync
IB stuff
https://github.com/erdewit/ib_insync
https://nbviewer.org/github/erdewit/ib_insync/blob/master/notebooks/tick_data.ipynb
https://ib-insync.readthedocs.io/api.html
https://ib-insync.readthedocs.io/recipes.html


for MLP use:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    how to get intermediate activations?
        https://stackoverflow.com/questions/46728937/retrieve-final-hidden-activation-layer-output-from-sklearns-mlpclassifier
        https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/neural_network/_multilayer_perceptron.py#L1266
            search: _forward_pass

use fine grain data to train; but trade on longer time scale
advantage of our network: scalable training i.e. each portion is trainable quickly; training time grows sublinearly with size of network, potentially


# TODO options for training:
#   - train in batches of history data, every K steps per batch (start with this one)
#       - parameters:
#           - how long history to use
#           -
#   - train continuously using partial_fit (experimental)

# every timestep of the simulation is a possible training sample
# need a max time for future event? otherwise when train?
# how to encode/decode event times: exponential: plot to debug


later, max_predict_time for upper layers (if layer dependent) should potentially be much longer

what are inputs to network? since events are "instantaneous"
time since last event (-exp)

for tiling, will need GPU.
for gpu we will need to use tensorflow:
https://www.tensorflow.org/tutorials/quickstart/advanced
first will need to replicate sklearn module
multiple models on same GPU?
https://www.google.com/search?q=tensorflow+many+small+models+in+parallel&oq=tensorflow+many+small+models+in+parallel&aqs=chrome..69i57j33i160l2.6673j0j7&sourceid=chrome&ie=UTF-8


other algorithms knn etc GPU accelerated:
https://github.com/rapidsai/cuml





        if self.t < self.input_concat_timesteps:
            self.t += 1
            return

        # if self.input_concat_timesteps > 1:
        #     state_seq = self.input_history.get_state_sequence(delay_long=self.input_concat_timesteps - 1, delay_short=0)
        #     print(state_seq.shape, self.input_state_dim, self.input_concat_timesteps, input_state.shape)  # (1, 512) 512 1 (512,)
        #
        #     input_state = np.sum(state_seq, axis=0)
        #     input_state[input_state > 1] = 1
        #     nnz_input = np.nonzero(input_state)[0]


'''





def try_nl_functions():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    import matplotlib.pyplot as plt

    fig_bar = plt.figure(figsize=(40, 20))
    ax_bar = fig_bar.add_subplot(1, 1, 1)
    ax_bar.cla()
    ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
    ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

    x = np.linspace(start=0.0, stop=1.0, num=200, endpoint=True)
    y = np.power(x, 100)

    ax_bar.cla()
    ax_bar.plot(x, y, color='k', marker='.')
    #fig_bar.show()

    fig_bar.savefig("temp_plot.png", dpi=100)


import numpy as np
import cv2

def main():
    import matplotlib
    import matplotlib.pyplot as plt
    fig_bar = plt.figure(figsize=(40, 20))
    ax_bar = fig_bar.add_subplot(1, 1, 1)
    ax_bar.cla()
    ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
    ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

    #x = np.linspace(start=0.0, stop=1.0, num=200, endpoint=True)
    #y = np.power(x, 100)

    x_label = np.arange(0, 101)
    x = x_label.astype(np.float)
    #x[100] = np.inf
    y = np.exp(-x*0.06)

    y_reverted = -np.log(y)/0.06

    ax_bar.cla()
    ax_bar.plot(x_label, y, color='k', marker='.')
    plt.show()

    # this doesn't plot for some reason but it is the right inverse
    # ax_bar.cla()
    # ax_bar.plot(x_label, y_reverted, color='g', marker='.')
    # plt.show()

    #while True:
    #    pass
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()
