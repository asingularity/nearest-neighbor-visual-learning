


'''

copy vision stuff, just what is needed, from NL repo

for IB:
https://github.com/erdewit/ib_insync

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

    x = np.arange(0, 100)
    y = np.exp(-x*0.06)

    ax_bar.cla()
    ax_bar.plot(x, y, color='k', marker='.')
    plt.show()
    #while True:
    #    pass
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
