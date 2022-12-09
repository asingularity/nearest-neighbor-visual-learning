
import tensorflow as tf
num_gpu = len(tf.config.list_physical_devices('GPU'))
print()
print("Num GPUs Available: ", num_gpu)
print()


class TFRegressor(object):
    def __init__(self, hidden_layer_sizes, random_state, max_iter):

        # self.net = MLPRegressor(hidden_layer_sizes=(self.hidden_state_dim,), random_state=1, max_iter=500)
        # output_values = self.net.predict(X=mlp_input[np.newaxis, :])


        pass

    def fit(self, X, y):
        # self.net.fit(X=mlp_input_train, y=mlp_output_train)
        pass

    def predict(self, X):

        

        return output_values
