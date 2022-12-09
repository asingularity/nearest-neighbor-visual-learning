
import tensorflow as tf
num_gpu = len(tf.config.list_physical_devices('GPU'))
print()
print("Num GPUs Available: ", num_gpu)
print()


class TFRegressor(object):
    def __init__(self, hidden_layer_sizes, random_state, max_iter, output_layer_size):

        # self.net = MLPRegressor(hidden_layer_sizes=(self.hidden_state_dim,), random_state=1, max_iter=500)
        # output_values = self.net.predict(X=mlp_input[np.newaxis, :])

        #print((None, hidden_layer_sizes[0]))

        self.model = tf.keras.models.Sequential([
            #tf.keras.layers.Flatten(input_shape=(hidden_layer_sizes[0])),
            tf.keras.layers.Dense(hidden_layer_sizes[0], activation='relu'),
            tf.keras.layers.Dense(output_layer_size)  #,
            #tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.Dense(10)
        ])

        loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

        self.model.compile(optimizer='adam',
                           loss=loss_fn,
                           metrics=['accuracy'])


    def fit(self, X, y):

        x_train = X
        y_train = y

        self.model.fit(x_train, y_train, epochs=5)

    def predict(self, X):

        #self.model.evaluate(x_test, y_test, verbose=2)
        output_values = self.model.predict(X, verbose=2)

        return output_values



