
import tensorflow as tf

# helps slow predict on small batch?
tf.compat.v1.disable_eager_execution()


num_gpu = len(tf.config.list_physical_devices('GPU'))
print()
print("Num GPUs Available: ", num_gpu)
print()


class TFRegressor(object):
    def __init__(self, input_layer_size, hidden_layer_sizes, random_state, max_iter, output_layer_size):

        # self.net = MLPRegressor(hidden_layer_sizes=(self.hidden_state_dim,), random_state=1, max_iter=500)
        # output_values = self.net.predict(X=mlp_input[np.newaxis, :])

        #print((None, hidden_layer_sizes[0]))

        self.model = tf.keras.models.Sequential([
            #tf.keras.layers.Flatten(input_shape=(hidden_layer_sizes[0])),
            tf.keras.layers.Flatten(input_shape=[input_layer_size]),
            tf.keras.layers.Dense(hidden_layer_sizes[0], activation='relu'),
            tf.keras.layers.Dense(output_layer_size)  #,
            #tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.Dense(10)
        ])


        self.aux_model = tf.keras.Model(inputs=self.model.inputs,
                                        outputs=self.model.outputs + [self.model.layers[1].output])

        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

        self.model.compile(optimizer='adam',
                           loss=self.loss_fn,
                           metrics=['accuracy'])

        self.optimizer = tf.keras.optimizers.Adam()

        # intermediate values
        # https://stackoverflow.com/questions/59504884/how-tensorflow2-gets-intermediate-layer-output
        # https://stackoverflow.com/questions/63297838/how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction


    def fit_one_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss_value = self.loss_fn(labels, predictions)
        grads = tape.gradient(loss_value, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    def fit(self, X, y):

        x_train = X
        y_train = y

        self.model.fit(x_train, y_train, epochs=5)

    def predict(self, X):

        #output_values = self.model.predict(X, verbose=2)
        final_output, intermediate_layer_output = self.aux_model.predict(X, verbose=2)

        return final_output, intermediate_layer_output



