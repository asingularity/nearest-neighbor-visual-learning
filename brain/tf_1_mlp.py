
import tensorflow as tf

# helps slow predict on small batch
# but THIS SILENTLY DISABLES self.optimizer.apply_gradients!!!
# https://stackoverflow.com/questions/59332812/keras-optimizers-adam-apply-gradients-fails
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
            tf.keras.layers.Flatten(input_shape=[input_layer_size]),
            tf.keras.layers.Dense(hidden_layer_sizes[0], activation='relu'),
            tf.keras.layers.Dense(output_layer_size)  #,
        ])

        self.aux_model = tf.keras.Model(inputs=self.model.inputs,
                                        outputs=self.model.outputs + [self.model.layers[1].output])

        # Build the computation graph
        self.inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, input_layer_size])
        self.labels = tf.compat.v1.placeholder(tf.float32, shape=[None, output_layer_size])

        # Get the final output and intermediate output
        self.final_output, self.intermediate_output = self.aux_model(self.inputs)

        self.loss = tf.reduce_mean(tf.square(self.labels - self.final_output))  # mean square error loss
        self.optimizer = tf.compat.v1.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)

        # Build the computation graph for prediction
        self.pred_inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, input_layer_size])
        self.pred_output, self.hidden_layer_state = self.aux_model(self.pred_inputs)

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # intermediate values
        # https://stackoverflow.com/questions/59504884/how-tensorflow2-gets-intermediate-layer-output
        # https://stackoverflow.com/questions/63297838/how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction

    @tf.function
    def _tf_fit(self, inputs, labels, model):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss_value = self.loss_fn(labels, predictions)

        grads = tape.gradient(loss_value, model.trainable_variables)

        return grads, loss_value

    def fit_one_step(self, inputs, labels):

        sample_input = inputs
        sample_label = labels

        # tf_fit seems to be source of memory leak!
        final_output_val, intermediate_output_val, _ = self.sess.run([self.final_output, self.intermediate_output, self.train_op],
                                                                     feed_dict={self.inputs: sample_input, self.labels: sample_label})


        #grads, loss_value = self._tf_fit(inputs=inputs, labels=labels, model=self.model)
        #self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    # def fit(self, X, y):
    #
    #     x_train = X
    #     y_train = y
    #
    #     self.model.fit(x_train, y_train, epochs=5)

    def predict(self, X):

        #output_values = self.model.predict(X, verbose=2)
        #final_output, intermediate_layer_output = self.aux_model.predict(X, verbose=0)  # 2

        final_output, intermediate_layer_output = self.sess.run([self.pred_output, self.hidden_layer_state], feed_dict={self.pred_inputs: X})

        return final_output, intermediate_layer_output



