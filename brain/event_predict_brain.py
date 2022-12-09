import os

import time
import cv2
from sklearn.neural_network import MLPRegressor
from brain.tf_mlp import TFRegressor


import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from utils.states_history import ProximalEventsHistory, StatesLimitedHistory
from utils.rf_im_maker import make_im


# WHYI IS THIS BROKEN NOW
#from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

np.set_printoptions(suppress=True, precision=2)

random.seed(0)
np.random.seed(0)


class EventPredictBrain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = params['num_rfs']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_plots_every_k_sec = params['do_plots_every_k_sec']
        self.last_plot_time = time.time()

        self.t = 0

        # **************** input ****************

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.plots_folder = "."
        self.input_concat_timesteps = 1
        assert self.input_concat_timesteps == 1
        # we use input_history also for network, so init below instead with longer history
        # self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
        #                                                   'states_dim_list': [self.input_state_dim],
        #                                                   'store_extra_data': False})

        # **************** network ****************

        # old networks had this. need intermediate weights in this way later to visualize:
        # self.rf_weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-2  # 1e-12

        self.max_predict_time = 100  # assume next event time > max_predict_time == infinite
        self.exp_decay = 0.06  # 0.06: roughly matches 100 max predict time (use temp.py to plot and set)
        self.num_training_points_per_batch = 20000  # 10K how many points for training one batch?
        self.retrain_every_k_steps = 20000  # 5K how often to retrain network i.e. to train based on a new batch. half of num training points: half overlap to last batch

        # TODO dim expansion of *10 is too large here!
        self.hidden_state_dim = self.input_state_dim * 2  # size of MLP hidden layer

        # why 2 * max_predict_time? this is the length of data you need for one training point: need max_predict_time behind, and max_predict_time_ahead
        #   why + num_training_points? this is how many training points you will actually have

        print()
        print('Init ProximalEventsHistory with max_delay:', 2 * self.max_predict_time + 1, ', states_dim:', self.input_state_dim)

        self.input_prox = ProximalEventsHistory(params={'max_delay': 2 * self.max_predict_time + 1,  # always taking the middle for one training point!
                                                        'states_dim': self.input_state_dim})

        self.input_prox_right_history = StatesLimitedHistory(params={'max_delay': self.num_training_points_per_batch,
                                                                     'states_dim_list': [self.input_state_dim],
                                                                     'store_extra_data': False})

        assert self.num_training_points_per_batch > self.max_predict_time, 'Error: Need at least max_predict_time in input_prox_right_history for comparison to prediction!'

        self.predicted_prox_right_history = StatesLimitedHistory(params={'max_delay': self.max_predict_time,
                                                                         'states_dim_list': [self.input_state_dim],
                                                                         'store_extra_data': False})

        self.input_prox_left_history = StatesLimitedHistory(params={'max_delay': self.num_training_points_per_batch,
                                                                    'states_dim_list': [self.input_state_dim],
                                                                    'store_extra_data': False})

        # self.context_history = BinaryEventsHistory(params={'max_delay': 2 * self.max_predict_time + self.num_training_points_per_batch,
        #                                                    'states_dim': self.hidden_state_dim})

        # for storing self output as context

        # input size to net: input_state_dim + context_state_dim
        self.net = MLPRegressor(hidden_layer_sizes=(self.hidden_state_dim,), random_state=1, max_iter=500)

        self.last_train_t = -np.inf
        self.net_trained_once = False

        # **************** plots ****************

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

        self.fig_bar = plt.figure(figsize=(20, 40))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        self.fig_bar_horiz = plt.figure(figsize=(40, 20))
        self.ax_bar_horiz = self.fig_bar_horiz.add_subplot(1, 1, 1)
        self.ax_bar_horiz.cla()
        self.ax_bar_horiz.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar_horiz.get_yaxis().get_major_formatter().set_scientific(False)

        self.plot_num = 0

        self.prediction_error = np.zeros(10000000)
        self.error_t = 0

        # self.fig_bar = plt.figure(figsize=(40, 40))
        # self.ax_bar_list = []
        # self.rfs_to_plot = 16  # 8 or self.num_rfs
        #
        # for k in range(self.rfs_to_plot):
        #     subpl = self.fig_bar.add_subplot(self.rfs_to_plot, 1, k + 1)
        #     subpl.cla()
        #     subpl.get_xaxis().get_major_formatter().set_scientific(False)
        #     subpl.get_yaxis().get_major_formatter().set_scientific(False)
        #     self.ax_bar_list.append(subpl)

    def get_final_errors_dict(self):
        d = {}
        return d

    def get_table_ims(self):

        ims_list = []
        ims_names_list = []

        # TODO has to be based on intermediate layer weights instead of self.rf_weights

        # rfs_im, rf_ims_dict = make_im(self.rf_weights,
        #                               num_bins_per_pixel=1,
        #                               input_im_dim=self.input_im_dim,
        #                               im_final_dim=int(int(sqrt(self.num_rfs)) * 100),  # /800 for two-im per rf display
        #                               mod_for_disp=int(sqrt(self.num_rfs)),
        #                               normalize_weights=True)
        #
        # ims_list.append(rfs_im)
        # ims_names_list.append('prob_weights')

        return ims_list, ims_names_list

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def _get_last_actual_and_predicted_times(self):
        # gets delayed predicted times, and actual times, for plotting or error comparison
        # masked for infinite time

        # self.input_prox_right_history stores from:
        #   ProximalEventsHistory.get_right_left_from_middle,
        # which uses:
        #   mid_index = int((self.max_delay - 1) / 2)
        # where:
        #   max_delay = 2 * self.max_predict_time + 1

        actual_times = self.input_prox_right_history.get_state(delay=0, state_index=0)
        # where do "1000" values here come from? it is "inf_val" in ProximalEventsHistory
        # actual_times[actual_times > self.max_predict_time] = self.max_predict_time  # infinite in future: peg to 100
        actual_times = np.ma.masked_where(actual_times > self.max_predict_time, actual_times)

        predicted_times = self.predicted_prox_right_history.get_state(delay=self.max_predict_time, state_index=0)
        # why this is at 500 sometimes? this is 1e-12: basically no event. also set to max predict time for now
        # predicted_times[predicted_times > self.max_predict_time] = self.max_predict_time
        predicted_times = np.ma.masked_where(predicted_times > self.max_predict_time, predicted_times)

        return actual_times, predicted_times

    def do_plots(self):
        # look at dynamic_coincidence.py, others
        if self.net_trained_once:

            actual_times, predicted_times = self._get_last_actual_and_predicted_times()

            # make a "raster" of this prediction
            self.ax_bar.cla()
            self.ax_bar.plot(actual_times, np.arange(actual_times.shape[0]), color='g', marker='o', linestyle='')
            self.ax_bar.plot(predicted_times, np.arange(predicted_times.shape[0]), color='b', marker='o', linestyle='')

            # for k in range(predicted_times.shape[0]):
            #     self.ax_bar.axhline(y=k, color='g')

            self.ax_bar.set_ylim(-0.5, predicted_times.shape[0] - 0.5)

            self.fig_bar.savefig(self.plots_folder + "/actual_g_predict_b_" + str(self.plot_num) + ".png", dpi=100)

            self.ax_bar_horiz.cla()
            self.ax_bar_horiz.plot(self.prediction_error[0:self.error_t+1])
            mean_err = np.mean(self.prediction_error[0:self.error_t+1])
            self.ax_bar_horiz.axhline(y=mean_err, color='r')
            self.fig_bar_horiz.savefig(self.plots_folder + "/predict_error_" + str(self.plot_num) + ".png", dpi=100)

            self.plot_num += 1

        # THIS IS NORMAL INPUT RASTER
        # self.ax_bar.cla()
        # num_rf = self.input_raster_history.shape[0]
        # raster_plot = np.transpose(np.multiply(self.input_raster_history[0:num_rf, :],
        #                                        np.arange(num_rf)[:, np.newaxis]))
        # t = np.arange(raster_plot.shape[0])
        # self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        # self.fig_bar.savefig(self.plots_folder + "/raster_input.png", dpi=100)

        # THIS IS RF RASTER OLD?
        # self.ax_2.cla()
        # num_rf = self.rfs_0_raster_history.shape[0]
        # raster_plot = np.transpose(np.multiply(self.rfs_0_raster_history[0:num_rf, max(0, self.t - 200):self.t],
        #                                        np.arange(num_rf)[:, np.newaxis]))
        # t = np.arange(raster_plot.shape[0])
        # self.ax_2.plot(t, raster_plot, color='b', marker='.', linestyle='')
        # self.fig.savefig(self.plots_folder + "/raster_" + self.layer_name + "_input_output.png", dpi=100)

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):
        '''


        :param input_events_p:
        :param input_events_n:
        :param event_coords_r:
        :param event_coords_c:
        :param original_input_image:
        :return:
        '''

        # *************** input ***************

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        # for prediction, Don't skip zeros!!!
        # if np.count_nonzero(input_state) == 0:
        #     return

        self.input_prox.store_new_states(binary_states_arr=input_state.astype(np.int))

        if self.t > 2 * self.max_predict_time + 1:

            # *************** step history for training ***************

            # get middle as the training point
            prox_right_arr, prox_left_arr = self.input_prox.get_right_left_from_middle()

            self.input_prox_right_history.store_new_states(newest_states_list=[prox_right_arr])
            self.input_prox_left_history.store_new_states(newest_states_list=[prox_left_arr])

            # *************** step network ***************

            # prox_right: how far in the future is the next event after now; interpreteable on the right side of history diagram
            #   how much time "from now"
            # prox_left: how far in the past was the last event before now; interpretable on the left side of the history diagram
            #   how much time "ago"

            prox_left_now = self.input_prox.get_prox_left_now()
            input_values = np.exp(-prox_left_now * self.exp_decay)
            mlp_input = input_values.copy()

            # (3) get context times, transform to values
            # last_context_times = self.context_history.get_last_event_times_before(delay=0)
            # context_values = np.exp(-last_input_times * 0.06)
            # mlp_input = np.concatenate((mlp_input, context_values))

            # (4) step MLP (if trained already)
            if self.net_trained_once:
                output_values = self.net.predict(X=mlp_input[np.newaxis, :]).flatten()

                output_values[output_values<=0] = 1e-12
                output_values[output_values>1] = 1

                predicted_prox_right = -np.log(output_values)/self.exp_decay

                self.predicted_prox_right_history.store_new_states(newest_states_list=[predicted_prox_right])

                # plot error over time
                actual_times, predicted_times = self._get_last_actual_and_predicted_times()
                error = np.mean(np.abs(actual_times - predicted_times))
                self.prediction_error[self.error_t] = error
                self.error_t += 1

                # TODO also need to evaluate and plot hidden layer values over time

            # (5) update context states history (from step, or zero values otherwise)
            # self.context_history.store_new_states(newest_states_list=[context_state])

            # *************** train network ***************

            if self.t > self.last_train_t + self.retrain_every_k_steps and self.t > self.num_training_points_per_batch + 2 * self.max_predict_time + 1:

                # (1) get forward input history for batch

                train_prox_left = self.input_prox_left_history.get_state_array_history()
                train_input_values = np.exp(-train_prox_left * self.exp_decay)
                mlp_input_train = train_input_values

                # (2) get context input history for batch
                # mlp_input_train.append(...)

                # (3) get output history for batch

                train_prox_right = self.input_prox_right_history.get_state_array_history()
                train_output_values = np.exp(-train_prox_right * self.exp_decay)
                mlp_output_train = train_output_values

                # (4) do training for batch

                # X: (n_samples, n_features), Y:  (n_samples, n_outputs)
                print('starting:    self.net.fit(X=mlp_input_train, y=mlp_output_train)    ...')
                print()

                print()
                print('train_prox_left', train_prox_left.shape)
                print('train_prox_right', train_prox_right.shape)
                print('mlp_input_train', mlp_input_train.shape)
                print('mlp_output_train', mlp_output_train.shape)
                print()

                self.net.fit(X=mlp_input_train, y=mlp_output_train)

                print('finished training!')
                print()

                self.last_train_t = self.t
                self.net_trained_once = True

        # *************** plots ***************

        if self.do_plots_every_k_sec is not None:
            if time.time() - self.last_plot_time > self.do_plots_every_k_sec:
                self.do_plots()
                self.last_plot_time = time.time()

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0
