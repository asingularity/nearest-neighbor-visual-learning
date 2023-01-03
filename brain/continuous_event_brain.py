import os
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
import warnings

warnings.filterwarnings("error", category=UserWarning, message='Warning: converting')

import time
import cv2
# from sklearn.neural_network import MLPRegressor
from brain.tf_1_mlp import TFRegressor as MLPRegressor
from preprocessors.event_pre_processor import SimpleEventPreProcessor

import numpy as np

np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from utils.states_history import ProximalEventsHistory, StatesLimitedHistory
from utils.rf_im_maker import make_im

# WHYI IS THIS BROKEN NOW
# from cuda_dist_query import CudaTable

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt

from math import log

np.set_printoptions(suppress=True, precision=2)

random.seed(0)
np.random.seed(0)


class ContinuousEventBrain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_plots_every_k_sec = params['do_plots_every_k_sec']
        self.last_plot_time = time.time()

        # params specific to this model
        self.use_context = params['use_context']
        self.hidden_state_factor = params['hidden_state_factor']

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

        self.hidden_state_dim = int(self.input_state_dim * self.hidden_state_factor)  # size of MLP hidden layer
        self.context_state_dim = 2 * self.hidden_state_dim  # +/- event changes so double

        # why 2 * max_predict_time? this is the length of data you need for one training point: need max_predict_time behind, and max_predict_time_ahead
        #   why + num_training_points? this is how many training points you will actually have

        print()
        print('Init ProximalEventsHistory with max_delay:', 2 * self.max_predict_time + 1, ', states_dim:', self.input_state_dim)

        self.input_prox = ProximalEventsHistory(params={'max_delay': 2 * self.max_predict_time + 1,  # always taking the middle for one training point!
                                                        'states_dim': self.input_state_dim})

        # self.input_prox_right_history = StatesLimitedHistory(params={'max_delay': self.num_training_points_per_batch,
        #                                                              'states_dim_list': [self.input_state_dim],
        #                                                              'store_extra_data': False})

        # assert self.num_training_points_per_batch > self.max_predict_time, 'Error: Need at least max_predict_time in input_prox_right_history for comparison to prediction!'

        self.predicted_prox_right_history = StatesLimitedHistory(params={'max_delay': self.max_predict_time,
                                                                         'states_dim_list': [self.input_state_dim],
                                                                         'store_extra_data': False})

        # self.input_prox_left_history = StatesLimitedHistory(params={'max_delay': self.num_training_points_per_batch,
        #                                                             'states_dim_list': [self.input_state_dim],
        #                                                             'store_extra_data': False})

        # intermediate events (2x hidden dim) as context
        self.context_prox = ProximalEventsHistory(params={'max_delay': 2 * self.max_predict_time + 1,  # always taking the middle for one training point!
                                                          'states_dim': self.context_state_dim})

        # self.context_prox_left_history = StatesLimitedHistory(params={'max_delay': self.num_training_points_per_batch,
        #                                                               'states_dim_list': [self.context_state_dim],
        #                                                               'store_extra_data': False})

        # for storing self output as context

        # input size to net: input_state_dim + context_state_dim
        # !!! this is no longer compatible with sklearn MLPRegressor! due to output_layer_size
        self.total_net_input_size = self.input_state_dim
        if self.use_context:
            self.total_net_input_size = self.total_net_input_size + self.context_state_dim

        print()
        print('Initializing network with parameters:')
        print('    input_layer_size', self.total_net_input_size)
        print('    hidden_layer_sizes', (self.hidden_state_dim,))
        print('    output_layer_size', self.input_state_dim)
        print()

        self.net = MLPRegressor(input_layer_size=self.total_net_input_size,
                                hidden_layer_sizes=(self.hidden_state_dim,),
                                random_state=1,
                                max_iter=500,
                                output_layer_size=self.input_state_dim)

        # create events from intermediate layer:
        self.interm_events_proc = SimpleEventPreProcessor(params={'value_threshold': 0.05,
                                                                  'dim': self.hidden_state_dim})  # this is the input values dim. the output events dim is 2x this.

        # **************** plots ****************

        # intermediate values history
        # random set of indices to store
        self.interm_to_store = np.random.permutation(self.hidden_state_dim)[0:5]
        self.interm_history = np.zeros((self.max_time, 5))
        self.interm_t = 0

        print('interm to store', self.interm_to_store)

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        # *2? n and p events
        self.interm_events_raster_history = np.zeros((self.hidden_state_dim * 2, self.raster_steps), np.uint8)

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

        self.prediction_error = np.zeros(self.max_time)
        self.mean_prediction_error = np.zeros_like(self.prediction_error)
        self.tau_mean_error = 10000
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

    def get_results_dict(self):

        # TODO truncate arrays if needed !!!

        return {
            'mean_prediction_error': self.mean_prediction_error[0:self.error_t + 1],
            'mean_mean_prediction_error': np.mean(self.mean_prediction_error[0:self.error_t + 1]),
            # 'prediction_error': self.prediction_error[0:self.error_t+1]
        }

    def do_plots(self, actual_times, predicted_times, debug_vals=None):

        # make a "raster" of this prediction
        self.ax_bar.cla()
        self.ax_bar.plot(actual_times, np.arange(actual_times.shape[0]), color='g', marker='o', linestyle='')
        self.ax_bar.plot(predicted_times, np.arange(predicted_times.shape[0]), color='b', marker='o', linestyle='')

        # for k in range(predicted_times.shape[0]):
        #     self.ax_bar.axhline(y=k, color='g')

        self.ax_bar.set_ylim(-0.5, predicted_times.shape[0] - 0.5)

        self.fig_bar.savefig(self.plots_folder + "/actual_g_predict_b_" + str(self.plot_num) + ".png", dpi=100)

        self.ax_bar_horiz.cla()
        self.ax_bar_horiz.plot(self.prediction_error[0:self.error_t + 1])
        # mean_err = np.mean(self.prediction_error[0:self.error_t+1])

        # for k in range(1, 10):
        #    self.ax_bar_horiz.axhline(y=k, color='g')

        self.ax_bar_horiz.plot(self.mean_prediction_error[0:self.error_t + 1], color='r')
        self.ax_bar_horiz.set_ylim([-1, 5])

        # self.fig_bar_horiz.savefig(self.plots_folder + "/predict_error_" + str(self.plot_num) + ".png", dpi=100)
        self.fig_bar_horiz.savefig(self.plots_folder + "/predict_error" + ".png", dpi=100)

        # self.ax_bar_horiz.cla()
        # self.ax_bar_horiz.plot(self.interm_history[0:self.interm_t, :])
        # self.fig_bar_horiz.savefig(self.plots_folder + "/all_interm" + ".png", dpi=100)
        #
        # self.ax_bar_horiz.cla()
        # self.ax_bar_horiz.plot(self.interm_history[max(0, self.interm_t - 1000):self.interm_t, :])
        # self.fig_bar_horiz.savefig(self.plots_folder + "/recent_interm" + ".png", dpi=100)

        self.plot_num += 1

        # THIS IS NORMAL INPUT RASTER
        self.ax_bar_horiz.cla()
        num_rf = self.input_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.input_raster_history[0:num_rf, :],
                                               np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar_horiz.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.fig_bar_horiz.savefig(self.plots_folder + "/raster_input.png", dpi=100)

        self.ax_bar_horiz.cla()
        num_rf = self.interm_events_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.interm_events_raster_history[0:num_rf, :],
                                               np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar_horiz.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.fig_bar_horiz.savefig(self.plots_folder + "/raster_interm_" + str(self.plot_num) + ".png", dpi=100)

        print()
        print('doing plots')
        #print('actual_times', actual_times)
        #print('predicted_times', predicted_times)
        #print('debug_vals', debug_vals)
        print()

        # THIS IS RF RASTER OLD?
        # self.ax_2.cla()
        # num_rf = self.rfs_0_raster_history.shape[0]
        # raster_plot = np.transpose(np.multiply(self.rfs_0_raster_history[0:num_rf, max(0, self.t - 200):self.t],
        #                                        np.arange(num_rf)[:, np.newaxis]))
        # t = np.arange(raster_plot.shape[0])
        # self.ax_2.plot(t, raster_plot, color='b', marker='.', linestyle='')
        # self.fig.savefig(self.plots_folder + "/raster_" + self.layer_name + "_input_output.png", dpi=100)

    def _get_last_actual_and_predicted_times(self, do_mask=False):
        # gets delayed predicted times, and actual times, for plotting or error comparison
        # masked for infinite time

        # self.input_prox_right_history stores from:
        #   ProximalEventsHistory.get_right_left_from_middle,
        # which uses:
        #   mid_index = int((self.max_delay - 1) / 2)
        # where:
        #   max_delay = 2 * self.max_predict_time + 1

        print()
        print('ERROR: _get_last_actual_and_predicted_times: DEPRECATED FUNCTION!')
        print()
        exit(1)

        actual_times = self.input_prox_right_history.get_state(delay=0, state_index=0)
        # where do "1000" values here come from? it is "inf_val" in ProximalEventsHistory
        # actual_times[actual_times > self.max_predict_time] = self.max_predict_time  # infinite in future: peg to 100
        if do_mask:
            actual_times = np.ma.masked_where(actual_times > self.max_predict_time, actual_times)

        predicted_times = self.predicted_prox_right_history.get_state(delay=self.max_predict_time, state_index=0)
        # why this is at 500 sometimes? this is 1e-12: basically no event. also set to max predict time for now
        # predicted_times[predicted_times > self.max_predict_time] = self.max_predict_time
        if do_mask:
            predicted_times = np.ma.masked_where(predicted_times > self.max_predict_time, predicted_times)

        return actual_times, predicted_times


    def _train_one_step(self, net, prox_right_arr, prox_left_arr, ctx_prox_left_arr, use_context):

        input_values = np.exp(-prox_left_arr * self.exp_decay)
        mlp_input = input_values.copy()

        if use_context:
            ctx_values = np.exp(-ctx_prox_left_arr * self.exp_decay)
            mlp_input = np.concatenate((mlp_input, ctx_values))

        output_values = np.exp(-prox_right_arr * self.exp_decay)

        net.fit_one_step(inputs=mlp_input[np.newaxis, :],
                         labels=output_values[np.newaxis, :])


    def _predict_one_step(self, net, prox_left_arr, ctx_prox_left_arr, use_context):
        input_values = np.exp(-prox_left_arr * self.exp_decay)
        mlp_input = input_values.copy()

        if use_context:
            ctx_values = np.exp(-ctx_prox_left_arr * self.exp_decay)
            mlp_input = np.concatenate((mlp_input, ctx_values))

        output_values, interm_output = net.predict(X=mlp_input[np.newaxis, :])
        output_values = output_values.flatten()
        interm_output = interm_output.flatten()

        events_arr_p, events_arr_n = self.interm_events_proc.step(input_arr=interm_output)
        context_states = np.concatenate((events_arr_p, events_arr_n))

        output_values[output_values <= 0] = 1e-12
        output_values[output_values > 1] = 1

        predicted_prox_right = -np.log(output_values) / self.exp_decay

        debug_vals = output_values

        return predicted_prox_right, context_states, debug_vals

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

        # if there is enough data to train one step
        if self.t > 2 * self.max_predict_time + 1:

            # ********** train a new point from middle of prox **********

            # get middle as the training point
            prox_right_arr, prox_left_arr = self.input_prox.get_right_left_from_middle()
            ctx_prox_right_arr, ctx_prox_left_arr = self.context_prox.get_right_left_from_middle()

            # training requires: prox left (forward input), ctx prox left (context input), prox right (training output)
            # handles converting prox times to/from values for the network, etc.

            self._train_one_step(net=self.net,
                                 prox_right_arr=prox_right_arr,
                                 prox_left_arr=prox_left_arr,
                                 ctx_prox_left_arr=ctx_prox_left_arr,
                                 use_context=self.use_context)

            # ********** make a prediction from end of prox **********
            prox_left_now = self.input_prox.get_prox_left_now()
            ctx_prox_left_now = self.context_prox.get_prox_left_now()

            # handles converting prox times to/from values for the network, etc.
            # also handles generation of new context state from interm output using an event preprocessor

            predicted_prox_right, new_context_states, debug_vals = self._predict_one_step(net=self.net,
                                                                                          prox_left_arr=prox_left_now,
                                                                                          ctx_prox_left_arr=ctx_prox_left_now,
                                                                                          use_context=self.use_context)

            self.context_prox.store_new_states(binary_states_arr=new_context_states)
            self.interm_events_raster_history[:, self.raster_t] = new_context_states[:]

            self.predicted_prox_right_history.store_new_states(newest_states_list=[predicted_prox_right])

            # store predicted_prox_right for error comparison
            #   at the current time, we make a prediction ahead; and we train the prediction we should have made in the past
            #   we use the same "actual" prox_right, delayed, to:
            #       set desired output for training point we are training now
            #       compare to actual prediction we made in the past, before this point was trained

            # get prediction error etc.
            # plot error over time

            # prox_right_arr is from ProximalEventsHistory.get_right_left_from_middle,
            # which uses:
            #   mid_index = int((self.max_delay - 1) / 2)
            # where:
            #   max_delay = 2 * self.max_predict_time + 1

            actual_times = prox_right_arr
            predicted_times = self.predicted_prox_right_history.get_state(delay=self.max_predict_time, state_index=0)

            error = np.mean(np.divide(np.abs(actual_times - predicted_times), (actual_times + 1)))
            self.prediction_error[self.error_t] = error

            min_index = max(0, self.error_t - self.tau_mean_error)
            self.mean_prediction_error[self.error_t] = np.mean(self.prediction_error[min_index:self.error_t])

            self.error_t += 1

            # *************** plots ***************

            if self.do_plots_every_k_sec is not None:
                if time.time() - self.last_plot_time > self.do_plots_every_k_sec:
                    actual_times_plot = actual_times.copy()
                    # where do "1000" values here come from? it is "inf_val" in ProximalEventsHistory
                    # actual_times[actual_times > self.max_predict_time] = self.max_predict_time  # infinite in future: peg to 100

                    actual_times_plot = np.ma.masked_where(actual_times_plot > self.max_predict_time, actual_times_plot)

                    predicted_times_plot = predicted_times.copy()
                    # why this is at 500 sometimes? this is 1e-12: basically no event. also set to max predict time for now
                    # predicted_times[predicted_times > self.max_predict_time] = self.max_predict_time

                    predicted_times_plot = np.ma.masked_where(predicted_times_plot > self.max_predict_time, predicted_times_plot)

                    self.do_plots(actual_times=actual_times_plot, predicted_times=predicted_times_plot, debug_vals=(debug_vals, predicted_prox_right))
                    self.last_plot_time = time.time()

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0















