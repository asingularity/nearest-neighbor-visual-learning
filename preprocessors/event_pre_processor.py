
import numpy as np



class EventPreProcessor(object):
    '''
        This class generates events from frame-based images
        i.e. it simulates an event-based camera
    '''

    def __init__(self, params):
        self.brightness_threshold = params['brightness_threshold']
        self.im_dim = params['im_dim']  # N, for NxN image

        self.state_dim = self.im_dim * self.im_dim
        self.last_event_brightness = np.zeros(self.state_dim)
        self.t = 0

        r = np.tile(np.arange(self.im_dim)[:, np.newaxis], self.im_dim).astype(np.int)
        c = np.transpose(r).astype(np.int)
        self.event_coords_r = r.flatten()
        self.event_coords_c = c.flatten()

    def step(self, input_frame):
        '''

        assumes input_frame is grayscale already

        :param input_frame:
        :return:
        '''

        # events_arr: binary array of pixels: 1 if event this time step, 0 if no event this time step
        input_state = input_frame.flatten()

        if self.t == 0:
            self.last_event_brightness[:] = input_state[:]

        nnz_events_p = np.nonzero((input_state - self.last_event_brightness) > self.brightness_threshold)
        nnz_events_n = np.nonzero((input_state - self.last_event_brightness) < -self.brightness_threshold)

        self.last_event_brightness[nnz_events_p] = input_state[nnz_events_p]
        self.last_event_brightness[nnz_events_n] = input_state[nnz_events_n]

        events_arr_p = np.zeros(input_state.shape[0], np.float32)
        events_arr_n = np.zeros(input_state.shape[0], np.float32)

        events_arr_p[nnz_events_p] = 1
        events_arr_n[nnz_events_n] = 1

        self.t += 1

        original_input_image = input_frame.copy()
        return events_arr_p, events_arr_n, self.event_coords_r.copy(), self.event_coords_c.copy(), original_input_image


class SimpleEventPreProcessor(object):
    '''
    like EventPreProcessor but without 2D / image assumption
    just takes a flat array of values and makes events, over time
    '''

    def __init__(self, params):
        self.value_threshold = params['value_threshold']
        self.dim = params['dim']
        self.last_event_values = np.zeros(self.dim)
        self.t = 0
        # self.last_values = None

    def step(self, input_arr):
        if self.t == 0:
            self.last_event_values = input_arr.copy()

        nnz_events_p = np.nonzero((input_arr - self.last_event_values) > self.value_threshold)
        nnz_events_n = np.nonzero((input_arr - self.last_event_values) < -self.value_threshold)

        self.last_event_values[nnz_events_p] = input_arr[nnz_events_p]
        self.last_event_values[nnz_events_n] = input_arr[nnz_events_n]

        events_arr_p = np.zeros(input_arr.shape[0], np.float32)
        events_arr_n = np.zeros(input_arr.shape[0], np.float32)

        events_arr_p[nnz_events_p] = 1
        events_arr_n[nnz_events_n] = 1

        self.t += 1

        return events_arr_p, events_arr_n

