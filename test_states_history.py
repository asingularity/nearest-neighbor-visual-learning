
import numpy as np
np.set_printoptions(suppress=True)
from utils.states_history import StatesLimitedHistory, ProximalEventsHistory
from utils.fps_counter import FPSCounter

if __name__ == '__main__':

    # test binary.
    # (1) show matrices for several steps of random!
    #       small matrix, a few steps; print nicely
    # (2) speed test
    #       realistically large matrix, many steps, use FPS


    function_test = False
    speed_test = True


    if function_test:
        max_delay = 6
        states_dim = 3

        tmp = ProximalEventsHistory(params={'max_delay': max_delay,
                                          'states_dim': states_dim})

        for step in range(10):
            print()
            print('>>>>>>>>>> >>>>>>>>>> >>>>>>>>>> step', step)
            print()

            rand_data = np.random.randint(low=0, high=1+1, size=(states_dim,))

            print('rand_data', rand_data)
            print()

            tmp.store_new_states(binary_states_arr=rand_data)

            print('events_history:')
            print(tmp.events_history)
            print()

            print('right_proximal:')
            print(tmp.right_proximal)
            print()

            print('left_proximal:')
            print(tmp.left_proximal)

            print()

    if speed_test:
        max_delay = 2 * 100 + 1  # 2 * max_predict_time
        states_dim = 2 * 16 * 16  # * 10  # hidden_state_dim -> input_state_dim * 10 -> 2 * input_im_dim * input_im_dim * 10

        fps = FPSCounter()
        print()
        print('Init ProximalEventsHistory with max_delay:', max_delay, ', states_dim:', states_dim)

        tmp = ProximalEventsHistory(params={'max_delay': max_delay,
                                            'states_dim': states_dim})

        for step in range(10000):

            rand_data = np.random.randint(low=0, high=1 + 1, size=(states_dim,))

            tmp.store_new_states(binary_states_arr=rand_data)

            fps.update()

        fps.update(force_display=True)
        print()


    test_states_limited = False
    if test_states_limited:
        tmp = StatesLimitedHistory(params={'max_delay': 10,
                                           'states_dim_list': [3],
                                           'store_extra_data': False})
        np.random.seed(0)
        print()
        for t in range(20):
            rand_data = np.random.random(3)

            print('t', t, 'data', rand_data)
            tmp.store_new_states(newest_states_list=[rand_data])

        print()
        tmp_data = tmp.get_state_sequence(state_index=0, delay_long=9, delay_short=0, oldest_first=False)
        print(tmp_data)

        print()
        tmp_data = tmp.get_state(state_index=0, delay=0)
        print(tmp_data)
        print()
        tmp_data = tmp.get_state(state_index=0, delay=9)
        print(tmp_data)


