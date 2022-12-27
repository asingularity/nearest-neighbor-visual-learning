import datetime
import os
import numpy as np
from run_demo_vision import get_components_params, init_components, run_demo

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt


def post_process_results(sweep_results, sweep_folder, param_set):

    fig = plt.figure(figsize=(40, 20))

    ax_1 = fig.add_subplot(2, 1, 1)
    ax_1.get_xaxis().get_major_formatter().set_scientific(False)
    ax_1.get_yaxis().get_major_formatter().set_scientific(False)

    ax_1.set_title('mean error over time')

    ax_2 = fig.add_subplot(2, 1, 2)
    ax_2.get_xaxis().get_major_formatter().set_scientific(False)
    ax_2.get_yaxis().get_major_formatter().set_scientific(False)

    ax_2.set_title('mean of mean error')

    ax_2_x = []
    ax_2_y = []

    for k in range(len(param_set[2])):

        sim_prefix = str(param_set[0]) + '_' + str(param_set[1]) + '_' + str(param_set[2][k])

        param_value = param_set[2][k]
        param_value_str = param_set[2][k]

        # this is the result of one simulation, given one parameter value
        # it is a dict
        # it may have multiple things in it: values, arrays etc.

        results_dict = sweep_results[k]

        ax_1.plot(results_dict['mean_prediction_error'], label=param_value_str)

        ax_2_x.append(param_value)
        ax_2_y.append(results_dict['mean_mean_prediction_error'])

    ax_1.legend()

    ax_2.plot(ax_2_x, ax_2_y, 'bo')

    #ax_1.plot(actual_times, np.arange(actual_times.shape[0]), color='g', marker='o', linestyle='')
    fig.savefig(os.path.join(sweep_folder, str(param_set[0]) + '_' + str(param_set[1]) + ".png"), dpi=100)

    # we are hard coding for now but later we can do something like thi:
    # for res_name in results_dict:
    #     res_value = results_dict[res_name]
    #
    #     # make a new subplot for res_name if doesn't exist yet
    #     if res_name not in things_to_plot:
    #         things_to_plot[res_name] = {'x': [], 'y': []}
    #
    #     if type(res_value) == np.ndarray:
    #         # if array: plot all with same name one per plot
    #         things_to_plot[res_name]['y'].append(res_value)
    #         pass
    #     else:
    #         # if value: will have one plot with x axis being param value str
    #
    #         pass



def run_one_sweep(param_set, sweeps_prefix):
    # does NOT run in parallel for now; assumes single GPU is used for each parameter value

    root_dir = '/home/csaba'

    main_params = {
        'MAX_TIME': 120000,  # 5000000
        'DISABLE_BRAIN': False,
        'RF_IM_DIM': 8,
        'ROOT_DIR': root_dir,
        'SIM_PREFIX': 'None',  # will be overwritten
        'DISABLE_VIZ': True
    }

    sweep_results = []
    sweep_folder = os.path.join(root_dir, 'projects/NL-sim-sweeps', sweeps_prefix)

    for k in range(len(param_set[2])):
        print()
        print('Stating sweep for parameter:', param_set[1], 'of module:', param_set[0], 'value:', param_set[2][k])
        print()

        sim_prefix = str(param_set[0]) + '_' + str(param_set[1]) + '_' + str(param_set[2][k])

        components_params = get_components_params(main_params=main_params)
        components_params[param_set[0]][param_set[1]] = param_set[2][k]

        # override sim folder manager params here!
        #          'sim_folders_path': main_params['ROOT_DIR'] + '/projects/NL-sim-venv/',
        #         'scripts_folder_path': main_params['ROOT_DIR'] + '/projects/nl_new/'

        components_params['sim_folder_manager']['sim_prefix'] = sim_prefix
        components_params['sim_folder_manager']['sim_folders_path'] = sweep_folder

        demo_components = init_components(components_params=components_params, main_params=main_params)
        resuts_dict = run_demo(demo_components, main_params)

        sweep_results.append(resuts_dict)

    post_process_results(sweep_results, sweep_folder, param_set)


def main():
    print()
    sweeps_prefix = input('sweep sets prefix? >> ')
    print()

    now = datetime.datetime.now().isoformat()

    sweeps_prefix = sweeps_prefix + '_' + now

    # run a sweep
    param_set = ['robot_brain', 'use_context', [True, False]]
    run_one_sweep(param_set=param_set, sweeps_prefix=sweeps_prefix)

    # hidden_state_factor: size of hidden layer dim relative to input state dim
    param_set = ['robot_brain', 'hidden_state_factor', [1, 2, 5, 10, 20]]
    run_one_sweep(param_set=param_set, sweeps_prefix=sweeps_prefix)


if __name__ == '__main__':
    main()