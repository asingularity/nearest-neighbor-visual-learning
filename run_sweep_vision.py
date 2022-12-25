import datetime
import os
from run_demo_vision import get_components_params, init_components, run_demo



def run_one_sweep(param_set, sweeps_prefix):
    # does NOT run in parallel for now; assumes single GPU is used for each parameter value

    # TODO WE ARE HERE
    # TODO modify sim folder manager params to make sim names with param values inside sweep folder etc.
    # TODO run_demo should run for MAX_TIME

    root_dir = '/home/csaba'

    main_params = {
        'MAX_TIME': 1000,  # 5000000
        'DISABLE_BRAIN': False,
        'RF_IM_DIM': 16,
        'ROOT_DIR': root_dir,
        'SIM_PREFIX': 'None',  # will be overwritten
        'DISABLE_VIZ': True
    }

    for k in range(len(param_set[2])):
        print()
        print('Stating sweep for parameter:', param_set[1], 'of module:', param_set[0], 'value:', param_set[2][k])
        print()

        sim_prefix = str(param_set[0]) + '_' + str(param_set[1]) + '_' + str(param_set[2][k])

        components_params = get_components_params(main_params=main_params)
        components_params[param_set[0]][param_set[1]] = param_set[2][k]

        # override sim folder manager params here!
        #          'sim_folders_path': main_params['ROOT_DIR'] + '/projects/NL-sim-venv/',
        #         'scripts_folder_path': main_params['ROOT_DIR'] + '/projects/NL/'

        components_params['sim_folder_manager']['sim_prefix'] = sim_prefix
        components_params['sim_folder_manager']['sim_folders_path'] = os.path.join(root_dir, 'projects/NL-sim-sweeps', sweeps_prefix)

        demo_components = init_components(components_params=components_params, main_params=main_params)
        run_demo(demo_components, main_params)


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