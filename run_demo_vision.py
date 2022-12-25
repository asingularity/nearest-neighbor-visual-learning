'''
    this file started 11/7/22
    new demo:
        predict next event times
        use MLP hidden layer to generate "current layer" events
'''

import os
# this impacts numpy -> and MLPRegressor, which has no n_jobs so this is where that is set
os.environ["OMP_NUM_THREADS"] = '1'

# this is unused currently as we are not using mkl with this numpy I think:
#import mkl
#mkl.set_num_threads(1)

import cv2
import random
import numpy as np
random.seed(6)
np.random.seed(6)

from utils.sim_folder_manager import SimFolderManager
from brain.event_predict_brain import EventPredictBrain
from visualizers.basic_visualizer import BasicVisualizer
from sensors.video_playback import VideoPlaybackSensor
from preprocessors.event_pre_processor import EventPreProcessor


def get_sensors_params(main_params):

    # 'video_filename': 'seattle-driving-fkps18H3SXY.mp4',

    square_crop = (128 - int(main_params['RF_IM_DIM'] / 2), 128 - int(main_params['RF_IM_DIM'] / 2), main_params['RF_IM_DIM'])

    params_video_playback = {
        'pickle_im_square_crop': None,
        'pickle_im_square_resize': 256,
        'force_reload_to_pkl': False,
        'gb_per_pkl_file': 1,
        'returned_im_square_crop': square_crop,
        'returned_im_dtype': np.float32,  # only np.float32 supported
        'returned_im_use_color': False,  # only False supported for now
        'video_dir': main_params['ROOT_DIR'] + '/projects/video-downloads',
        #'video_filename': 'sea-turtles-yLuEx-XH3Uc.mp4'  # not on laptop
        #'video_filename': 'seattle-driving-fkps18H3SXY.mp4'
        'video_filename': 'sea-turtles-11hr-spxtEt6RaS4.mp4'
    }

    return params_video_playback


def get_pre_proc_params(main_params):
    params_pre_proc = {
        'brightness_threshold': 2.0 / 255,
        'im_dim': main_params['RF_IM_DIM']
    }
    return params_pre_proc


def get_sim_folder_manager_params(main_params):
    sim_prefix = main_params['SIM_PREFIX']

    if sim_prefix[-1] != '_':
        sim_prefix += '_'

    if len(sim_prefix) == 0:
        sim_prefix = 'default_'

    params = {
        'sim_prefix': sim_prefix,
        'sim_folders_path': main_params['ROOT_DIR'] + '/projects/NL-sim-venv/',
        'scripts_folder_path': main_params['ROOT_DIR'] + '/projects/NL/'
    }
    return params


def get_brain_params(main_params):

    brain_params = {
        'input_im_dim': main_params['RF_IM_DIM'],
        'num_rfs': 64,  # 64
        'lr': 1.0 / 100,  # 1000
        'max_time': main_params['MAX_TIME'],
        'do_plots_every_k_sec': 10,  # 5 or None
        'use_context': True,
        'hidden_state_factor': 10
    }

    return brain_params


def get_visualizer_params(main_params):
    params = {
        'color_enabled': False,
        'fps_display_interval': 6,
        'image_display_secs_fast': 10.1,  # 8+ for good speed
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0.0, # 0.01  # 2,  # 0: every frame
        'waitKey_time_slow': 1,  # 1, 100, 5000
        'scale_camera_factor': 1,
        'auto_switch_to_slow_disp_time': None,
        'init_fast': True, #not DISABLE_BRAIN  # start with "fast" display if brain is enabled
    }
    return params


def get_components_params(main_params):
    return {
        'robot_brain': get_brain_params(main_params=main_params),
        'robot_sensors': get_sensors_params(main_params=main_params),
        'visualizer': get_visualizer_params(main_params=main_params),
        'sim_folder_manager': get_sim_folder_manager_params(main_params=main_params),
        'pre_processor': get_pre_proc_params(main_params=main_params)
    }


def init_components(components_params, main_params):
    return {
        'robot_brain': EventPredictBrain(components_params['robot_brain']),
        'robot_sensors': VideoPlaybackSensor(components_params['robot_sensors']),
        'visualizer': BasicVisualizer(components_params['visualizer']),
        'sim_folder_manager': SimFolderManager(components_params['sim_folder_manager']),
        'pre_processor': EventPreProcessor(components_params['pre_processor'])
    }


def run_demo(demo_components, main_params):

    robot_brain = demo_components['robot_brain']
    robot_sensors = demo_components['robot_sensors']

    visualizer = demo_components['visualizer']
    sim_folder_manager = demo_components['sim_folder_manager']
    pre_proc = demo_components['pre_processor']

    robot_brain.set_plots_folder(sim_folder_manager.get_plots_save_folder())

    random.seed(1233)

    for timestep in range(main_params['MAX_TIME']):

        # reference
        # a = np.dot(np.random.random((200, 200)), np.random.random((200, 200)))

        im = robot_sensors.read_input()

        events_p, events_n, event_coords_r, event_coords_c, original_input_image = pre_proc.step(input_frame=im)

        if not main_params['DISABLE_BRAIN']:
            robot_brain.process_input(input_events_p=events_p, input_events_n=events_n,
                                      event_coords_r=event_coords_r, event_coords_c=event_coords_c,
                                      original_input_image=original_input_image)
        if not main_params['DISABLE_VIZ']:
            visualizer.visualize(input_im=im,
                                 segment_brain=robot_brain,
                                 disable_brain=main_params['DISABLE_BRAIN'])  # So it can call .get_table_ims() only sometimes

    # robot_brain.save_model(models_save_folder=sim_folder_manager.get_models_save_folder())

    print ('Finished Evaluation.')

    # while True:
    #     cv2.waitKey(1)


def demo():
    '''
    MAX_TIME = 5000000
    DISABLE_BRAIN = False  # for testing input by itself; also starts slow display
    RF_IM_DIM = 16  # 8, 16, 32, 64, 128

    # ROOT_DIR = '/srv'
    ROOT_DIR = '/home/csaba'

    :return:
    '''
    print()
    sim_prefix = input('sim prefix? >> ')
    print()

    main_params = {
        'MAX_TIME': 5000000,
        'DISABLE_BRAIN': False,
        'RF_IM_DIM': 16,
        'ROOT_DIR': '/home/csaba',
        'SIM_PREFIX': sim_prefix,
        'DISABLE_VIZ': False
    }

    components_params = get_components_params(main_params=main_params)
    demo_components = init_components(components_params=components_params, main_params=main_params)
    run_demo(demo_components, main_params)



if __name__ == '__main__':
    demo()
















