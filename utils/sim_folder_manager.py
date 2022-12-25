import os
import datetime
import shutil


class SimFolderManager(object):
    def __init__(self, params):
        self.sim_prefix = params['sim_prefix']

        now = datetime.datetime.now().isoformat()
        self.sim_folder_path = params['sim_folders_path'] + '/' + self.sim_prefix + '_' + now
        os.makedirs(self.sim_folder_path)

        self.models_save_folder = self.sim_folder_path
        self.plots_save_folder = self.sim_folder_path

        # TODO fix this again (just copy whole repo) or just use git commit id
        file_list = []
        files = os.listdir(params['scripts_folder_path'])

        shutil.copytree(params['scripts_folder_path'], os.path.join(self.sim_folder_path, 'code'), ignore=shutil.ignore_patterns('venv'))

        #for filename in file_list:
        #    shutil.copy2(params['scripts_folder_path'] + '/' + filename, self.sim_folder_path)

    def get_models_save_folder(self):
        return self.models_save_folder

    def get_plots_save_folder(self):
        return self.plots_save_folder
