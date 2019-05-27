import getpass
import socket
import os


def get_abailoni_hci_home_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    if username == 'abailoni':
        if hostname == 'trendytukan':
            return '/net/hciserver03/storage/abailoni/'
        elif hostname == 'ialgpu01' or hostname == 'birdperson' or hostname == 'sirherny':
            return '/home/abailoni/hci_home/'
        elif hostname == 'quadxeon5':
            return '/srv/scratch/abailoni'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        return '/home/abailoni_local/hci_home/'
    else:
        return '/net/hciserver03/storage/abailoni/'

def get_trendytukan_drive_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    # print(username, hostname)
    if hostname == 'trendytukan' and username == 'abailoni':
        return '/mnt/localdata0/abailoni/'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        return '/home/abailoni_local/trendyTukan_localdata0/'
    elif (hostname == 'ialgpu01' or hostname == 'birdperson' or hostname == 'sirherny') and username == 'abailoni':
        return '/home/abailoni/trendyTukan_drive/'
    elif hostname == 'quadxeon5' and username == 'abailoni':
        return '/srv/scratch/abailoni'
    else:
        raise NotImplementedError("Trendytukan local drive not accessible by the current user")
        # return '/net/hciserver03/storage/abailoni/trendyTukan_drive/'


def get_source_dir():
    username = getpass.getuser()
    hostname = socket.gethostname()
    if username == 'abailoni' or username == 'abailoni_local':
        return os.path.join(get_abailoni_hci_home_path(), 'pyCharm_projects/quantized_vector_DT')
    elif username == 'claun':
        return os.path.join('/export/home/claun/', 'PycharmProjects/quantized_vector_DT')
    else:
        raise NotImplementedError("Set up your source_dir path for more hard-coded fun!")
