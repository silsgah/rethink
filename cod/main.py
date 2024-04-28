import argparse
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.configer import Configer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')

   

    args_parser = parser.parse_args()

    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn

    configer = Configer(args_parser=args_parser)
    data_dir = configer.get('data', 'data_dir')
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    abs_data_dir = [os.path.expanduser(x) for x in data_dir]
    configer.update(['data', 'data_dir'], abs_data_dir)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)

    if configer.get('logging', 'log_to_file'):
        log_file = configer.get('logging', 'log_file')
        new_log_file = '{}_{}'.format(log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
        configer.update(['logging', 'log_file'], new_log_file)
    else:
        configer.update(['logging', 'logfile_level'], None)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    model = None
    if configer.get('method') == 'fcn_segmentor':
        if configer.get('phase') == 'train':
            from segmentor.trainer_contrastive import Trainer
            model = Trainer(configer)
        elif configer.get('phase') == 'test':
            from segmentor.tester import Tester 
            model = Tester(configer)    
        elif configer.get('phase') == 'test_offset':
            from segmentor.tester_offset import Tester
            model = Tester(configer)
    else:
        Log.error('Method: {} is not valid.'.format(configer.get('task')))
        exit(1)

    if configer.get('phase') == 'train':
        model.train()
    elif configer.get('phase').startswith('test') and configer.get('network', 'resume') is not None:
        model.test()
    else:
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        exit(1)