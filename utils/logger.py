import os
import math

class logger(object):
    def __init__(self, log_base_dir, log_list):
        self.detail_log = open('{}/{}'.format(log_base_dir, 'detail.log'), 'a')
        self.target_only_dict = {}
        for log_target in log_list:
            target_log_path = '{}/{}'.format(log_base_dir, '{}.log'.format(log_target))
            self.target_only_dict[log_target] = target_log_path
    
    def metric_log_list(self, key):
        return self.target_only_dict[key]

    def detail_write(self, info):
        self.detail_log.write(info + '\n')

    def target_save(self, epoch, key, val):
        with open(self.target_only_dict[key], 'a') as f:
            f.write('epoch{} {} {}\n'.format(epoch, key, val))