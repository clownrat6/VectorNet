import os
import re
import sys

class arg_parser(object):
    args_dict = {}
    key_list = []
    map_dict = \
    {

    }
    special_key_list = []
    def __init__(self):
        self.special_key_list += list(self.map_dict.keys())
        self.sys_args = sys.argv[1:]

    def add_map(self, arg_key, arg_map_dict):
        assert '--' in arg_key, 'Error, the arg key must be the parttern like --xxx'
        arg_key = re.search('--(.*?)\$', arg_key + '$').group(1)
        if(arg_key in self.special_key_list):
            print('The arg {} has been created.'.format(arg_key))
        else:
            self.map_dict[arg_key] = arg_map_dict
            self.special_key_list.append(arg_key)

    def add_val(self, arg_key, arg_default_val):
        """
        insert item to parse list.
        Args:
            arg_key: the arg which will be the key, and it must seem like --xxx.
            arg_default_val: give a prior val to key.
        Return:
            None
        """
        assert '--' in arg_key, 'Error, the arg key must be the parttern like --xxx'
        arg_key = re.search('--(.*?)\$', arg_key + '$').group(1)
        self.args_dict[arg_key] = arg_default_val
        self.key_list.append(arg_key)
        exec('self.{} = arg_default_val'.format(arg_key))

    def _convert(self, val, val_type):
        if(val_type == type(1)):
            val = int(val)
        if(val_type == type(1.1)):
            val = float(val)
        if(val_type == type(True)):
            val = True if(val == 'True') else False if(val == 'False') else None
        if(val_type == type(None)):
            val = val

        return val

    def __call__(self):
        """
        Extract the arg like --xxx as the key, and the next arg is the val.
        Args:
            sys_args: raw command line args.
        Return:
            args_dict: like {'--abc': 'abc'}
        """
        sys_args = self.sys_args
        i = 0
        while(i < len(sys_args)):
            arg = sys_args[i].replace('--', '')
            i += 1
            if(arg in self.special_key_list):
                val = sys_args[i]
                self.args_dict[arg] = self.map_dict[arg][val]
                exec('self.{} = self.args_dict[arg]'.format(arg))
                i += 1
                continue
            if(arg in self.key_list):
                val = sys_args[i]
                val = self._convert(val, type(self.args_dict[arg]))
                assert val != None, 'Error, arg {} get invalid type input.'.format(arg)
                self.args_dict[arg] = val
                exec('self.{} = self.args_dict[arg]'.format(arg))
                i += 1

        return self
