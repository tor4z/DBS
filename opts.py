import datetime
import yaml


class Opts(object):
    def __init__(self, cfg):
        with open('conf/' + cfg, 'r') as fo:
            self.cfg = yaml.load(fo, Loader=yaml.FullLoader)
        self.post_config()
        self.init_attr()

    def init_attr(self):
        for k, v in self.cfg.items():
            self.__setattr__(k, v)

    def post_config(self):
        if self.cfg['runtime_id'] is None:
            self.cfg['runtime_id'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    def __str__(self):
        string = ''
        for k, v in self.cfg.items():
            string += f'({k},{v})'
        return string

    __repr__ = __str__
