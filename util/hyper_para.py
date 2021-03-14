from argparse import ArgumentParser


def none_or_default(x, default):
    return x if x is not None else default

class HyperParameters():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Data parameters
        parser.add_argument('--yv_root', help='YouTubeVOS data root', default='../YouTube')
        parser.add_argument('--davis_root', help='DAVIS data root', default='../DAVIS')
        parser.add_argument('--bl_root', default='../BL30K')

        parser.add_argument('--fusion_root', default='../fusion_data/davis')
        parser.add_argument('--fusion_bl_root', default='../fusion_data/bl')

        parser.add_argument('--stage', type=int, default=0)

        # Generic learning parameters
        parser.add_argument('-i', '--iterations', help='Number of training iterations', default=None, type=int)
        parser.add_argument('-b', '--batch_size', help='Batch size', default=12, type=int)
        parser.add_argument('--lr', help='Initial learning rate', default=1e-4, type=float)
        parser.add_argument('--steps', help='Iteration at which learning rate is decayed by gamma', default=None, type=int, nargs='*')
        parser.add_argument('--gamma', help='Gamma used in learning rate decay', default=0.1, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_prop', default='saves/propagation_model.pth')
        parser.add_argument('--load_model', help='Path to the model file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        # Multiprocessing parameters
        parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        # Stage-dependent hyperparameters
        # Assign default if not given
        if self.args['stage'] == 0:
            self.args['iterations'] = none_or_default(self.args['iterations'], 30000)
            self.args['steps'] = none_or_default(self.args['steps'], [20000])
        elif self.args['stage'] == 1:
            self.args['iterations'] = none_or_default(self.args['iterations'], 10000)
            self.args['steps'] = none_or_default(self.args['steps'], [7500])

    def __getitem__(self, key):
        return self.args[key]

    def __str__(self):
        return str(self.args)
