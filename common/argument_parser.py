import argparse


class Parse:
    """
    Example
    parser._flag('local', 'Enables local run')
    parser._val('base_folder', '/Users/mkakodka/Code/Research/RBM_MATRYOSHKA/', 'Base Folder for all directory paths')
    parser._list('k', [1], "steps")
    """

    def __init__(self, description):
        self.___parser = argparse.ArgumentParser(description)
        self.___type_lambdas = {}

    def ___get_help(self, help, variable, default=None):
        return f"{help} \n Sets: Conf.{variable} \n Default: {default}"

    def ___set_type_lambda(self, variable, tl):
        if tl is not None:
            self.___type_lambdas[variable] = tl

    def flag(self, default=False, tl=None, help=None):
        def ___anon(variable):
            helptext = self.___get_help(help, variable)
            self.___set_type_lambda(variable, tl)
            self.___parser.add_argument('--%s' % variable.replace("_", "-"), dest=variable, action='store_const',
                                        const=not default, default=default,
                                        help=helptext)

        return ___anon

    def val(self, default, tl=None, help=None):
        def ___anon(variable):
            helptext = self.___get_help(help, variable, default)
            self.___set_type_lambda(variable, tl)
            self.___parser.add_argument('--%s' % variable.replace("_", "-"), dest=variable, action='store',
                                        default=default,
                                        help=helptext)

        return ___anon

    def list(self, default, tl=None, help=None):
        def ___anon(variable):
            helptext = self.___get_help(help, variable, default)
            self.___set_type_lambda(variable, tl)
            self.___parser.add_argument('--%s' % variable.replace("_", "-"), dest=variable, action='store',
                                        default=default, nargs='+',
                                        help=helptext)

        return ___anon

    def parse_top_level_arguments(self):
        args = self.___parser.parse_args()
        type_lambdas = self.___type_lambdas
        processed_args = {}
        for k, v in args.__dict__.items():
            if k in type_lambdas:
                tl = type_lambdas[k]
                if type(v) == list:
                    v = [tl(i) for i in v]
                else:
                    v = tl(v)
            processed_args[k] = v
        return processed_args
