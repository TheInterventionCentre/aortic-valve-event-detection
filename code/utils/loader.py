import importlib

def import_function(name):
    """ Loads the module (class) given by the input 'name' """
    *parts, fn = name.split('.')
    for idx, part in enumerate(parts):
        mod = '.'.join(parts[:idx+1])
        module = importlib.import_module(mod)
    return getattr(module, fn)

def load_class(cfg, *extra_args, post_arg=None, **extra_kwargs):
    """ Creates an instance of the python class specified at cfg['type'].

     Args:
        extra_args: The input arguments
        post_arg:  The post input arguments
        extra_kwargs: The input keyword arguments

     Returns:
         The class instance
     """
    fn = import_function(cfg['type'])
    args = tuple(extra_args) + tuple(cfg.get('args',tuple()))

    converted_args = []
    for arg in args:
        if isinstance(arg, dict) and len(arg)==1:
            converted_args.append(list(arg.values())[0])
        else:
            converted_args.append(arg)

    args = tuple(converted_args)

    if post_arg is not None:
        args = args+ (post_arg,)

    kwargs = {**cfg.get('kwargs',{}), **extra_kwargs}

    result = fn(*args, **kwargs)
    return result
