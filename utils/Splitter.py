from inspect import signature


class Splitter:
    def __init__(self, function, **kwargs):
        self.kwargs = self.split_kwargs(function, **kwargs)

    def split_kwargs(self, function, **kwargs):
        args = [k for k, v in signature(function).parameters.items()]
        args_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in args}

        return args_dict