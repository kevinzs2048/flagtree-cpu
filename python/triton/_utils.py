# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import reduce


def get_iterable_path(iterable, path):
    return reduce(lambda a, idx: a[idx], path, iterable)


def set_iterable_path(iterable, path, val):
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    prev[path[-1]] = val


def find_paths_if(iterable, pred):
    from .language import core
    is_iterable = lambda x: isinstance(x, (list, tuple, core.tuple, core.tuple_type))
    ret = dict()

    def _impl(current, path):
        path = (path[0], ) if len(path) == 1 else tuple(path)
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl(item, path + (idx, ))
        elif pred(path, current):
            if len(path) == 1:
                ret[(path[0], )] = None
            else:
                ret[tuple(path)] = None

    if is_iterable(iterable):
        _impl(iterable, [])
    elif pred(list(), iterable):
        ret = {tuple(): None}
    else:
        ret = dict()
    return list(ret.keys())
