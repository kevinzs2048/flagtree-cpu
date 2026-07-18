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

import os

from ..backends import backends
from ..backends import DriverBase


def _create_driver():
    if os.getenv("TRITON_CPU_BACKEND", "0") == "1":
        if "cpu" not in backends:
            raise RuntimeError("TRITON_CPU_BACKEND is set, but CPU backend is unavailable.")
        return backends["cpu"].driver()

    actives = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(actives) >= 2 and backends["cpu"].driver.is_active():
        print("Both CPU and GPU backends are available. Using the GPU backend.")
        actives.remove(backends["cpu"].driver)
    if len(actives) != 1:
        raise RuntimeError(f"{len(actives)} active drivers ({actives}). There should only be one.")
    return actives[0]()


class LazyProxy:

    def __init__(self, init_fn):
        self._init_fn = init_fn
        self._obj = None

    def _initialize_obj(self):
        if self._obj is None:
            self._obj = self._init_fn()

    def __getattr__(self, name):
        self._initialize_obj()
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name in ["_init_fn", "_obj"]:
            super().__setattr__(name, value)
        else:
            self._initialize_obj()
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        self._initialize_obj()
        delattr(self._obj, name)

    def __repr__(self):
        if self._obj is None:
            return f"<{self.__class__.__name__} for {self._init_fn} not yet initialized>"
        return repr(self._obj)

    def __str__(self):
        self._initialize_obj()
        return str(self._obj)


class DriverConfig:

    def __init__(self):
        self.default = LazyProxy(_create_driver)
        self.active = self.default

    def set_active(self, driver: DriverBase):
        self.active = driver

    def reset_active(self):
        self.active = self.default

    def set_active_to_cpu(self):
        if "cpu" not in backends:
            raise RuntimeError("CPU backend is unavailable")
        self.active = backends["cpu"].driver()

    def set_active_to_gpu(self):
        active_gpus = [(name, backend.driver)
                       for name, backend in backends.items()
                       if backend.driver.is_active() and name != "cpu"]
        if len(active_gpus) != 1:
            raise RuntimeError(f"{len(active_gpus)} active GPU drivers ({active_gpus}). There should only be one GPU.")
        self.active = active_gpus[0][1]()
        return active_gpus[0][0]

    def get_active_gpus(self):
        return [name for name, backend in backends.items() if backend.driver.is_active() and name != "cpu"]


driver = DriverConfig()
