import collections
import operator
import functools
import inspect
from argparse import Namespace

from typing import Any

from .selflib import util as RT


_ConsBase = collections.namedtuple('_ConsBase', ['name', 'rest'])
_SymbolBase = collections.namedtuple('_SymbolBase', ['name', 'value'])
_SelfBase = collections.namedtuple('_SelfBase', ['tag', 'rep'])



class Self(type):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""
    def __new__(cls, self, *args, **kwargs):
        RT = Runtime(cls, self, *args, **kwargs)
        result = [RT.typename(self), self, *args, kwargs]
        print('Self', cls, result)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Runtime(type):
    def __new__(cls, self, *args, **kwargs):
        print('get Runtime for', cls, [cls.typename(self), self, *args, kwargs])
        return cls

    @classmethod
    def get(cls, self, name):
        pass

    @classmethod
    def type(cls, self):
        return cls.typename(self)


    @classmethod
    def typename(cls, self):
        # print(f'typename({cls!r}, {self!r})')
        # if isinstance(self, mtftorch.Tensor):
        #     return self.type()

        module = ''
        class_name = ''
        if hasattr(self, '__module__') and self.__module__ != 'builtins' \
                and self.__module__ != '__builtin__' and self.__module__ is not None:
            module = self.__module__ + '.'

        if hasattr(self, '__qualname__'):
            class_name = self.__qualname__
        elif hasattr(self, '__name__'):
            class_name = self.__name__
        else:
            class_name = self.__class__.__name__

        return module + class_name


class TorchRuntime(Runtime):
    @classmethod
    def typename(cls, self):
        print('TorchRuntime.typename', cls, self, super())
        return super().typename(self)

    @classmethod
    def torch(cls, self):
        return


class Traits(type):
    def __new__(cls, self, *args, **kwargs):
        print('Traits', cls, self, args, kwargs)
        #return super().__new__(cls, *args, **kwargs)
        return

    def parents(cls):
        return []
    def scope(cls, self=None):
        if self is not None:
            cls = self.__class__
        return cls.__qualname__

class Call(type):
    def __new__(cls, self, *args, **kwargs):
        print('Call', cls, self, *args, kwargs)
        #return super().__new__(cls, *args, **kwargs)
        return self


class SymbolTraits(Traits):
    def __new__(cls, *args, **kwargs):
        print('__new__', cls, args, kwargs)
        return super().__new__(cls, *args, **kwargs)
    def parents(cls):
        return ['symbol'] + super().parents()

class Symbol(_SymbolBase, metaclass=SymbolTraits):
    @classmethod
    def call(cls, self, *args, **kwargs):
        print('call', cls, self, args, kwargs)

class KeywordType(type):
    def __new__(cls, *args, **kwargs):
        print(cls, *args, **kwargs)
    def __instancecheck__(cls, self):
        print('isinstance', cls, self)
        if isinstance(self, str) and len(self) > 1 and self[0] == ':':
            return True
        return super().__instancecheck__(self)


class ConsMeta(type):
    def __instancecheck__(cls, self):
        print('isinstance', cls, self)
        if isinstance(self, tuple) and len(self) == 2:
            return True
        return super().__instancecheck__(self)

class Cons(_ConsBase, metaclass=ConsMeta):
    def __new__(self, name=None, *rest, index=0, **kwargs):
        if kwargs:
            rest = Cons()
        if rest:
            rest = Cons(*rest, index = index + 1, **kwargs)
        if name is None and len(rest) == 0:
            return tuple()
        if stringlike(name):
            return name, *Cons(*rest)
        return *items(name), *Cons(*rest)


class NamedCons(Cons, metaclass=ConsMeta):
    pass



