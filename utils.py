import numpy as np


class Args:
    def __init__(self) -> None:
        self.__args = {}

    def add(self, key, e):
        if key in self.__args.keys():
            print("key already exist")
            return None
        if e == None:
            print("key cannot be None")
            return None
        self.__args[key] = e

    def update(self, key, e):
        self.__args.update({key: e})

    def delete(self, key):
        if not key in self.__args.keys():
            return None
        ret = self.__args[key]
        del self.__args[key]
        return ret

    def get(self, key):
        if not key in self.__args.keys():
            return None
        return self.__args[key]

    def find_key(self, e):
        return list(self.__args.keys())[list(self.__args.values()).index(e)]
