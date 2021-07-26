import numpy as np
import torch

class Test(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self,male):
        print(self.name, self.age)
        print(male)

if __name__ == '__main__':
    test = Test("aa",10)
    print(test.__dict__)


