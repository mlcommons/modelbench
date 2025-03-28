from abc import ABC, abstractmethod


class AnnotatorSet(ABC):
    @property
    def annotators(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
