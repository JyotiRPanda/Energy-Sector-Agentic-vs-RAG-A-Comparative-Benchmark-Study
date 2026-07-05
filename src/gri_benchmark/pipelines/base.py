from __future__ import annotations

from abc import ABC, abstractmethod

from gri_benchmark.types import BenchmarkExample, Prediction


class QAPipeline(ABC):
    name: str

    @abstractmethod
    def answer(self, example: BenchmarkExample) -> Prediction:
        raise NotImplementedError
