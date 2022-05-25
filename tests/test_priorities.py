import random

import numpy as np

from commands.prioritize import Prioritizer, PriorityQueue


class TestQueue:
    def test_queue(self):
        q = PriorityQueue([1, 0, 2, 3])
        assert q.to_prioritized_list() == [1, 0, 2, 3]

        q = PriorityQueue([1, 0, 2, 3, 5, 4])
        assert q.to_prioritized_list() == [1, 0, 2, 3, 5, 4]

        q = PriorityQueue([6, 3, 0, 5, 4, 7, 1, 2, 9, 8])
        assert q.to_prioritized_list() == [6, 3, 0, 5, 4, 7, 1, 2, 9, 8]

    def test_normalize_priorities(self):
        q = PriorityQueue([1, 0.2, 2, 2.5])
        assert q.to_prioritized_list() == [1, 0, 2, 3]

    def test_transfert_queue(self):
        q = PriorityQueue([1, 0, 2, 3])
        q_copy = PriorityQueue([1, 0, 2, 3])
        q_res = PriorityQueue([])

        while q.is_empty() is False:
            q_res.append(q.pop())

        assert q_res.queue == q_copy.queue

    def test_transfert_priorities(self):
        prio = [1.0, 0.0, 2.0, 3.0]
        q = PriorityQueue(prio)
        q_res = PriorityQueue([])

        while q.is_empty() is False:
            q_res.append(q.pop())

        assert q_res.to_prioritized_list() == prio


class TestPrioritizer:
    def test_only_uncertainty_sampling(self):
        embeddings = np.random.rand(10, 10)
        predictions_probability = list([0.1, 0.2, 0.3, 0.4, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9])
        prioritizer = Prioritizer(embeddings, predictions_probability=predictions_probability)
        priorities = prioritizer.get_priorities(diversity_sampling=0, uncertainty_sampling=1)
        assert priorities == [8, 7, 6, 5, 9, 4, 3, 2, 1, 0]

    def test_combine_2_priorities(self):
        # initialize Prioritizer class
        embeddings = np.random.rand(10, 10)
        predictions_probability = list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        prioritizer = Prioritizer(embeddings, predictions_probability=predictions_probability)

        # test private method
        priorities = [list(range(10)) for _ in range(2)]
        [random.shuffle(list_) for list_ in priorities]

        combined_priorities = prioritizer.combine_priorities(
            priorities_a=priorities[0], priorities_b=priorities[1], proba_a=1  # type:ignore
        )
        assert combined_priorities == priorities[0]

        combined_priorities = prioritizer.combine_priorities(
            priorities_a=priorities[0], priorities_b=priorities[1], proba_a=0  # type:ignore
        )
        assert combined_priorities == priorities[1]

    def test_combine_multiple_priorities(self):
        # initialize Prioritizer class
        embeddings = np.random.rand(10, 10)
        predictions_probability = list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        prioritizer = Prioritizer(embeddings, predictions_probability=predictions_probability)

        # test private method
        priorities = [list(range(10)) for _ in range(3)]
        [random.shuffle(list_) for list_ in priorities]

        probas = [1.0, 0.0, 0.0]
        combined_priorities = prioritizer.combine_multiple_priorities(
            priorities, probas  # type:ignore
        )
        assert combined_priorities == priorities[0]

        probas = [0.0, 1.0, 0.0]
        combined_priorities = prioritizer.combine_multiple_priorities(
            priorities, probas  # type:ignore
        )
        assert combined_priorities == priorities[1]

        probas = [0.0, 0.0, 1.0]
        combined_priorities = prioritizer.combine_multiple_priorities(
            priorities, probas  # type:ignore
        )
        assert combined_priorities == priorities[2]
