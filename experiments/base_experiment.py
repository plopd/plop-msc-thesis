#!/usr/bin/env python
"""Abstract environment base class for RL-Glue-py.
"""
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod


class BaseExperiment:

    __metaclass__ = ABCMeta

    @abstractmethod
    def init_experiment(self):
        """

        Returns:

        """

    @abstractmethod
    def save_experiment(self):
        """

        Returns:

        """

    @abstractmethod
    def run_experiment(self):
        """

        Returns:

        """

    @abstractmethod
    def learn(self):
        """

        Returns:

        """

    @abstractmethod
    def _learn(self, episode):
        """

        Returns:

        """

    @abstractmethod
    def cleanup_experiment(self):
        """

        Returns:

        """

    @abstractmethod
    def message_experiment(self, message):
        """
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
