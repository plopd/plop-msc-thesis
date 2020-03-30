#!/usr/bin/env python
"""Abstract environment base class for RL-Glue-py.
"""
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod


class BaseExperiment:

    __metaclass__ = ABCMeta

    @abstractmethod
    def init(self):
        """

        Returns:

        """

    @abstractmethod
    def save(self, path, data):
        """

        Returns:

        """

    @abstractmethod
    def run(self):
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
    def cleanup(self):
        """

        Returns:

        """

    @abstractmethod
    def message(self, message):
        """
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
