#############################################################################
# Adapted from:
# Copyright (c) 2022, Daylight Academy
# Author: Grégory Hammad
# Owner: Daylight Academy (https://daylight.academy)
# Maintainer: Grégory Hammad
# Email: gregory.hammad@uliege.be
# Status: development
#############################################################################
# Adaptation: Arthur Valencio, 2023
# The purpose of the adaptation is to create a class and methods for the
# newly developed contact device (capacitive sensors) included in the ActLumus.
# The class and methods could be similar to the ones used for the light device
# at the moment.
#############################################################################
# The development of a module for analysing light exposure
# data was led and financially supported by members of the Daylight Academy
# Project “The role of daylight for humans” (led by Mirjam Münch, Manuel
# Spitschan). The module is part of the Human Light Exposure Database. For
# more information about the project, please see
# https://daylight.academy/projects/state-of-light-in-humans/.
#
# This module is also part of the pyActigraphy software.
# pyActigraphy is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# pyActigraphy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
############################################################################
import warnings
from ..mask import _create_dummy_mask
from ..mask import _add_mask_period
from ..mask import _add_mask_periods
from ..recording import BaseRecording
from .contact_metrics import ContactMetricsMixin


class ContactRecording(ContactMetricsMixin, BaseRecording):
    """ Base class for contact recordings. Derives from
    :mod:`pyActigraphy.recording.BaseRecording`.

    Parameters
    ----------
    name: str
        Name of the contact recording.
    data: pandas.DataFrame
        Dataframe containing the contact data found in the recording.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        r"""Name of the contact recording."""
        return self.__name

    def get_channel(self, channel):
        r"""Contact channel accessor

        Get access to the requested channel.

        Parameters
        ----------
        channel: str.
            Channel to access.

        Returns
        -------
        contact: pd.Series
            Series with the requested channel.

        """
        if channel not in self.data.columns:
            raise ValueError(
                'The contact channel you tried to access ({}) '.format(channel)
                + 'is not available.\n Available channels:{}'.format(
                    '\n- {}'.format('\n- '.join(self.data.columns))
                )
            )

        return self.data.loc[:, channel]

    def get_channels(self, channels=None):
        r"""Contact channel accessor

        Get access to the requested channels.

        Parameters
        ----------
        channels: list of str, optional.
            Channel list. If set to None, use all available channels.
            Default is None.

        Returns
        -------
        contact: pd.DataFrame
            Dataframe with the requested channels.

        """

        # Select channels of interest
        if channels is None:
            channels_sel = self.data.columns
        else:
            # Check if some required channels are not available:
            channels_unavail = set(channels)-set(self.data.columns)
            if channels_unavail == set(channels):
                raise ValueError(
                    'None of the requested channels ({}) is available.'.format(
                        ', '.join(channels)
                    )
                )
            elif len(channels_unavail) > 0:
                warnings.warn(
                    'Required but unavailable channel(s): {}'.format(
                        ', '.join(channels_unavail)
                    )
                )
            channels_sel = [ch for ch in self.data.columns if ch in channels]

        return self.data.loc[:, channels_sel]

    def get_channel_list(self):
        r"""List of contact channels"""
        return self.data.columns

    def _check_contact_mask(self):
        """ Check if mask is not None"""
        if self.mask is None:
            raise ValueError(
                "No contact mask available. Please create one with the function"
                + " 'create_contact_mask' before adding mask periods."
            )

    def create_contact_mask(self):
        """Create a blank mask for all contact channels.

        This mask has the same length as its underlying data and can be used
        to offuscate meaningless periods.

        The mask is empty (filled with 1s) and is meant to be edited by adding
        mask periods manually.

        """

        # Create a mask filled with ones by default.
        self.mask = _create_dummy_mask(self.data)

    def add_contact_mask_period(self, start, stop, channel=None):
        """Add a masking period.

        This period extends between the specified start and stop times.
        It is possible to target a specific channel. If None is used, the
        masking period is set on all channels.

        Parameters
        ----------
        start: str
            Start time (YYYY-MM-DD HH:MM:SS) of the masking period.
        stop: str
            Stop time (YYYY-MM-DD HH:MM:SS) of the masking period.
        channel: str, optional
            Set masking period to a specific channel (i.e. column).
            If set to None, the period is set on all channels.
            Default is None.
        """

        # Check if mask is not None
        self._check_contact_mask()

        # Define correct channel
        if channel is not None:
            current_channel = channel
        else:
            current_channel = self.get_channel_list()

        # Add specified period
        _add_mask_period(
            self.mask,
            start=start,
            stop=stop,
            channel=current_channel
        )

    def add_contact_mask_periods(
        self, input_fname, channel=None, *args, **kwargs
    ):
        """Add masking periods from a file.

        Function to read masking periods (start and stop times) from a Mask log
        file. Supports different file format (.ods, .xls(x), .csv).

        Parameters
        ----------
        input_fname: str
            Path to the log file.
        channel: str, optional
            Set masking period to a specific channel (i.e. column).
            If set to None, the period is set on all channels.
            Default is None.
        *args
            Variable length argument list passed to the subsequent reader
            function.
        **kwargs
            Arbitrary keyword arguments passed to the subsequent reader
            function.
        """

        # Check if mask is not None
        self._check_contact_mask()

        # Define correct channel
        if channel is not None:
            current_channel = channel
        else:
            current_channel = self.get_channel_list()

        # Add specified period
        _add_mask_periods(
            input_fname=input_fname,
            mask=self.mask,
            channel=current_channel,
            *args, **kwargs
        )
