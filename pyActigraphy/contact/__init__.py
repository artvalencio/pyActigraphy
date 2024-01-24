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
"""IO module for contact data."""

from .contact import ContactRecording
from .gendevice import GenContactDevice
from .gendevice import read_raw_gcd
from .contact_metrics import ContactMetricsMixin


__all__ = [
    "ContactMetricsMixin",
    "ContactRecording",
    "GenContactDevice",
    "read_raw_gcd"
]
