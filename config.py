# !/usr/bin/env python
# --------------------------------------------------------------
# File:          config.py
# Project:       RLRC
# Created:       Thursday, 23rd July 2020 10:58:11 pm
# @Author:       Molin Liu, MSc in Data Science, University of Glasgow
# Contact:       molin@live.cn
# Last Modified: Thursday, 23rd July 2020 10:58:12 pm
# Copyright  © Rockface 2019 - 2020
# --------------------------------------------------------------

from configparser import ConfigParser


class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()
        raw_config.read(config_file)
        self.cast_values(raw_config)

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                setattr(self, key, val)
