# -*- coding: UTF-8 -*-
from configparser import ConfigParser


class BaseParams(object):
    def __init__(self, conf_fp: str = 'configs/config.ini'):
        self.config = ConfigParser()
        self.config.read(conf_fp, encoding='utf8')


class ModelParams(BaseParams):
    def __init__(self, conf_fp: str = 'configs/config.ini'):
        super(ModelParams, self).__init__(conf_fp)
        section_name = 'gemini_configs'
        self.gemini_api_key = self.config.get(section_name, 'gemini_api_key')
        # self.temperature = self.config.get(section_name, 'temperature')
        # self.azure_endpoint = self.config.get(section_name, 'AZURE_ENDPOINT')
        # self.azure_openai_endpoint = self.config.get(section_name, 'AZURE_OPENAI_ENDPOINT')
        # self.azure_openai_version = self.config.get(section_name, 'AZURE_OPENAI_VERSION')
        # self.azure_openai_key = self.config.get(section_name, 'AZURE_OPENAI_API_KEY')
        # self.model = self.config.get(section_name, 'temperature')
