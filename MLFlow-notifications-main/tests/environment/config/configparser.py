import configparser


class ConfigParser:

    def __init__(self, env):
        """
            This class is using for parsing config
            And giving appropriate output which can be used in rest of flow
        """
        self.config = None
        self.parse_config(None, env)

    def parse_config(self, config_path, env):
        """
            Parse config
        """

        if config_path is None:
            config_path = f'./tests/environment/conf/{env}.config'

        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path)
        self.config = config

        return self.config

    def get_config_param(self, section, option):
        """
            Get config record by section and option
        """
        if section is None:
            section = 'DEFAULT'

        return self.config.get(section, option)
