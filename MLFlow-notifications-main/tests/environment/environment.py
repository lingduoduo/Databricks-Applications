import json
import logging
import os
from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.clusters.api import ClusterApi
from databricks_cli.dbfs.api import DbfsApi
from databricks_cli.dbfs.dbfs_path import DbfsPath
from databricks_cli.repos.api import ReposApi
from databricks_cli.jobs.api import JobsApi

from config import configparser


class Environment:
    def __init__(self):
        self.api_client = ApiClient(
            host=os.getenv('DATABRICKS_HOST'),
            token=os.getenv('DATABRICKS_TOKEN')
        )

        # prepare config
        config_parser = configparser.ConfigParser(os.getenv('ENVIRONMENT'))
        self.config = config_parser.config

    def process_env(self) -> bool:
        return self.__validate()

    def __validate(self) -> bool:
        """Validates ML environment

        Returns:
            True if environment is healthy, false if something wrong with environment
        """

        log = ''
        is_valid = True

        log += self.__validate_cluster()
        log += self.__validate_directory()
        log += self.__validate_repos()
        log += self.__validate_jobs()

        if log:
            is_valid = False
            logging.log(logging.WARNING,
                        f'\n\n============================== VALIDATION RESULTS =============================='
                        f'\nFor environment there were found next issues:\n' + log)

        return is_valid

    def __validate_cluster(self) -> str:
        """Check if cluster exists and it's up
        """
        cluster_id = self.config.get('DEFAULT', 'CLUSTER_ID')

        clusters_api = ClusterApi(self.api_client)
        clusters_list = clusters_api.list_clusters()

        log = ''
        cluster_present = False

        for cluster in clusters_list['clusters']:
            if cluster['cluster_id'] == cluster_id:
                cluster_present = True

        if cluster_present is False:
            log += f'\tRequired cluster = {cluster_id} is not found'

        return log

    def __validate_directory(self) -> str:
        """Check if directory on it's place
        """
        dir_prefix = self.config.get('DEFAULT', 'DIR_PREFIX')
        dir_path = json.loads(self.config.get('DEFAULT', 'DIR_PATH'))

        dbfs_api = DbfsApi(self.api_client)

        log = ''

        for dir in dir_path:
            dbfs_path = DbfsPath(f'{dir_prefix}{dir}')
            result = dbfs_api.file_exists(dbfs_path)
            if result is False:
                log += f'\tRequired DBFS Dir = {dir_prefix}{dir} is not found'

        return log

    def __validate_repos(self) -> str:
        """Check if databricsk repo on it's place
        """
        log = ''

        repo_name = self.config.get('DEFAULT', 'REPO_NAME')
        repos_api = ReposApi(self.api_client)

        try:
            result = repos_api.get_repo_id(repo_name)
        except:
            log += f'\tRequired databricks repository = {repo_name} is not found'

        return log

    def __validate_jobs(self) -> str:
        """Check if job on it's place
        """
        log = ''

        job_id = self.config.get('DEFAULT', 'JOB_ID')
        jobs_api = JobsApi(self.api_client)

        try:
            result = jobs_api.get_job(job_id)
        except:
            log += f'\tRequired job = {job_id} is not found'

        return log

