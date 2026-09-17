from fdb.challenge.deepmimic_training import training_config


def test_cpu_training_config_has_at_least_one_update() -> None:
    config = training_config(32768, 32)
    updates = config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs
    assert updates == 16
    assert config.experiment.env_params.use_mjwarp is False
    assert config.experiment.task_factory.params.lafan1_dataset_conf.dataset_name == ["walk1_subject1"]

