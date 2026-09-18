from fdb.challenge.deepmimic_training import training_config


def test_cpu_training_config_has_at_least_one_update() -> None:
    config = training_config(32768, 32)
    updates = config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs
    assert updates == 16
    assert config.experiment.env_params.use_mjwarp is False
    assert config.experiment.env_params.horizon >= 580
    assert config.experiment.task_factory.params.lafan1_dataset_conf.dataset_name == ["walk1_subject1"]


def test_behavioral_stage_selects_contact_reward() -> None:
    config = training_config(32768, 32, behavioral_reward=True)
    assert config.experiment.env_params.reward_type == "BehavioralMimicReward"
    assert config.experiment.env_params.reward_params.single_support_weight > 0


def test_position_control_and_subject_five_are_selectable() -> None:
    config = training_config(32768, 32, True, True, "walk1_subject5")
    assert config.experiment.env_params.control_type == "PDControl"
    assert len(config.experiment.env_params.control_params.p_gain) == 19
    assert config.experiment.task_factory.params.lafan1_dataset_conf.dataset_name == ["walk1_subject5"]
