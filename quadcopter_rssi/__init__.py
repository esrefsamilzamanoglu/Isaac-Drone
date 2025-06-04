import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Drone-Simple-RSSI-Seeker",
    entry_point=f"{__name__}.simple_rssi_seeker:QuadcopterRSSIEnv",       
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_rssi_seeker:QuadcopterRSSIEnvCfg",  
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lstm_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Sionna-RSSI-Seeker",
    entry_point=f"{__name__}.sionna_rssi_seeker:QuadcopterRSSIEnv",       
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sionna_rssi_seeker:QuadcopterRSSIEnvCfg",  
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lstm_finetune_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
