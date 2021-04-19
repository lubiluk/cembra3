from gym.envs.registration import register

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    # Fetch
    register(
        id="FetchSlide{}-v2".format(suffix),
        entry_point="gym_fetch.envs:FetchSlideEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="FetchPickAndPlace{}-v2".format(suffix),
        entry_point="gym_fetch.envs:FetchPickAndPlaceEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="FetchReach{}-v2".format(suffix),
        entry_point="gym_fetch.envs:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="FetchPush{}-v2".format(suffix),
        entry_point="gym_fetch.envs:FetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )