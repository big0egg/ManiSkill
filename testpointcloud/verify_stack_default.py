"""Check StackCube's unchanged default wrist-camera robot with the new base camera."""
import json
from camera_experiment import HERE, array
from obs_adapter import pointcloud_features
import mani_skill.envs  # noqa: F401
import gymnasium as gym
import numpy as np


def main():
    output = HERE/'runs/integration-stack-wristcam01.json'
    if output.exists(): raise FileExistsError(output)
    env = gym.make('StackCube-v1', num_envs=1, obs_mode='pointcloud',
                   sim_backend='physx_cpu', render_backend='cpu',
                   sensor_configs={'shader_pack':'default'}, reconfiguration_freq=1)
    frames = []
    try:
        for seed in [0,1,2]:
            obs, _ = env.reset(seed=seed);base = env.unwrapped
            assert base.robot_uids == 'panda_wristcam'
            assert list(base._sensors) == ['base_camera','hand_camera']
            assert base._sensors['base_camera'].config.width == 256
            assert base._sensors['hand_camera'].config.width == 128
            ext = array(obs['sensor_param']['base_camera']['extrinsic_cv'])[0]
            assert np.allclose(-ext[:, :3].T@ext[:, 3],(.3,.35,.28),atol=1e-6)
            features, detail = pointcloud_features(obs,base.agent,env_id='StackCube-v1')
            assert features.shape == (1,512,4)
            frames.append({'seed':seed,**detail})
    finally: env.close()
    output.write_text(json.dumps({'passed':True,'robot':'panda_wristcam','frames':frames},indent=2)+'\n')
    print('Stack default wrist-camera robot retains both sensors; FPS512 passed on three seeds')


if __name__ == '__main__': main()
