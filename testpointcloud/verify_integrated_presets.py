"""Render task defaults and compare FPS512 with approved experiment recordings."""
import json
from pathlib import Path
import time

from camera_experiment import HERE, ROOT, array, source_state, state_error
from multi_task_experiment import TASKS, labels_for
from obs_adapter import ObservationConfig, adapt_observation, config_from_contract, make_env, pointcloud_features
from mani_skill.utils.task_pointcloud import TASK_POINTCLOUD_PRESETS
import mani_skill.envs  # noqa: F401
import gymnasium as gym
import h5py
import numpy as np


def run():
    output = HERE/'runs/integration-defaults01.json'
    if output.exists(): raise FileExistsError(output)
    multi = json.loads((HERE/'runs/multi-final-report/analysis.json').read_text())
    pick = json.loads((HERE/'runs/final-report/analysis.json').read_text())
    result = {'tasks': {}, 'started': time.time()}
    for task, preset in TASK_POINTCLOUD_PRESETS.items():
        if task == 'PickCube-v1':
            source = ROOT/'.runtime/flow_dp3/pickcube-100-scene-v3.raw.h5'
            reference = pick['validation']['low_left256_60']['recording'];selected = [1, 9, 66];robot = 'panda'
        else:
            source = ROOT/'.runtime/crop-inspection-20261006-1'/task/(TASKS[task]['source']+'.h5')
            reference = multi['tasks'][task]['result']['recording'];selected = [0, 1, 2];robot = TASKS[task]['robot']
        metadata = json.loads(source.with_suffix('.json').read_text())
        by_id = {ep['episode_id']: ep for ep in metadata['episodes']}
        # No base_camera override: verifies the actual gym task defaults.
        env = gym.make(task, robot_uids=robot, num_envs=1, obs_mode='pointcloud',
                       control_mode='pd_joint_pos', sim_backend='physx_cpu', render_backend='cpu',
                       sensor_configs={'shader_pack':'default'}, reconfiguration_freq=1)
        records = []
        try:
            with h5py.File(source, 'r') as h, h5py.File(reference, 'r') as expected:
                for episode in selected:
                    env.reset(**by_id[episode]['reset_kwargs']);base = env.unwrapped
                    group = h[f'traj_{episode}'];length = len(group['actions'])
                    for frame in sorted(set([0, length//2, length])):
                        state = source_state(group, frame);base.set_state_dict(state)
                        if task == 'DrawTriangle-v1': base.draw_step = frame
                        assert state_error(env, state) < 1e-5
                        obs = base.get_obs();actual, detail = pointcloud_features(obs, base.agent, env_id=task)
                        known = expected[f'traj_{episode}/frame_{frame:05d}/fps512_features'][:]
                        error = float(np.abs(array(actual)[0]-known).max())
                        assert error < 1e-6, (task, episode, frame, error)
                        camera = base._sensors['base_camera'].config
                        assert (camera.width, camera.height) == (256,256)
                        assert abs(camera.fov - np.deg2rad(preset.fov_degrees)) < 1e-6
                        ext = array(obs['sensor_param']['base_camera']['extrinsic_cv'])[0]
                        eye = -ext[:, :3].T @ ext[:, 3]
                        assert np.allclose(eye, preset.eye, atol=1e-6)
                        assert actual.shape == (1,512,4)
                        records.append({'episode':episode,'frame':frame,'max_feature_error':error,
                                        'eye':eye.tolist(),**detail})
            result['tasks'][task] = {'passed':True,'native_sensors':list(env.unwrapped._sensors),
                                     'frames':records}
            if task == 'PegInsertionSide-v1': assert 'hand_camera' in result['tasks'][task]['native_sensors']
            print(task, 'default camera/FPS matches approved experiment', flush=True)
        finally: env.close()
    # Restore the old v1 camera and compare real rendered input to historical data.
    old_path = ROOT/'.runtime/flow_dp3/pickcube-100-scene-v3.h5'
    with h5py.File(old_path,'r') as old:
        contract = json.loads(old.attrs['contract']);config = config_from_contract(contract)
        episode = old['episode_00000'];meta = json.loads(episode.attrs['metadata'])
        expected_pc = episode['pointcloud_distance'][0];expected_state = episode['state'][0]
    source = ROOT/'.runtime/flow_dp3/pickcube-100-scene-v3.raw.h5'
    metadata = json.loads(source.with_suffix('.json').read_text())
    ep = next(e for e in metadata['episodes'] if e['episode_id']==meta['source_episode'])
    env = make_env(contract=contract)
    try:
        env.reset(**ep['reset_kwargs'])
        with h5py.File(source,'r') as h: env.unwrapped.set_state_dict(source_state(h[f'traj_{ep["episode_id"]}'],0))
        actual,_ = adapt_observation(env.unwrapped.get_obs(),env.unwrapped.agent,config)
        pc_error = float(np.abs(array(actual['pointcloud_distance'])[0]-expected_pc).max())
        state_err = float(np.abs(array(actual['state'])[0]-expected_state).max())
        assert pc_error < 1e-6 and state_err < 1e-6
        assert env.unwrapped._sensors['base_camera'].config.width == 128
        result['legacy'] = {'passed':True,'version':contract['version'],'width':128,
                            'pointcloud_error':pc_error,'state_error':state_err}
    finally: env.close()
    result['passed'] = True;result['elapsed_seconds'] = time.time()-result.pop('started')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('Passed all five default cameras, 45 reference frames, and real legacy replay',flush=True)


if __name__ == '__main__': run()
