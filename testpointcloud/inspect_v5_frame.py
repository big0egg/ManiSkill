"""Label formal v5 frame 0 using renderer entity IDs and propagated pixel indices."""
import json
from dataclasses import asdict

from camera_experiment import HERE, ROOT, Camera, array, capture, labels_for, source_state
from analyse_cameras import plot_frame
from obs_adapter import make_env
import h5py
import numpy as np


def main():
    prefix = HERE/'runs/integration-camera-v5-semantics'
    recording = prefix.with_suffix('.h5');report = prefix.with_suffix('.json')
    if recording.exists() or report.exists():raise FileExistsError(prefix)
    data = ROOT/'.runtime/flow_dp3/pickcube-100-camera-v5.h5'
    with h5py.File(data,'r') as h:
        contract = json.loads(h.attrs['contract']);ep = h['episode_00000']
        meta = json.loads(ep.attrs['metadata']);expected = ep['pointcloud_distance'][0];length = len(ep['action'])
    source = ROOT/'.runtime/flow_dp3/pickcube-100-scene-v3.raw.h5'
    metadata = json.loads(source.with_suffix('.json').read_text())
    ep = next(e for e in metadata['episodes'] if e['episode_id']==meta['source_episode'])
    camera = Camera('camera_v5',(.3,-.3,.35),(0,0,.12),256,60)
    env = make_env(contract=contract)
    try:
        env.reset(**ep['reset_kwargs'])
        with h5py.File(source,'r') as h:
            env.unwrapped.set_state_dict(source_state(h[f'traj_{ep["episode_id"]}'],0))
        labels, links = labels_for(env)
        with h5py.File(recording,'w') as h:
            g = h.create_group('traj_0');g.attrs['entity_ids'] = json.dumps(labels);g.attrs['robot_links'] = json.dumps(links)
            row = capture(env,env.unwrapped.get_obs(),camera,g,0,length,ep['episode_id'],ep['episode_seed'],labels,'snapshot')
            error = float(np.abs(g['frame_00000/fps512_features'][:]-expected).max());assert error<1e-6
            row['prepared_data_feature_error'] = error
        report.write_text(json.dumps(row,indent=2)+'\n')
        results = {'camera_v5':{'recording':str(recording),'camera':asdict(camera)}}
        plot_frame(results,['camera_v5'],0,0,'Formal camera-v5 frame 0',prefix.with_suffix('.png'))
        print(json.dumps({k:v for k,v in row.items() if k.startswith(('raw_','pre_','fps512_'))},indent=2))
    finally:env.close()


if __name__=='__main__':main()
