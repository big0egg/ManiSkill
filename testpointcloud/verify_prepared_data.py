"""Audit formal camera-v5 data: contract, temporal alignment, crop, distance and splits."""
import argparse
import json
from pathlib import Path

from camera_experiment import HERE, ROOT, sha256
from obs_adapter import config_from_contract
from dataset import DemoDataset
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'.runtime/flow_dp3/pickcube-100-camera-v5.h5')
    p.add_argument('--output',type=Path,default=HERE/'runs/integration-dataset01.json')
    args=p.parse_args();assert args.output.resolve().is_relative_to(HERE)
    if args.output.exists():raise FileExistsError(args.output)
    result={'frames':0,'actions':0,'episodes':0,'max_distance_error':0.,'max_old_state_error':0.,
            'max_old_action_error':0.,'matched_old_episodes':0,'dataset':str(args.dataset.resolve())}
    old_path=ROOT/'.runtime/flow_dp3/pickcube-100-scene-v3.h5'
    with h5py.File(args.dataset,'r') as h,h5py.File(old_path,'r') as old:
        contract=json.loads(h.attrs['contract']);config=config_from_contract(contract)
        assert contract['version']==2 and config.num_points==512
        assert contract['sensor_configs']['base_camera']['width']==256
        manifest=json.loads(h.attrs['manifest']);assert manifest['saved']==len(h)
        assert sha256(Path(manifest['source']))==manifest['source_sha256']
        assert sha256(Path(manifest['source']).with_suffix('.json'))==manifest['source_json_sha256']
        old_by_source={json.loads(g.attrs['metadata'])['source_episode']:g for g in old.values()}
        for name,g in h.items():
            pc=g['pointcloud_distance'][:];state=g['state'][:];action=g['action'][:]
            t=len(action);meta=json.loads(g.attrs['metadata']);assert meta['success_end']
            assert pc.shape==(t+1,512,4) and state.shape==(t+1,28) and action.shape==(t,4)
            assert all(np.isfinite(a).all() for a in [pc,state,action])
            assert np.abs(action).max()<=1.0001
            error=float(np.abs(np.linalg.norm(pc[...,:3],axis=-1)-pc[...,3]).max())
            assert error<1e-6;result['max_distance_error']=max(result['max_distance_error'],error)
            xyz=pc[...,:3]*config.length_scale+state[:,None,18:21]
            assert ((xyz>=np.array(config.crop_min)-1e-6)&(xyz<=np.array(config.crop_max)+1e-6)).all()
            assert xyz[...,2].min()>-.04  # Ground surface is around -0.92m.
            assert meta['min_cropped_points']>=512
            prior=old_by_source.get(meta['source_episode'])
            if prior is not None and prior['state'].shape==state.shape:
                result['max_old_state_error']=max(result['max_old_state_error'],float(np.abs(prior['state'][:]-state).max()))
                result['max_old_action_error']=max(result['max_old_action_error'],float(np.abs(prior['action'][:]-action).max()))
                result['matched_old_episodes']+=1
            result['episodes']+=1;result['frames']+=t+1;result['actions']+=t
        result['contract']=contract
    train=DemoDataset(args.dataset,split='train');val=DemoDataset(args.dataset,split='val')
    try:
        assert not set(train.episodes)&set(val.episodes)
        for data in [train,val]:
            for i in [0,len(data)//2,len(data)-1]:
                sample=data[i];assert sample['obs']['pointcloud_distance'].shape==(2,512,4)
                assert sample['action'].shape==(16,4)
        result['training_episodes']=len(train.episodes);result['validation_episodes']=len(val.episodes)
        result['training_windows']=len(train);result['validation_windows']=len(val)
    finally:train.close();val.close()
    result['passed']=True;result['sha256']=sha256(args.dataset)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='contract'},indent=2))


if __name__=='__main__':main()
