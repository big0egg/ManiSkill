"""Audit saved pixel indices, segmentation, crop and physical distance features."""
import argparse
import json
from pathlib import Path

from camera_experiment import HERE, CROP_MIN, CROP_MAX
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(HERE) and not args.output.exists()
    report = {'frames':0,'sample_arrays':0,'max_xyz_reconstruction_error':0.,'max_distance_norm_error':0.,'runs':{}}
    for folder in args.inputs:
        manifest=json.loads((folder/'manifest.json').read_text());assert manifest.get('completed')
        summary=json.loads((folder/'summary.json').read_text());count=0
        for name, entry in summary.items():
            with h5py.File(entry['recording'],'r') as h:
                for trajectory in h.values():
                    labels=json.loads(trajectory.attrs['entity_ids'])
                    for frame in trajectory.values():
                        row=json.loads(frame.attrs['statistics']);xyz=frame['xyz_base_crop'][:]
                        crop_ids=frame['crop_source_indices'][:];seg=frame['segmentation_image'][:].ravel()
                        assert np.all(np.isfinite(xyz)) and np.all((xyz>=np.array(CROP_MIN)-1e-6)&(xyz<=np.array(CROP_MAX)+1e-6))
                        assert len(crop_ids)==len(xyz)==row['crop_total'] and np.all(np.diff(crop_ids)>0)
                        assert len(frame['raw_cube_xyz_base'])==row['raw_cube']
                        # All cube pixels are within far=100m; invalid background pixels do not affect this assertion.
                        assert int(np.isin(seg,labels['cube']).sum())==row['raw_cube']
                        for n in [512,1024]:
                            if not row[f'fps{n}_valid']:continue
                            p=frame[f'fps{n}_features'][:];ids=frame[f'fps{n}_source_indices'][:]
                            sampled_seg=frame[f'fps{n}_segmentation'][:]
                            assert p.shape==(n,4) and np.isfinite(p).all() and len(np.unique(ids))==n
                            locations=np.searchsorted(crop_ids,ids);assert np.array_equal(crop_ids[locations],ids)
                            assert np.array_equal(seg[ids],sampled_seg)
                            error=float(np.abs(p[:,:3]+frame['tcp_base'][:]-xyz[locations]).max())
                            normerror=float(np.abs(np.linalg.norm(p[:,:3],axis=-1)-p[:,3]).max())
                            assert error<1e-6 and normerror<1e-6
                            for group in ['cube','robot','wrist','table','ground']:
                                assert int(np.isin(sampled_seg,labels[group]).sum())==row[f'fps{n}_{group}']
                            assert row[f'fps{n}_ground']==0
                            report['max_xyz_reconstruction_error']=max(report['max_xyz_reconstruction_error'],error)
                            report['max_distance_norm_error']=max(report['max_distance_norm_error'],normerror)
                            report['sample_arrays']+=1
                        count+=1
        assert count==sum(v['frames'] for v in summary.values())
        report['frames']+=count;report['runs'][str(folder)]=count
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
