"""Person A entrypoint: python scripts/prepare_person_a.py (no model training)."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from src.data import prepare_dataset,PITCH_CLASSES

def main():
    data=ROOT/'data/problem2';out=ROOT/'results/problem2/data';out.mkdir(parents=True,exist_ok=True)
    arrays=prepare_dataset(data/'audio',data/'processed')
    manifest=json.loads((data/'processed/manifest.json').read_text())
    summary={'split_policy':manifest['split_policy'],'skipped':manifest['skipped'],'splits':{}}
    for split,x in arrays.items():
        songs=manifest['songs'][split]
        summary['splits'][split]=dict(shape=list(x.shape),dtype=str(x.dtype),songs_with_windows=sum(r['windows']>0 for r in songs),
            windows=len(x),invalid_beats=sum(r['invalid_beats'] for r in songs),
            rejected_windows=sum(r['rejected_windows'] for r in songs),tail_beats=sum(r['tail_beats'] for r in songs),
            finite=bool(np.isfinite(x).all()),minimum=float(x.min()),max_row_sum_error=float(np.abs(x.sum(-1)-1).max()))
        assert len(manifest['windows'][split])==len(x)
        for w in manifest['windows'][split]:
            record=next(r for r in songs if r['file']==w['file'])
            assert not set(range(w['start_beat'],w['start_beat']+16))&set(record['invalid_beat_indices'])
        assert x.dtype==np.float32 and np.isfinite(x).all() and (x>=0).all() and np.allclose(x.sum(-1),1,atol=1e-5)
    train_files={r['file'] for r in manifest['songs']['train']}
    test_files={r['file'] for r in manifest['songs']['test']}
    assert not train_files&test_files
    # Training examples only, selected before evaluating any model.
    songs=sorted(manifest['songs']['train'],key=lambda r:r['interval_cv'])
    chosen=[songs[0],songs[len(songs)//2],songs[-1]]
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    for ax,r in zip(axes,chosen):
        idx=r['row_start'];x=arrays['train'][idx]
        im=ax.imshow(x.T,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma')
        ax.set_yticks(range(12),PITCH_CLASSES);ax.set_xticks(range(16),range(1,17))
        ax.set(xlabel='Beat within window',ylabel='Pitch class',title=f"{r['file']} | {r['canonical_composer']} | window row {idx}")
    fig.colorbar(im,ax=axes,label='L1-normalized chroma',shrink=.8)
    fig.savefig(out/'real_chroma_heatmap.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    checks=[]
    for ax,r in zip(axes,chosen):
        y,sr=librosa.load(data/'audio'/r['file'],sr=22050,mono=True)
        times=np.array(r['beat_times_seconds'])
        start=float(times[4]);stop=min(start+12,len(y)/sr)
        local=times[(times>=start)&(times<=stop)]
        onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=512)
        ot=librosa.frames_to_time(np.arange(len(onset)),sr=sr,hop_length=512)
        mask=(ot>=start)&(ot<=stop)
        ax.plot(ot[mask],onset[mask],color='steelblue',label='Onset strength')
        for j,t in enumerate(local):ax.axvline(t,color='crimson',alpha=.55,label='Detected beat' if j==0 else None)
        ax.set(title=f"{r['file']} | estimated {r['tempo_bpm']:.1f} BPM | interval CV {r['interval_cv']:.3f}",xlabel='Seconds',ylabel='Onset strength')
        ax.legend(loc='upper right')
        checks.append(dict(file=r['file'],start_seconds=start,end_seconds=stop,tempo_bpm=r['tempo_bpm'],interval_cv=r['interval_cv']))
    fig.savefig(out/'beat_tracking_spotcheck.png',dpi=160);plt.close(fig)
    summary['beat_spotchecks']=checks
    summary['beat_check_limitation']='Visual diagnostic only; no ground-truth beat annotations. Half/double-tempo and rubato errors remain possible.'
    (out/'validation.json').write_text(json.dumps(summary,indent=2))
    rows=[{k:r[k] for k in ['file','canonical_composer','canonical_title','split','tempo_bpm','beat_intervals','invalid_beats','windows','rejected_windows','tail_beats','interval_cv']} for group in manifest['songs'].values() for r in group]
    with (out/'song_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('PERSON_A_DATA_OK',json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
