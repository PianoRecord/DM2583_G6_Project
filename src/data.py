"""Person A: official song split, beat chroma, contiguous valid 16-beat windows."""
from pathlib import Path
import hashlib
import json
import numpy as np

PITCH_CLASSES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def normalize_chroma(chroma, eps=1e-8):
    """L1 per beat. Invalid low-energy rows stay zero for window rejection."""
    x = np.asarray(chroma, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected finite nonnegative (beats,12) chroma')
    total = x.sum(-1, keepdims=True)
    return np.divide(x, total, out=np.zeros_like(x), where=total > eps)


def make_windows(chroma, length=16, return_starts=False):
    """Reject whole windows with invalid beats; never join across silence gaps."""
    if length < 2:
        raise ValueError('length must be >=2')
    x = normalize_chroma(chroma)
    count = len(x)//length
    blocks = x[:count*length].reshape(count,length,12)
    valid = (blocks.sum(-1) > 0).all(-1)
    windows, starts = blocks[valid], np.arange(count)[valid]*length
    return (windows,starts) if return_starts else windows


def split_songs(paths, test_fraction=0.2, seed=42):
    """Fallback for non-MAESTRO data only; caller must group repeated works."""
    paths = sorted(set(map(str, paths)))
    if len(paths)<2 or not 0<test_fraction<1:
        raise ValueError('Need >=2 songs and test_fraction in (0,1)')
    order=np.random.default_rng(seed).permutation(len(paths))
    n=min(len(paths)-1,max(1,round(len(paths)*test_fraction)))
    return [paths[i] for i in order[n:]], [paths[i] for i in order[:n]]


def official_split(paths, audio_dir, source_manifest):
    """Require complete metadata coverage, verified audio hashes and official labels."""
    audio_dir=Path(audio_dir)
    source=json.loads(Path(source_manifest).read_text())
    tracks=source['tracks']
    lookup={r['local_file']:r for r in tracks}
    if len(lookup)!=len(tracks):
        raise ValueError('Duplicate filenames in source manifest')
    actual={str(Path(p).relative_to(audio_dir)) for p in paths}
    if actual!=set(lookup):
        raise ValueError('Audio files and source manifest differ; update the source manifest first')
    split={'train':[], 'test':[]}
    seen_hashes, seen_titles={},{}
    for name in sorted(actual):
        r=lookup[name]; label=r['split']
        if label not in split:
            raise ValueError(f'Unsupported official split: {label}')
        path=audio_dir/name
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=r['sha256']:
            raise ValueError(f'Audio hash changed: {name}')
        title=(r['canonical_composer'].casefold().strip(),r['canonical_title'].casefold().strip())
        for key,seen in [(digest,seen_hashes),(title,seen_titles)]:
            if key in seen and seen[key]!=label:
                raise ValueError(f'Cross-split duplicate: {name}')
            seen[key]=label
        split[label].append(str(path))
    if not all(split.values()):
        raise ValueError('Official train/test split must both be nonempty')
    return split,lookup


def extract_chroma(path, sr=22050, hop_length=512, return_info=False,
                   silence_relative_db=-60.0, silence_absolute_rms=1e-5):
    """Average chroma between detected beats; reject quiet beat intervals using RMS.

    Relative threshold uses each recording's own peak frame RMS (no fitted corpus
    statistics). Invalid beats remain at their original indices as zero rows.
    """
    import librosa
    y,sr=librosa.load(path,sr=sr,mono=True)
    if not len(y) or not np.isfinite(y).all() or np.max(np.abs(y))<1e-8:
        raise ValueError('Empty, nonfinite or silent audio')
    tempo,beats=librosa.beat.beat_track(y=y,sr=sr,hop_length=hop_length)
    raw=librosa.feature.chroma_stft(y=y,sr=sr,hop_length=hop_length,norm=None)
    rms=librosa.feature.rms(y=y,frame_length=2048,hop_length=hop_length)[0]
    beats=np.unique(np.asarray(beats,dtype=int))
    beats=beats[(beats>=0)&(beats<min(raw.shape[1],len(rms)))]
    if len(beats)<17:
        raise ValueError('Fewer than 16 complete beat intervals')
    chroma=np.stack([raw[:,a:b].mean(1) for a,b in zip(beats[:-1],beats[1:])])
    beat_rms=np.array([np.sqrt(np.mean(rms[a:b]**2)) for a,b in zip(beats[:-1],beats[1:])])
    threshold=max(silence_absolute_rms,float(rms.max())*10**(silence_relative_db/20))
    invalid=(beat_rms<=threshold)|(chroma.sum(-1)<=1e-8)
    chroma[invalid]=0
    chroma=normalize_chroma(chroma)
    times=librosa.frames_to_time(beats,sr=sr,hop_length=hop_length)
    intervals=np.diff(times)
    info=dict(duration_seconds=len(y)/sr,tempo_bpm=float(np.asarray(tempo).reshape(-1)[0]),
              beat_intervals=len(chroma),invalid_beats=int(invalid.sum()),
              silence_rms_threshold=threshold,beat_times_seconds=times.tolist(),
              invalid_beat_indices=np.flatnonzero(invalid).tolist(),
              interval_median_seconds=float(np.median(intervals)),
              interval_cv=float(intervals.std()/intervals.mean()))
    return (chroma,info) if return_info else chroma


def prepare_dataset(audio_dir, output_dir, seed=42, test_fraction=0.2, source_manifest=None):
    """Auto-use sources.json when present. Write indexed, validated train/test arrays."""
    audio_dir,output_dir=Path(audio_dir),Path(output_dir)
    paths=sorted(p for p in audio_dir.rglob('*') if p.suffix.lower() in {'.wav','.flac','.mp3','.ogg','.m4a'})
    candidate=audio_dir.parent/'sources.json'
    source_manifest=Path(source_manifest) if source_manifest is not None else candidate if candidate.exists() else None
    if source_manifest is not None:
        splits,lookup=official_split(paths,audio_dir,source_manifest)
        policy='MAESTRO official train/test; no rerandomization'
    else:
        train,test=split_songs(paths,test_fraction,seed)
        splits,lookup={'train':train,'test':test},{}
        policy='seeded file split; user responsible for grouping repeated compositions'
    manifest=dict(version=2,split_policy=policy,seed=seed,shape=[16,12],dtype='float32',
                  pitch_classes=PITCH_CLASSES,sr=22050,hop_length=512,
                  silence_relative_db=-60.0,silence_absolute_rms=1e-5,
                  normalization='per-beat L1; reject entire windows containing invalid beats',
                  source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest() if source_manifest else None,
                  preprocessing_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  skipped=[],songs={},windows={})
    arrays={}
    for split,songs in splits.items():
        chunks,records,window_records=[],[],[]
        offset=0
        for song in songs:
            name=str(Path(song).relative_to(audio_dir))
            try:
                chroma,info=extract_chroma(song,return_info=True)
                windows,starts=make_windows(chroma,return_starts=True)
                record=dict(file=name,**info,windows=len(windows),candidate_windows=len(chroma)//16,
                            rejected_windows=len(chroma)//16-len(windows),tail_beats=len(chroma)%16,
                            row_start=offset,row_stop=offset+len(windows))
                if name in lookup:
                    record.update({k:lookup[name][k] for k in ['canonical_composer','canonical_title','split','sha256']})
                records.append(record)
                if not len(windows):
                    manifest['skipped'].append(dict(split=split,file=name,reason='No valid 16-beat windows'))
                else:
                    chunks.append(windows)
                    for start in starts:
                        window_records.append(dict(row=offset,file=name,start_beat=int(start),
                            start_seconds=info['beat_times_seconds'][start],end_seconds=info['beat_times_seconds'][start+16]))
                        offset+=1
                print(f'{split}: {name}: {len(windows)} windows, {info["invalid_beats"]} invalid beats',flush=True)
            except (ValueError,RuntimeError,OSError) as exc:
                manifest['skipped'].append(dict(split=split,file=name,reason=str(exc)))
        manifest['songs'][split]=records;manifest['windows'][split]=window_records
        arrays[split]=np.concatenate(chunks) if chunks else np.empty((0,16,12),dtype=np.float32)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    for split,x in arrays.items():
        if not len(x):raise ValueError('Empty split; inspect manifest.json')
        if not np.isfinite(x).all() or (x<0).any() or not np.allclose(x.sum(-1),1,atol=1e-5):
            raise ValueError('Invalid normalized windows')
    for split,x in arrays.items():np.save(output_dir/f'{split}.npy',x)
    return arrays
