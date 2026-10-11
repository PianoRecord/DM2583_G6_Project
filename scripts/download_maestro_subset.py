"""Download a fixed MAESTRO v3 subset via validated HTTP ranges; no full ZIP download.
Run with: python scripts/download_maestro_subset.py (from project root).
Selection and original ZIP index are stored beside the source manifest.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv, hashlib, io, json, math, struct, time, zlib
import requests
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

URL='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip'
ROOT=Path(__file__).resolve().parents[1]

def get_range(start,end):
    for attempt in range(4):
        try:
            r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=(20,180))
            r.raise_for_status()
            if r.status_code!=206 or len(r.content)!=end-start+1 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                raise ValueError('Server did not return the requested range')
            return r.content
        except (requests.RequestException,ValueError):
            if attempt==3: raise
            time.sleep(2*(attempt+1))


def download_one(row):
    target=ROOT/'data/problem2/audio'/row['local_file']
    if target.exists() and row.get('sha256')==hashlib.sha256(target.read_bytes()).hexdigest():
        return row
    entry=row['zip_entry']
    offset=entry['offset']
    head=get_range(offset,offset+29)
    if head[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP header')
    name_len,extra_len=struct.unpack_from('<HH',head,26)
    start=offset+30+name_len+extra_len
    compressed=get_range(start,start+entry['compressed']-1)
    raw=zlib.decompress(compressed,-15) if entry['method']==8 else compressed
    if len(raw)!=entry['size'] or (zlib.crc32(raw)&0xffffffff)!=entry['crc']:
        raise ValueError('ZIP CRC/size mismatch')
    audio,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
    mono=audio.mean(axis=1)
    divisor=math.gcd(sr,22050)
    mono=resample_poly(mono,22050//divisor,sr//divisor)
    if not np.isfinite(mono).all():raise ValueError('Nonfinite audio')
    temp=target.with_suffix('.part')
    sf.write(temp,mono,22050,format='FLAC',subtype='PCM_16')
    temp.replace(target)
    return {**row,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_wav_sha256':hashlib.sha256(raw).hexdigest(),
            'source_sample_rate':sr,'audio_duration_seconds':len(mono)/22050,
            'local_bytes':target.stat().st_size,'zip_crc_verified':True}


def main():
    (ROOT/'data/problem2/audio').mkdir(parents=True, exist_ok=True)
    manifest_path=ROOT/'data/problem2/sources.json'
    manifest=json.loads(manifest_path.read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(download_one,r):i for i,r in enumerate(manifest['tracks'])}
        for completed,future in enumerate(as_completed(jobs),1):
            i=jobs[future]
            manifest['tracks'][i]=future.result()
            manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
            print(f"{completed}/50 {manifest['tracks'][i]['canonical_composer']} — {manifest['tracks'][i]['canonical_title']}",flush=True)
    print('DOWNLOAD_COMPLETE',sum(r['local_bytes'] for r in manifest['tracks']),flush=True)
if __name__=='__main__':main()
