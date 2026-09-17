"""Download one selected published checkpoint and verify SHA256 before use."""
import argparse,hashlib,json,os,tempfile,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['w64d8','w128d10','w128d20'],required=True)
    p.add_argument('--protocol',choices=['detector','gt2d'],required=True)
    p.add_argument('--output-dir',type=Path,default=Path('weights'))
    a=p.parse_args()
    rows=json.loads((ROOT/'checkpoints/manifest.json').read_text(encoding='utf-8'))
    row=next(r for r in rows if r['model']==a.model and r['protocol']==a.protocol)
    a.output_dir.mkdir(parents=True,exist_ok=True);target=a.output_dir/row['filename']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
            raise SystemExit('Existing file has wrong hash; refusing to overwrite')
        print('Already verified:',target);return
    fd,tmp=tempfile.mkstemp(prefix='download-',suffix='.part',dir=a.output_dir)
    try:
        with os.fdopen(fd,'wb') as out,urllib.request.urlopen(row['url'],timeout=60) as response:
            h=hashlib.sha256();size=0
            for chunk in iter(lambda:response.read(1024*1024),b''):
                out.write(chunk);h.update(chunk);size+=len(chunk)
        if size!=row['bytes'] or h.hexdigest()!=row['sha256']:
            raise RuntimeError('Downloaded weight failed integrity check')
        if target.exists():raise RuntimeError('Destination appeared during download')
        Path(tmp).rename(target);print('Verified:',target)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()

if __name__=='__main__':main()
