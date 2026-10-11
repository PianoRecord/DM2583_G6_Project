import hashlib,json,tempfile,unittest
from pathlib import Path
from src.data import official_split

class OfficialSplitTests(unittest.TestCase):
    def test_official_labels_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);tracks=[];paths=[]
            for name,label,content in [('a.flac','test',b'a'),('b.flac','train',b'b')]:
                p=root/name;p.write_bytes(content);paths.append(p)
                tracks.append(dict(local_file=name,split=label,sha256=hashlib.sha256(content).hexdigest(),canonical_composer='composer',canonical_title=name))
            manifest=root/'sources.json';manifest.write_text(json.dumps({'tracks':tracks}))
            split,_=official_split(paths,root,manifest)
            self.assertEqual(split['train'],[str(root/'b.flac')])
            self.assertEqual(split['test'],[str(root/'a.flac')])
            tracks[1]['canonical_title']='a.flac'
            manifest.write_text(json.dumps({'tracks':tracks}))
            with self.assertRaisesRegex(ValueError,'Cross-split'):official_split(paths,root,manifest)
            tracks[1]['canonical_title']='b.flac'
            manifest.write_text(json.dumps({'tracks':tracks}))
            (root/'a.flac').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash changed'):official_split(paths,root,manifest)
