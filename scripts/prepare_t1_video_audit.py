"""Inventory old formal records and extract review sheets without new rollouts."""
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'artifacts/libero40_t1_video_audit_v1'
    out.mkdir(exist_ok=True)
    sources = {'T10': ('libero40_public_v1', 'smolvla_official'),
               'T5': ('libero40_t5_direct_v1', 'T5'),
               'S5': ('libero40_s5_b16_v1', 'S5'),
               'T2': ('libero40_t2_direct_v1', 'T2')}
    inventory, selected = {}, []
    for model, (directory, name) in sources.items():
        source = root / 'artifacts' / directory
        for path in sorted(source.rglob('*.json')):
            inventory[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
            for kind in ('success', 'failure'):
                candidates = sorted((source / 'formal' / name / suite).glob(f'task*/first_{kind}.mp4'))
                if not candidates:
                    selected.append({'model': model, 'suite': suite, 'kind': kind, 'missing': True})
                    continue
                video = candidates[0]
                rows = [json.loads(p.read_text()) for p in sorted(video.parent.glob('init*.json'))]
                matches = [r for r in rows if r.get('video') and Path(r['video']).name == video.name]
                # The frozen runner repeats the existing video path in later
                # rows; only the earliest matching outcome created the file.
                if not matches:
                    raise ValueError(f'Video has no matching row: {video}')
                row = matches[0]
                frame_count = int(subprocess.check_output(['ffprobe', '-v', 'error',
                    '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames',
                    '-of', 'default=nw=1:nk=1', str(video)], text=True))
                if frame_count != row['steps'] + 1:
                    raise ValueError(f'First-outcome video frame mismatch: {video}')
                duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                    'format=duration', '-of', 'default=nw=1:nk=1', str(video)], text=True))
                sheet = out / f'{model}_{suite}_{kind}.jpg'
                subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video), '-vf',
                    f'fps={8/duration},scale=256:-1,tile=4x2', '-frames:v', '1', str(sheet)], check=True)
                selected.append({'model': model, 'suite': suite, 'kind': kind, 'video': str(video),
                    'sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'sheet': str(sheet),
                    'task_name': row['task_name'], 'language': row['language'],
                    'init_index': row['init_index'], 'success': row['success'],
                    'visual_verdict': 'pending_review'})
    (out/'source_json_sha256.json').write_text(json.dumps(inventory, indent=2))
    (out/'selection.json').write_text(json.dumps(selected, indent=2))
    print(f'{len(selected)} selected video slots; {len(inventory)} JSON hashes')


if __name__ == '__main__':
    main()
