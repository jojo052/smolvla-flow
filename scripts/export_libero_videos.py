"""Copy completed LIBERO video artifacts and verify every file by SHA256."""
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    destination = Path('/Volumes/DataLabWork/projects/smolvla-flow/LIBERO40_T10_T5_S5_视频')
    destination.mkdir(exist_ok=False)
    sources = {'T10': ('libero40_public_v1', 'smolvla_official', 68),
               'T5': ('libero40_t5_direct_v1', 'T5', 68),
               'S5': ('libero40_s5_b16_v1', 'S5', 66)}
    inventory = []; counts = {}
    for model, (folder, original_model, expected) in sources.items():
        source = root / 'artifacts' / folder
        formal = source / 'formal' / original_model
        videos = sorted(formal.rglob('*.mp4'))
        assert len(videos) == expected, (model, len(videos))
        counts[model] = len(videos)
        files = [(source / 'manifest.json', Path(model) / 'manifest.json')]
        files += [(p, Path(model) / p.relative_to(formal)) for p in sorted(formal.rglob('*')) if p.is_file() and p.suffix in ('.mp4', '.json')]
        for src, relative in files:
            dst = destination / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            sha = digest(src)
            assert digest(dst) == sha, relative
            inventory.append({'file': str(relative), 'bytes': dst.stat().st_size, 'sha256': sha})
    (destination / 'SHA256_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False))
    lines = ['# LIBERO40 T10 / T5 / S5 视频', '', '仅复制已有正式评测，不重新运行。', '',
             '- T10：官方模型，10次积分。', '- T5：官方模型直接减为5次积分。', '- S5：蒸馏学生，5次积分。',
             '- 每套件10任务，每任务10回合；每任务只保存首个成功和首个失败视频。没有该类型结果时，不补跑。',
             '- first_success.mp4 / first_failure.mp4 为已保存视频；initNN.json 为各回合记录。JSON中的远端video路径保留原样，可按同目录视频文件名查找。',
             '- 视频以20 FPS编码，不能用播放时长推断真实推理速度。',
             '- 所有复制文件经过源/目标SHA256比对；清单见SHA256_inventory.json。', '', '## 视频数量', '']
    lines += [f'- {m}: {n}' for m, n in counts.items()]
    (destination / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'destination': str(destination), 'videos': counts, 'verified_files': len(inventory), 'bytes': sum(x['bytes'] for x in inventory)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
