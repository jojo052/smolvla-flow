"""Materialize a local LeRobot split without contacting the Hugging Face Hub.

The official edit-dataset command resolves the dataset codebase tag through the
Hub even when ``--root`` points at a local directory. This helper keeps the
same episode split, filters the local parquet shards, and writes self-contained
train/validation/test roots for offline training on the 4090 host.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _read_nested_parquet(directory: Path) -> pa.Table:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {directory}")
    return pq.read_table(files[0]) if len(files) == 1 else pa.concat_tables([pq.read_table(f) for f in files])


def _filter_rows(source_root: Path, episode_indices: list[int]) -> pa.Table:
    data_files = sorted((source_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet files under {source_root / 'data'}")

    values = pa.array(episode_indices, type=pa.int64())
    tables: list[pa.Table] = []
    for path in data_files:
        table = pq.read_table(path)
        mask = pc.is_in(table["episode_index"], value_set=values)
        selected = table.filter(mask)
        if selected.num_rows:
            tables.append(selected)

    if not tables:
        raise ValueError(f"no rows found for episodes {episode_indices[:5]}...")

    table = pa.concat_tables(tables, promote_options="default")
    order = pc.sort_indices(table, sort_keys=[("episode_index", "ascending"), ("frame_index", "ascending")])
    return table.take(order)


def _remap_column(table: pa.Table, name: str, values: list[int]) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        return table
    return table.set_column(index, name, pa.array(values, type=table[name].type))


def _materialize_one(
    *,
    source_root: Path,
    source_meta: pa.Table,
    source_info: dict,
    split_name: str,
    episode_indices: list[int],
    output_root: Path,
) -> dict:
    split_root = output_root / split_name
    if split_root.exists():
        shutil.rmtree(split_root)
    data_dir = split_root / "data" / "chunk-000"
    episodes_dir = split_root / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)

    rows = _filter_rows(source_root, episode_indices)
    old_to_new = {old: new for new, old in enumerate(episode_indices)}
    remapped_episodes = [old_to_new[int(value)] for value in rows["episode_index"].to_pylist()]
    rows = _remap_column(rows, "episode_index", remapped_episodes)
    rows = _remap_column(rows, "index", list(range(rows.num_rows)))
    pq.write_table(rows, data_dir / "file-000.parquet", compression="zstd")

    episode_mask = pc.is_in(source_meta["episode_index"], value_set=pa.array(episode_indices, type=pa.int64()))
    episodes = source_meta.filter(episode_mask)
    episode_order = pc.sort_indices(episodes, sort_keys=[("episode_index", "ascending")])
    episodes = episodes.take(episode_order)
    episodes = _remap_column(
        episodes,
        "episode_index",
        [old_to_new[int(value)] for value in episodes["episode_index"].to_pylist()],
    )
    episodes = _remap_column(episodes, "data/chunk_index", [0] * episodes.num_rows)
    episodes = _remap_column(episodes, "data/file_index", [0] * episodes.num_rows)
    episodes = _remap_column(episodes, "meta/episodes/chunk_index", [0] * episodes.num_rows)
    episodes = _remap_column(episodes, "meta/episodes/file_index", [0] * episodes.num_rows)

    frame_counts = rows["episode_index"].value_counts()
    count_by_episode = {int(item["values"]): int(item["counts"]) for item in frame_counts.to_pylist()}
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for new_index in range(episodes.num_rows):
        starts.append(cursor)
        cursor += count_by_episode[new_index]
        ends.append(cursor)
    episodes = _remap_column(episodes, "dataset_from_index", starts)
    episodes = _remap_column(episodes, "dataset_to_index", ends)
    pq.write_table(episodes, episodes_dir / "file-000.parquet", compression="zstd")

    meta_out = split_root / "meta"
    shutil.copy2(source_root / "meta" / "tasks.parquet", meta_out / "tasks.parquet")
    shutil.copy2(source_root / "meta" / "stats.json", meta_out / "stats.json")
    info = dict(source_info)
    info["total_episodes"] = episodes.num_rows
    info["total_frames"] = rows.num_rows
    info["splits"] = {split_name: f"0:{episodes.num_rows}"}
    info["data_files_size_in_mb"] = round((data_dir / "file-000.parquet").stat().st_size / 1_000_000, 3)
    # The current LIBERO parquet stores image bytes inline. LeRobot still
    # validates this metadata field as a positive value even when no external
    # video files are present, so preserve the source estimate (or use 1 MB as
    # a valid sentinel for an inline-image split).
    info["video_files_size_in_mb"] = max(1, int(source_info.get("video_files_size_in_mb", 1)))
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    return {
        "split": split_name,
        "episode_count": episodes.num_rows,
        "frame_count": rows.num_rows,
        "episode_indices": episode_indices,
        "root": str(split_root),
    }


def main() -> None:
    args = _parse_args()
    source_meta = _read_nested_parquet(args.source_root / "meta" / "episodes")
    source_info = json.loads((args.source_root / "meta" / "info.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for split_name in ("train", "validation", "test"):
        episode_indices = [int(value) for value in manifest["splits"][split_name]["episode_indices"]]
        results.append(
            _materialize_one(
                source_root=args.source_root,
                source_meta=source_meta,
                source_info=source_info,
                split_name=split_name,
                episode_indices=episode_indices,
                output_root=args.output_root,
            )
        )
    print(json.dumps({"output_root": str(args.output_root), "splits": results}, indent=2))


if __name__ == "__main__":
    main()
