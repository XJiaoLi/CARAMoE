from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


SCHEMA_VERSION = "china328_mask_overlay_v5.1"
REASON_NAMES = {
    0: "observed",
    1: "natural_missing",
    2: "continuous_history_interruption",
    3: "channel_failure",
    4: "station_failure",
}
EVENT_SPECS = {
    "continuous_history_interruption": {"reason_code": 2, "min_days": 3, "max_days": 24},
    "channel_failure": {"reason_code": 3, "min_days": 30, "max_days": 90},
    "station_failure": {"reason_code": 4, "min_days": 7, "max_days": 30},
}
EVENT_FIELDS = [
    "event_id",
    "event_type",
    "reason_code",
    "year",
    "station_index",
    "station_id",
    "pollutant_index",
    "pollutant",
    "start_index",
    "end_index",
    "start_date",
    "end_date",
    "duration_days",
    "region_cells",
    "naturally_observed_dropped",
]

CODE_ROOT = Path(__file__).resolve().parents[1]
CHINA_ROOT = CODE_ROOT.parent
DEFAULT_SOURCE = CHINA_ROOT / "processed_per_pollutant_zero_mask_v4_co_native" / "data"
DEFAULT_OUTPUT = (
    CHINA_ROOT
    / "processed_per_pollutant_zero_mask_v5_missing20_co_native"
    / "mask_overlays"
)


def decode_strings(values):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_source(source_path):
    source_path = Path(source_path)
    with h5py.File(source_path, "r") as handle:
        required = {"masks", "dates", "station_ids", "params"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"{source_path.name}: missing datasets {missing}")
        natural_mask = np.asarray(handle["masks"][:], dtype=np.bool_)
        dates = decode_strings(handle["dates"][:])
        station_ids = decode_strings(handle["station_ids"][:])
        params = decode_strings(handle["params"][:])
        frequency = handle.attrs.get("frequency", "")
        if isinstance(frequency, bytes):
            frequency = frequency.decode("utf-8")
    if natural_mask.ndim != 3:
        raise ValueError(f"{source_path.name}: masks must have shape (T,N,C)")
    if natural_mask.shape != (len(dates), len(station_ids), len(params)):
        raise ValueError(f"{source_path.name}: mask and metadata dimensions disagree")
    return natural_mask, dates, station_ids, params, str(frequency)


def event_region(event_type, start, end, station, pollutant):
    if event_type == "station_failure":
        return (slice(start, end), station, slice(None))
    return (slice(start, end), station, pollutant)


def append_event(
    events,
    event_type,
    natural_mask,
    synthetic_keep,
    drop_reason,
    dates,
    station_ids,
    params,
    year,
    station,
    pollutant,
    start,
    duration,
):
    end = start + duration
    region = event_region(event_type, start, end, station, pollutant)
    keep_view = synthetic_keep[region]
    if not np.all(keep_view):
        return 0
    natural_view = natural_mask[region]
    observed_count = int(natural_view.sum())
    if observed_count <= 0:
        return 0

    reason_code = EVENT_SPECS[event_type]["reason_code"]
    synthetic_keep[region] = False
    reason_view = drop_reason[region]
    reason_view[natural_view] = reason_code
    drop_reason[region] = reason_view

    event_id = f"Y{year}_{len(events) + 1:06d}"
    is_station = event_type == "station_failure"
    events.append(
        {
            "event_id": event_id,
            "event_type": event_type,
            "reason_code": reason_code,
            "year": year,
            "station_index": station,
            "station_id": station_ids[station],
            "pollutant_index": "" if is_station else pollutant,
            "pollutant": "ALL" if is_station else params[pollutant],
            "start_index": start,
            "end_index": end - 1,
            "start_date": dates[start],
            "end_date": dates[end - 1],
            "duration_days": duration,
            "region_cells": duration * (natural_mask.shape[2] if is_station else 1),
            "naturally_observed_dropped": observed_count,
        }
    )
    return observed_count


def add_random_category_events(
    event_type,
    target_count,
    rng,
    events,
    natural_mask,
    synthetic_keep,
    drop_reason,
    dates,
    station_ids,
    params,
    year,
):
    spec = EVENT_SPECS[event_type]
    time_count, station_count, pollutant_count = natural_mask.shape
    dropped = 0
    failures = 0
    max_failures = 20000

    while dropped < target_count and failures < max_failures:
        remaining = target_count - dropped
        duration = int(rng.integers(spec["min_days"], spec["max_days"] + 1))
        start = int(rng.integers(0, time_count - duration + 1))
        station = int(rng.integers(0, station_count))
        pollutant = None
        if event_type != "station_failure":
            pollutant = int(rng.integers(0, pollutant_count))
        region = event_region(event_type, start, start + duration, station, pollutant)
        if not np.all(synthetic_keep[region]):
            failures += 1
            continue
        observed_count = int(natural_mask[region].sum())
        if observed_count <= 0 or observed_count > remaining:
            failures += 1
            continue
        added = append_event(
            events,
            event_type,
            natural_mask,
            synthetic_keep,
            drop_reason,
            dates,
            station_ids,
            params,
            year,
            station,
            pollutant,
            start,
            duration,
        )
        dropped += added
        failures = 0
    return dropped


def choose_exact_block_duration(remaining, rng):
    if remaining < 3:
        raise RuntimeError(
            f"Exact structured completion requires at least 3 cells, got {remaining}"
        )
    if remaining <= 24:
        return remaining
    upper = min(24, remaining - 3)
    choices = [
        value for value in range(3, upper + 1) if remaining - value not in (1, 2)
    ]
    return int(rng.choice(choices))


def find_full_valid_block(natural_mask, synthetic_keep, duration, rng):
    time_count, station_count, pollutant_count = natural_mask.shape
    for _ in range(5000):
        station = int(rng.integers(0, station_count))
        pollutant = int(rng.integers(0, pollutant_count))
        start = int(rng.integers(0, time_count - duration + 1))
        region = (slice(start, start + duration), station, pollutant)
        if np.all(natural_mask[region]) and np.all(synthetic_keep[region]):
            return station, pollutant, start

    pair_order = rng.permutation(station_count * pollutant_count)
    kernel = np.ones(duration, dtype=np.int16)
    for flat_pair in pair_order:
        station = int(flat_pair // pollutant_count)
        pollutant = int(flat_pair % pollutant_count)
        eligible = natural_mask[:, station, pollutant] & synthetic_keep[:, station, pollutant]
        run_counts = np.convolve(eligible.astype(np.int16), kernel, mode="valid")
        starts = np.flatnonzero(run_counts == duration)
        if starts.size:
            return station, pollutant, int(rng.choice(starts))
    raise RuntimeError(f"No unmasked fully observed run of {duration} days remains")


def add_exact_continuous_blocks(
    target_count,
    rng,
    events,
    natural_mask,
    synthetic_keep,
    drop_reason,
    dates,
    station_ids,
    params,
    year,
):
    remaining = int(target_count)
    while remaining:
        duration = choose_exact_block_duration(remaining, rng)
        station, pollutant, start = find_full_valid_block(
            natural_mask, synthetic_keep, duration, rng
        )
        added = append_event(
            events,
            "continuous_history_interruption",
            natural_mask,
            synthetic_keep,
            drop_reason,
            dates,
            station_ids,
            params,
            year,
            station,
            pollutant,
            start,
            duration,
        )
        if added != duration:
            raise RuntimeError(f"Expected exact block of {duration}, added {added}")
        remaining -= added


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def generate_year(source_path, output_dir, year, target_rate, base_seed, force=False):
    natural_mask, dates, station_ids, params, frequency = read_source(source_path)
    total_count = int(natural_mask.size)
    natural_missing_count = int((~natural_mask).sum())
    target_missing_count = int(round(total_count * target_rate))
    required_synthetic = target_missing_count - natural_missing_count
    if required_synthetic <= 0:
        raise ValueError(
            f"{year}: natural missing count {natural_missing_count} already meets/exceeds "
            f"target {target_missing_count}"
        )

    overlay_path = output_dir / f"mask_overlay_{year}.npz"
    events_path = output_dir / f"mask_events_{year}.csv"
    if not force and (overlay_path.exists() or events_path.exists()):
        raise FileExistsError(
            f"{year}: output exists; use --force only when replacement is intentional"
        )

    year_seed = int(np.random.SeedSequence([base_seed, year]).generate_state(1)[0])
    rng = np.random.default_rng(year_seed)
    synthetic_keep = np.ones(natural_mask.shape, dtype=np.bool_)
    drop_reason = np.zeros(natural_mask.shape, dtype=np.uint8)
    drop_reason[~natural_mask] = 1
    events = []

    station_target = int(round(required_synthetic * 0.15))
    channel_target = int(round(required_synthetic * 0.25))
    station_dropped = add_random_category_events(
        "station_failure",
        station_target,
        rng,
        events,
        natural_mask,
        synthetic_keep,
        drop_reason,
        dates,
        station_ids,
        params,
        year,
    )
    channel_dropped = add_random_category_events(
        "channel_failure",
        channel_target,
        rng,
        events,
        natural_mask,
        synthetic_keep,
        drop_reason,
        dates,
        station_ids,
        params,
        year,
    )
    block_target = required_synthetic - station_dropped - channel_dropped
    add_exact_continuous_blocks(
        block_target,
        rng,
        events,
        natural_mask,
        synthetic_keep,
        drop_reason,
        dates,
        station_ids,
        params,
        year,
    )

    effective_mask = natural_mask & synthetic_keep
    final_missing_count = int((~effective_mask).sum())
    if final_missing_count != target_missing_count:
        raise RuntimeError(
            f"{year}: final missing {final_missing_count} != target {target_missing_count}"
        )
    source_hash = sha256_file(source_path)
    event_counts = Counter(event["event_type"] for event in events)
    reason_counts = np.bincount(drop_reason.ravel(), minlength=5)

    atomic_npz(
        overlay_path,
        schema_version=np.asarray(SCHEMA_VERSION),
        year=np.asarray(year, dtype=np.int16),
        target_missing_rate=np.asarray(target_rate, dtype=np.float64),
        source_sha256=np.asarray(source_hash),
        synthetic_keep_mask=synthetic_keep,
        effective_mask=effective_mask,
        drop_reason=drop_reason,
        dates=np.asarray(dates, dtype=np.str_),
        station_ids=np.asarray(station_ids, dtype=np.str_),
        params=np.asarray(params, dtype=np.str_),
    )
    atomic_csv(events_path, events)
    return {
        "year": year,
        "seed": year_seed,
        "shape": list(natural_mask.shape),
        "frequency": frequency,
        "total_cells": total_count,
        "natural_missing_cells": natural_missing_count,
        "natural_missing_rate": natural_missing_count / total_count,
        "synthetic_missing_cells": required_synthetic,
        "final_missing_cells": final_missing_count,
        "final_missing_rate": final_missing_count / total_count,
        "target_missing_cells": target_missing_count,
        "reason_cell_counts": {
            REASON_NAMES[index]: int(reason_counts[index]) for index in range(5)
        },
        "event_counts": dict(sorted(event_counts.items())),
        "requested_synthetic_mix": {
            "continuous_history_interruption": 0.60,
            "channel_failure": 0.25,
            "station_failure": 0.15,
        },
        "realized_synthetic_mix": {
            REASON_NAMES[index]: int(reason_counts[index]) / required_synthetic
            for index in (2, 3, 4)
        },
        "source_file": str(Path(source_path).resolve()),
        "source_sha256": source_hash,
        "overlay_file": overlay_path.name,
        "overlay_sha256": sha256_file(overlay_path),
        "events_file": events_path.name,
        "events_sha256": sha256_file(events_path),
    }


def validate_year(source_path, overlay_path, events_path, expected_target_rate=None):
    natural_mask, dates, station_ids, params, frequency = read_source(source_path)
    source_hash = sha256_file(source_path)
    with np.load(overlay_path, allow_pickle=False) as overlay:
        required = {
            "schema_version", "year", "target_missing_rate", "source_sha256",
            "synthetic_keep_mask", "effective_mask", "drop_reason", "dates",
            "station_ids", "params",
        }
        missing = sorted(required.difference(overlay.files))
        if missing:
            raise ValueError(f"{Path(overlay_path).name}: missing arrays {missing}")
        schema_version = str(np.asarray(overlay["schema_version"]).item())
        year = int(np.asarray(overlay["year"]).item())
        target_rate = float(np.asarray(overlay["target_missing_rate"]).item())
        stored_source_hash = str(np.asarray(overlay["source_sha256"]).item())
        synthetic_keep = np.asarray(overlay["synthetic_keep_mask"], dtype=np.bool_)
        effective_mask = np.asarray(overlay["effective_mask"], dtype=np.bool_)
        drop_reason = np.asarray(overlay["drop_reason"], dtype=np.uint8)
        overlay_dates = decode_strings(overlay["dates"])
        overlay_station_ids = decode_strings(overlay["station_ids"])
        overlay_params = decode_strings(overlay["params"])

    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"{year}: schema {schema_version!r} != {SCHEMA_VERSION!r}")
    if expected_target_rate is not None and not np.isclose(target_rate, expected_target_rate):
        raise ValueError(f"{year}: stored target rate {target_rate} != {expected_target_rate}")
    if stored_source_hash.upper() != source_hash:
        raise ValueError(f"{year}: source HDF5 hash differs from overlay metadata")
    if synthetic_keep.shape != natural_mask.shape:
        raise ValueError(f"{year}: synthetic_keep_mask shape mismatch")
    if effective_mask.shape != natural_mask.shape or drop_reason.shape != natural_mask.shape:
        raise ValueError(f"{year}: effective/reason shape mismatch")
    if overlay_dates != dates or overlay_station_ids != station_ids or overlay_params != params:
        raise ValueError(f"{year}: overlay metadata order differs from source HDF5")
    if not np.array_equal(effective_mask, natural_mask & synthetic_keep):
        raise ValueError(f"{year}: effective mask identity failed")

    covered = np.zeros(natural_mask.shape, dtype=np.bool_)
    expected_reason = np.zeros(natural_mask.shape, dtype=np.uint8)
    expected_reason[~natural_mask] = 1
    event_counts = Counter()
    observed_by_event_type = Counter()
    with Path(events_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            event_type = row["event_type"]
            if event_type not in EVENT_SPECS:
                raise ValueError(f"{year}: unknown event type {event_type!r}")
            if int(row["year"]) != year:
                raise ValueError(f"{year}: event {row['event_id']} stores a different year")
            station = int(row["station_index"])
            if row["station_id"] != station_ids[station]:
                raise ValueError(f"{year}: event {row['event_id']} station metadata mismatch")
            pollutant = None
            if event_type != "station_failure":
                pollutant = int(row["pollutant_index"])
                if row["pollutant"] != params[pollutant]:
                    raise ValueError(f"{year}: event {row['event_id']} pollutant mismatch")
            start = int(row["start_index"])
            end = int(row["end_index"]) + 1
            duration = int(row["duration_days"])
            spec = EVENT_SPECS[event_type]
            if duration != end - start or not (spec["min_days"] <= duration <= spec["max_days"]):
                raise ValueError(f"{year}: event {row['event_id']} has invalid duration")
            if row["start_date"] != dates[start] or row["end_date"] != dates[end - 1]:
                raise ValueError(f"{year}: event {row['event_id']} date bounds mismatch")
            region = event_region(event_type, start, end, station, pollutant)
            if np.any(covered[region]):
                raise ValueError(f"{year}: overlapping event {row['event_id']}")
            covered[region] = True
            observed_count = int(natural_mask[region].sum())
            if observed_count != int(row["naturally_observed_dropped"]):
                raise ValueError(f"{year}: event {row['event_id']} observed count mismatch")
            reason_view = expected_reason[region]
            natural_view = natural_mask[region]
            reason_view[natural_view] = spec["reason_code"]
            expected_reason[region] = reason_view
            event_counts[event_type] += 1
            observed_by_event_type[event_type] += observed_count

    if not np.array_equal(~synthetic_keep, covered):
        raise ValueError(f"{year}: synthetic mask contains cells not reconstructed by events")
    if not np.array_equal(drop_reason, expected_reason):
        raise ValueError(f"{year}: drop_reason differs from source and event ledger")
    total_count = int(natural_mask.size)
    target_count = int(round(total_count * target_rate))
    final_missing_count = int((~effective_mask).sum())
    if final_missing_count != target_count:
        raise ValueError(f"{year}: final missing {final_missing_count} != target {target_count}")

    return {
        "year": year,
        "status": "valid",
        "frequency": frequency,
        "shape": list(natural_mask.shape),
        "target_missing_rate": target_rate,
        "final_missing_rate": final_missing_count / total_count,
        "final_missing_cells": final_missing_count,
        "natural_missing_cells": int((~natural_mask).sum()),
        "synthetic_missing_cells": int((natural_mask & ~synthetic_keep).sum()),
        "event_counts": dict(sorted(event_counts.items())),
        "event_observed_drop_counts": dict(sorted(observed_by_event_type.items())),
        "source_sha256": source_hash,
        "overlay_sha256": sha256_file(overlay_path),
        "events_sha256": sha256_file(events_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate/validate CHINA328 v5 structured mask overlays at 20% total missingness"
    )
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--years", nargs="+", type=int, default=[2017, 2018, 2019, 2020, 2021])
    parser.add_argument("--target-missing-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_missing_rate < 1.0:
        parser.error("--target-missing-rate must be between 0 and 1")
    if len(args.years) != len(set(args.years)):
        parser.error("--years must be unique")
    args.years = sorted(args.years)
    return args


def main():
    args = parse_args()
    source_dir = args.source_data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source data directory not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        validations = []
        for year in args.years:
            validations.append(
                validate_year(
                    source_dir / f"aq_compact_{year}.h5",
                    output_dir / f"mask_overlay_{year}.npz",
                    output_dir / f"mask_events_{year}.csv",
                    args.target_missing_rate,
                )
            )
        print(json.dumps(validations, ensure_ascii=False, indent=2))
        return

    original_hashes_before = {
        year: sha256_file(source_dir / f"aq_compact_{year}.h5") for year in args.years
    }
    year_manifests = []
    for year in args.years:
        source_path = source_dir / f"aq_compact_{year}.h5"
        generated = generate_year(
            source_path,
            output_dir,
            year,
            args.target_missing_rate,
            args.seed,
            force=args.force,
        )
        validation = validate_year(
            source_path,
            output_dir / generated["overlay_file"],
            output_dir / generated["events_file"],
            args.target_missing_rate,
        )
        generated["validation"] = validation["status"]
        year_manifests.append(generated)
        print(
            f"{year}: valid | final_missing={generated['final_missing_rate']:.9f} | "
            f"synthetic={generated['synthetic_missing_cells']} | events={sum(generated['event_counts'].values())}"
        )

    original_hashes_after = {
        year: sha256_file(source_dir / f"aq_compact_{year}.h5") for year in args.years
    }
    if original_hashes_before != original_hashes_after:
        raise RuntimeError("A source HDF5 hash changed during mask generation")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_version": "processed_per_pollutant_zero_mask_v5_missing20_co_native",
        "description": "Independent structured input-mask overlay; v4 values remain read-only",
        "target_total_missing_rate": args.target_missing_rate,
        "base_seed": args.seed,
        "years": args.years,
        "reason_codes": {str(key): value for key, value in REASON_NAMES.items()},
        "event_definitions": EVENT_SPECS,
        "synthetic_mix_targets": {
            "continuous_history_interruption": 0.60,
            "channel_failure": 0.25,
            "station_failure": 0.15,
        },
        "source_data_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_files_unchanged": True,
        "source_sha256_before": original_hashes_before,
        "source_sha256_after": original_hashes_after,
        "year_manifests": year_manifests,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
