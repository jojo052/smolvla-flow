from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from smolvla_flow.toy_flow import (
    VelocityMLP,
    euler_sample,
    make_flow_matching_batch,
    sample_conditional_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/toy_flow.pt"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = VelocityMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x_data, condition = sample_conditional_data(args.batch_size, device=device)
        x_t, flow_time, target_velocity = make_flow_matching_batch(x_data)
        predicted_velocity = model(x_t, flow_time, condition)
        loss = F.mse_loss(predicted_velocity, target_velocity)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 250 == 0:
            print(f"step={step:04d} loss={loss.item():.6f}")

    model.eval()
    condition = torch.arange(2, device=device).repeat_interleave(1_000)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sample_started = time.perf_counter()
    samples = euler_sample(model, condition, num_steps=args.sample_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sample_ms = (time.perf_counter() - sample_started) * 1_000

    means = [samples[condition == index].mean(dim=0).tolist() for index in range(2)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "sample_means": means}, args.output)
    print(f"device={device} train_seconds={time.perf_counter() - started:.2f}")
    print(f"sampling_steps={args.sample_steps} batch=2000 latency_ms={sample_ms:.3f}")
    print(f"condition_0_mean={means[0]}")
    print(f"condition_1_mean={means[1]}")
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    main()

