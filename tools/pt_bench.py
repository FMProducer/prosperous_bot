#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pt_bench.py — мини-бенчмарк PyTorch (CUDA)
- Матмул N×N
- Свёртка BxCin@HxW -> Cout (k×k)
Проверяет конфигурацию CUDA и печатает производительность.

Запуск (примеры):
  python pt_bench.py
  python pt_bench.py --matmul 4096 --iters 30
  python pt_bench.py --conv-b 16 --conv-cin 64 --conv-cout 128 --conv-hw 128 --conv-k 3 --iters 50
  python pt_bench.py --dtype fp32   # fp16 доступен не на всех GPU (Pascal медленный в fp16)
"""
import os, time, argparse
import torch
import torch.backends.cudnn as cudnn

def human_bool(v: str) -> bool:
    return v.lower() in ("1","true","yes","y","on")

def parse_args():
    p = argparse.ArgumentParser(description="Mini CUDA benchmark for PyTorch")
    p.add_argument("--matmul", type=int, default=4096, help="Матрица N×N для матмул (по умолчанию 4096)")
    p.add_argument("--conv-b", type=int, default=16, help="Batch для свёртки (по умолчанию 16)")
    p.add_argument("--conv-cin", type=int, default=64, help="Входные каналы свёртки (по умолчанию 64)")
    p.add_argument("--conv-cout", type=int, default=128, help="Выходные каналы свёртки (по умолчанию 128)")
    p.add_argument("--conv-hw", type=int, default=128, help="Высота/ширина входа свёртки (по умолчанию 128)")
    p.add_argument("--conv-k", type=int, default=3, help="Кернел свёртки (по умолчанию 3)")
    p.add_argument("--warmup", type=int, default=10, help="Итераций прогрева (по умолчанию 10)")
    p.add_argument("--iters", type=int, default=50, help="Итераций измерения (по умолчанию 50)")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32","fp16"], help="Тип вычислений")
    p.add_argument("--no-benchmark", action="store_true", help="Отключить cudnn.benchmark")
    return p.parse_args()

def dtype_from_flag(flag: str):
    return torch.float16 if flag.lower()=="fp16" else torch.float32

def bench_matmul(N=4096, warmup=5, iters=30, dtype=torch.float32):
    a = torch.randn(N, N, device="cuda", dtype=dtype)
    b = torch.randn(N, N, device="cuda", dtype=dtype)
    for _ in range(warmup):
        _ = a @ b
        torch.cuda.synchronize()
    total = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize()
        total += time.perf_counter() - t0
    dt = total / iters
    gflops = (2 * (N**3)) / dt / 1e9
    print(f"[MATMUL] N={N} {str(dtype).split('.')[-1]}: {dt*1e3:.2f} ms/it, {gflops:.1f} GFLOP/s")

def bench_conv(b=16, cin=64, cout=128, hw=128, k=3, warmup=10, iters=50, dtype=torch.float32):
    x = torch.randn(b, cin, hw, hw, device="cuda", dtype=dtype)
    conv = torch.nn.Conv2d(cin, cout, k, padding=k//2, bias=False).cuda().to(dtype)
    # прогрев
    for _ in range(warmup):
        _ = conv(x)
        torch.cuda.synchronize()
    # измерение
    total = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = conv(x)
        torch.cuda.synchronize()
        total += time.perf_counter() - t0
    dt = total / iters
    # приблизительные FLOPs: 2*B*H*W*Cin*Co*K*K
    flops = 2 * b * hw * hw * cin * cout * k * k
    gflops = flops / dt / 1e9
    print(f"[CONV] {b}x{cin}->{cout} {hw}x{hw} k{k} {str(dtype).split('.')[-1]}: {dt*1e3:.2f} ms/it, ~{gflops:.1f} GFLOP/s")

def main():
    assert torch.cuda.is_available(), "CUDA not available"
    args = parse_args()
    if not args.no_benchmark:
        cudnn.benchmark = True
    dt = dtype_from_flag(args.dtype)

    print(f"Torch {torch.__version__} | CUDA {torch.version.cuda} | cuDNN benchmark={ 'on' if cudnn.benchmark else 'off' }")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>')}")
    print("-"*60)

    bench_matmul(N=args.matmul, warmup=args.warmup, iters=args.iters, dtype=dt)
    bench_conv(b=args.conv_b, cin=args.conv_cin, cout=args.conv_cout, hw=args.conv_hw,
               k=args.conv_k, warmup=args.warmup, iters=args.iters, dtype=dt)

if __name__ == "__main__":
    main()
