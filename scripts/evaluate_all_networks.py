# Evaluate all networks using the pre-tuned logs
import argparse

from common import run_cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['intel-cpu', 'nvidia-gpu'])
    parser.add_argument('--log-file', type=str, required=True)
    args = parser.parse_args()

    if args.backend == 'intel-cpu':
        run_cmd('python3.6 tune_network.py --network resnet-50           --target "llvm -mcpu=skylake-avx512" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network resnet-50           --target "llvm -mcpu=skylake-avx512" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network tflite-mobilenet-v2 --target "llvm -mcpu=skylake-avx512" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network tflite-mobilenet-v2 --target "llvm -mcpu=skylake-avx512" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network resnet3d-18         --target "llvm -mcpu=skylake-avx512" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network resnet3d-18         --target "llvm -mcpu=skylake-avx512" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network dcgan               --target "llvm -mcpu=skylake-avx512" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network dcgan               --target "llvm -mcpu=skylake-avx512" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network bert                --target "llvm -mcpu=skylake-avx512" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network bert                --target "llvm -mcpu=skylake-avx512" --batch-size 16 --tune false --log-file %s' % args.log_file)
    elif args.backend == 'nvidia-gpu':
        run_cmd('python3.6 tune_network.py --network resnet-50           --target "cuda" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network resnet-50           --target "cuda" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network tflite-mobilenet-v2 --target "cuda" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network tflite-mobilenet-v2 --target "cuda" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network resnet3d-18         --target "cuda" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network resnet3d-18         --target "cuda" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network dcgan               --target "cuda" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network dcgan               --target "cuda" --batch-size 16 --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network bert                --target "cuda" --batch-size 1  --tune false --log-file %s' % args.log_file)
        run_cmd('python3.6 tune_network.py --network bert                --target "cuda" --batch-size 16 --tune false --log-file %s' % args.log_file)

