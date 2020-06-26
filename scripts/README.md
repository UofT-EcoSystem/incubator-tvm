# Help
## Tune single operator / subgraph

- Intel CPU  
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 100 --target "llvm -mcpu=core-avx2"
  ```

- NVIDIA GPU  
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 100 --target "cuda"
  ```

- ARM CPU  
  ```
  python3 tune_test.py --wkl matmul-512 --n-trials 100 --target "llvm -target=arm-linux-gnueabihf -mattr=+neon" --rpc-device-key rasp4b --rpc-host kraken --rpc-port 9191
  ```

## Tune Network
- Intel CPU  
  ```
  python3 tune_network.py --network resnet-18 --n-trials 200 --target "llvm -mcpu=core-avx2"
  ```

- NVIDIA GPU  
  ```
  python3 tune_network.py --network resnet-18 --n-trials 200 --target "cuda"
  ```

- ARM CPU  
  ```
  python3 tune_network.py --network resnet-18 --n-trials 200 --target "llvm -target=arm-linux-gnueabihf -mattr=+neon" --rpc-device-key rasp4b --rpc-host kraken --rpc-port 9191
  ```

## Run single op & subgraph evaluation
- Intel CPU
  ```
  # tune
  python3 tune_op_subgraph.py --wkl all --target "llvm -mcpu=core-avx2" --n-trials-per-shape 1000 

  # replay
  python3 tune_op_subgraph.py --wkl all --target "llvm -mcpu=core-avx2" --tune false
  ```

