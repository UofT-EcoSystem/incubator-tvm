# Ansor: An Auto Scheduler for TVM

## Get Started
This is a fork of the official tvm repo. Please follow the [official doc](https://tvm.apache.org/docs/install/index.html) to build this fork.

### Tune a matmul on cpu
script : [/scripts/tune_test.py](https://github.com/merrymercy/Ansor/blob/master/scripts/tune_test.py)

* Tune:  
  `python3 tune_test.py --wkl matmul-512 --n-trials 100`  
  This command will run search and do measurement for 100 trials.
  All logs are saved to `matmul-512.json` eagerly, so you can kill this program at any point if you don't want to wait.

* Replay history best:  
  `python3 tune_test.py --wkl matmul-512 --tune false`  
  This command will load the history best from `matmul-512.json` and test it.
  It will print equivalent schedule code in TVM's python API, lowered TVM IR, and evalution results.

#### Sample output for replay
Please run tuning first to get `matmul-512.json`. Then we can reply the history best by
```
python3 tune_test.py --wkl matmul-512 --tune false
```
Here is a sample output:

```
==================== Equivalent Python Schedule Code ====================
i, j, k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
C_local, = s.cache_write([C], "local")
i_c, j_c, k = tuple(C_local.op.axis) + tuple(C_local.op.reduce_axis)
i_c_o_i, i_c_i = s[C_local].split(i_c, factor=1)
i_c_o_o_i, i_c_o_i = s[C_local].split(i_c_o_i, factor=16)
i_c_o_o_o, i_c_o_o_i = s[C_local].split(i_c_o_o_i, factor=16)
j_c_o_i, j_c_i = s[C_local].split(j_c, factor=1)
j_c_o_o_i, j_c_o_i = s[C_local].split(j_c_o_i, factor=64)
j_c_o_o_o, j_c_o_o_i = s[C_local].split(j_c_o_o_i, factor=4)
k_o, k_i = s[C_local].split(k, factor=4)
s[C_local].reorder(i_c_o_o_o, j_c_o_o_o, i_c_o_o_i, j_c_o_o_i, k_o, i_c_o_i, j_c_o_i, k_i, i_c_i, j_c_i)
i_o_i, i_i = s[C].split(i, factor=16)
i_o_o, i_o_i = s[C].split(i_o_i, factor=16)
j_o_i, j_i = s[C].split(j, factor=64)
j_o_o, j_o_i = s[C].split(j_o_i, factor=4)
s[C].reorder(i_o_o, j_o_o, i_o_i, j_o_i, i_i, j_i)
s[C_local].compute_at(s[C], j_o_i)
i_o_o_j_o_o_fused_i_o_i_fused_j_o_i_fused = s[C].fuse(i_o_o, j_o_o, i_o_i, j_o_i)
s[C].parallel(i_o_o_j_o_o_fused_i_o_i_fused_j_o_i_fused)
s[C_local].vectorize(j_c_i)
s[C_local].pragma(i_c_o_o_o, "auto_unroll_max_step", 64)
s[C_local].pragma(i_c_o_o_o, "unroll_explicit", True)

==================== Lowered TIR ====================
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: handle, float32, [512, 512], []),
             B: Buffer(B_2: handle, float32, [512, 512], []),
             A: Buffer(A_2: handle, float32, [512, 512], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused: int32, 0, 256) "parallel" {
    attr [C.local: handle] "storage_scope" = "local";
    allocate(C.local, float32, [1024]) {
      for (i.c.outer.inner.init: int32, 0, 16) {
        for (j.c.outer.inner.init: int32, 0, 64) {
          C.local[((i.c.outer.inner.init*64) + j.c.outer.inner.init)] = 0f32
        }
      }
      for (k.outer: int32, 0, 128) {
        for (i.c.outer.inner: int32, 0, 16) {
          for (j.c.outer.inner: int32, 0, 64) {
            C.local[((i.c.outer.inner*64) + j.c.outer.inner)] = ((float32*)C.local[((i.c.outer.inner*64) + j.c.outer.inner)]) + ((float32*)A_2[((((floordiv(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128)*131072) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 64), 4)*8192)) + (i.c.outer.inner*512)) + (k.outer*4))])*(float32*)B_2[((((k.outer*2048) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128), 64)*256)) + (floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*64)) + j.c.outer.inner)])))
            C.local[((i.c.outer.inner*64) + j.c.outer.inner)] = ((float32*)C.local[((i.c.outer.inner*64) + j.c.outer.inner)]) + ((float32*)A_2[(((((floordiv(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128)*131072) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 64), 4)*8192)) + (i.c.outer.inner*512)) + (k.outer*4)) + 1)])*(float32*)B_2[(((((k.outer*2048) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128), 64)*256)) + (floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*64)) + j.c.outer.inner) + 512)])))
            C.local[((i.c.outer.inner*64) + j.c.outer.inner)] = ((float32*)C.local[((i.c.outer.inner*64) + j.c.outer.inner)]) + ((float32*)A_2[(((((floordiv(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128)*131072) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 64), 4)*8192)) + (i.c.outer.inner*512)) + (k.outer*4)) + 2)])*(float32*)B_2[(((((k.outer*2048) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128), 64)*256)) + (floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*64)) + j.c.outer.inner) + 1024)])))
            C.local[((i.c.outer.inner*64) + j.c.outer.inner)] = ((float32*)C.local[((i.c.outer.inner*64) + j.c.outer.inner)]) + ((float32*)A_2[(((((floordiv(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128)*131072) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 64), 4)*8192)) + (i.c.outer.inner*512)) + (k.outer*4)) + 3)])*(float32*)B_2[(((((k.outer*2048) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128), 64)*256)) + (floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*64)) + j.c.outer.inner) + 1536)])))
          }
        }
      }
      for (i.inner: int32, 0, 16) {
        for (j.inner: int32, 0, 64) {
          C_2[((((((floordiv(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128)*131072) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 64), 4)*8192)) + (i.inner*512)) + (floordiv(floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 128), 64)*256)) + (floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*64)) + j.inner)] = (float32*)C.local[((i.inner*64) + j.inner)])
        }
      }
    }
  }
}

==================== Evaluate ====================
Best schedule: 262.09 GFLOPS	cost: 1.024 ms
```

### Tune a network on cpu
script : [/scripts/tune_network.py](https://github.com/merrymercy/Ansor/blob/master/scripts/tune_network.py)

* Tune:  
  Tuning a whole network can be time consuming. Here we provide two example commands. One for quick testing and one 
  for better performance with enough time budget.
  - A fast run for debug / testing  
  `python3 tune_network.py --network resnet-18 --n-trials 200`
  - A slow run to get better results  
  `python3 tune_network.py --network resnet-18 --n-trials 10000`

  The above commands will tune for all workloads in a network jointly.
  `n_trials` is the sum of measurement trials for all workload.
  The log is saved to `resnet-18-B1-llvm.json`.

* Replay the history best:  
  `python3 tune_network.py --network resnet-18 --tune false`  
  This command will load the tuning log from `resnet-18-B1-llvm.json`, compile the network,
  and print the evaluation results.


#### Sample output
Please run tuning first to get `resnet-18-B1-llvm.json`. Then we can evaluate the tuning results by
```
python3 tune_network.py --network resnet-18 --tune false
```

Here is a sample output:
```
=============== Compile ===============
=============== Compile Finish ===============
========== Evaluate ==========
Mean inference time (std dev): 26.49 ms (0.34 ms)
```

### More tuning instructions
More tuning instructions can be found at [/scripts/README.md](https://github.com/merrymercy/Ansor/blob/master/scripts/README.md)

## Project structure
- `scripts`: scripts to run auto-scheduling
- `python/tvm/ansor`: Python frontend API
  - `auto_schedule.py`: User interface to use the auto-scheduler
  - `cost_model`: Python part code of the cost model. We use some python code 
                  because most machine learning frameworks are in python
  - `compute_dag.py`: Compute declaration graph and its related analysis tools
  - `dispatcher.py`: A global context to dispatch configurations. This is migrated from the old autotvm.
  - `env.py`: The scope to store global variables 
  - `feature.py`: Feature extraction for the cost model
  - `measure.py`: Python part code of measurement infrastructure. We use python's multiprocessing and exception handling.
  - `relay_integratoin.py`: Integrate ansor into Relay and TOPI
  - `serialization.py`: IO utilities for tuning records
  - `task_scheduler.py`: Task scheduler which tunes multiple workloads jointly. This is implemented in pure python.
  - `utils.py`: Other utilities
  - `workload_registry.py`: Workload registry
- `src/tvm/ansor`: C++ core
  - `cost_model`: Cost models
  - `search_policy`: Search policies
    - `sketch_search_policy.h`: The core search policy
	- `utils.h`: The common utilities for search policy
	- `search_policy.h`: The base class for search policy
  - `auto_cheduler.h`: User interface to use the auto-scheduler.
  - `compute_dag.h`: Compute declaration graph and its related analysis tools
  - `feature.h`: Feature extraction for the cost model
  - `loop_state.h`: An simplified loop structure IR for search. This defines the "state" for the search problem.
  - `measure.h`: Measurement infrastructure.
  - `search_task.h`: Meta information for a search task & Hardware parameters.
  - `serialization.h`: Json serialization format for dumping and loading tuning records
  - `transform_steps.h`: Schedule primitives (i.e. transform steps) for our IR. This defines the "action" for the search problem.
  - `utils.h`: Other common utilities

