// RUN: triton-opt -split-input-file %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm="compute-capability=80 ptx-version=80" -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_num_ctas_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_barrier
    llvm.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#s = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80"} {
  tt.func private @shared_i8_add(%dst: !ttg.memdesc<32xi8, #s, #ttg.shared_memory, mutable>, %v: tensor<32xi8, #b>, %i: tensor<32xi32, #b>) -> tensor<32xi8, #b> {
    // expected-error@+2 {{no width-preserving NVIDIA shared atomic add for 'i8' on SM80}}
    // expected-error@+1 {{failed to legalize operation 'ttg.local_atomic_scatter_rmw'}}
    %old = ttg.local_atomic_scatter_rmw add, %dst[%i], %v {axis = 0 : i32} : (!ttg.memdesc<32xi8, #s, #ttg.shared_memory, mutable>, tensor<32xi8, #b>, tensor<32xi32, #b>) -> tensor<32xi8, #b>
    tt.return %old : tensor<32xi8, #b>
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#s = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80"} {
  tt.func private @shared_i16_exchange(%dst: !ttg.memdesc<32xi16, #s, #ttg.shared_memory, mutable>, %v: tensor<32xi16, #b>, %i: tensor<32xi32, #b>, %m: tensor<32xi1, #b>) {
    // expected-error@+2 {{no width-preserving NVIDIA shared atomic exch for 'i16' on SM80}}
    // expected-error@+1 {{failed to legalize operation 'ttg.local_atomic_scatter_rmw'}}
    %old = ttg.local_atomic_scatter_rmw exch, %dst[%i], %v, %m {axis = 0 : i32} : (!ttg.memdesc<32xi16, #s, #ttg.shared_memory, mutable>, tensor<32xi16, #b>, tensor<32xi32, #b>, tensor<32xi1, #b>) -> tensor<32xi16, #b>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#s = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80"} {
  tt.func private @shared_bf16_add(%dst: !ttg.memdesc<32xbf16, #s, #ttg.shared_memory, mutable>, %v: tensor<32xbf16, #b>, %i: tensor<32xi32, #b>) {
    // expected-error@+2 {{no width-preserving NVIDIA shared atomic fadd for 'bf16' on SM80}}
    // expected-error@+1 {{failed to legalize operation 'ttg.local_atomic_scatter_rmw'}}
    %old = ttg.local_atomic_scatter_rmw fadd, %dst[%i], %v {axis = 0 : i32} : (!ttg.memdesc<32xbf16, #s, #ttg.shared_memory, mutable>, tensor<32xbf16, #b>, tensor<32xi32, #b>) -> tensor<32xbf16, #b>
    tt.return
  }
}
