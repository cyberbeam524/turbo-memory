#!/bin/bash

# ============================
# CONFIG SECTION (EDIT BELOW)
# ============================

# Fill this on the worker node to point to head node
export MASTER_ADDR="172.21.0.2"
export MASTER_PORT="33"

# Interface name for NCCL to use (default is eth0 or podnet1 on RunPod)
export NCCL_SOCKET_IFNAME="eth0"

# Optional: disable InfiniBand (RunPod usually doesn’t support it)
export NCCL_IB_DISABLE=1

# Torch debug (optional)
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ============================
# SCRIPT STARTS
# ============================

echo "==== Interface Info ===="
ip -4 addr show "$NCCL_SOCKET_IFNAME" | grep inet

echo
echo "==== Environment Setup ===="
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

echo
echo "==== Port Test from Worker to Head ===="
if nc -zv "$MASTER_ADDR" "$MASTER_PORT"; then
    echo "✅ Port $MASTER_PORT is open on $MASTER_ADDR"
else
    echo "❌ Cannot reach $MASTER_ADDR:$MASTER_PORT"
    echo "Make sure ports are open and both nodes are in the same subnet."
fi

echo
echo "==== Optional: Run NCCL All-Reduce Test (1 GPU) ===="
if [ -f "./build/all_reduce_perf" ]; then
    ./build/all_reduce_perf -b 8 -e 64M -f 2 -g 1
else
    echo "⚠️  NCCL tests not found. You can clone with:"
    echo "git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make MPI=0"
fi

echo
echo "✅ Done."
