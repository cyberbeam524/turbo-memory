



pip install ray transformers datasets evaluate ray[train] pandas ray[default] scikit-learn transformers[torch] 'accelerate>=0.26.0'

apt update && apt install lsof -y
apt update && apt install vim -y
apt update && apt install iputils-ping -y
apt update && apt install netcat -y
apt update && apt install redis-server -y


export MASTER_ADDR=jkieqsjelapn0l.runpod.internal

export MASTER_ADDR=10.0.119.84
export MASTER_PORT=33
export NCCL_SOCKET_IFNAME=podnet1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache


---------huggingface tricks:---------------------------------------------------------
rm -rf ~/.cache/huggingface
ln -s /workspace/hf_cache ~/.cache/huggingface
---------------------------------------------------------


echo "HF_HOME=$HF_HOME, TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE, HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

    
### on head node:
r0ixqd7l4lroyd.runpod.internal
ssh eysfrdqrjotk7g-64410e75@ssh.runpod.io -i /Users/maaruni/runpod_ed

ip addr | grep inet
    inet 127.0.0.1/8 scope host lo
    inet6 ::1/128 scope host 
    inet 10.0.119.84/10 scope global podnet1
    inet6 fe80::bf3f:f396:31fa:5b8e/64 scope link stable-privacy 
    inet 172.23.0.2/16 brd 172.23.255.255 scope global eth0

### on worker node:
ssh mpmalwyl5c5yoi-64410e75@ssh.runpod.io -i /Users/maaruni/runpod_ed

ip addr | grep inet
    inet 127.0.0.1/8 scope host lo
    inet6 ::1/128 scope host 
    inet 10.0.116.15/10 scope global podnet1
    inet6 fe80::8947:32a8:103c:f5a2/64 scope link stable-privacy 
    inet 172.23.0.3/16 brd 172.23.255.255 scope global eth0

### on worker 2
ssh hyfa6v6cjt8gn1-64410eb0@ssh.runpod.io -i /Users/maaruni/runpod_ed
hyfa6v6cjt8gn1.runpod.internal
ip addr | grep inet
    inet 127.0.0.1/8 scope host lo
    inet6 ::1/128 scope host
    inet 10.1.114.1/10 scope global podnet1
    inet6 fe80::d12f:a91b:7ebb:8f8f/64 scope link stable-privacy
    inet 172.18.0.2/16 brd 172.18.255.255 scope global eth0

ping 10.1.114.1
PING 10.1.114.1 (10.1.114.1) 56(84) bytes of data.
64 bytes from 10.1.114.1: icmp_seq=1 ttl=64 time=52.9 ms
64 bytes from 10.1.114.1: icmp_seq=2 ttl=64 time=0.915 ms

### on worker 3
r0ixqd7l4lroyd.runpod.internal
ssh r0ixqd7l4lroyd-64410e75@ssh.runpod.io -i /Users/maaruni/runpod_ed

ip addr | grep inet
    inet 127.0.0.1/8 scope host lo
    inet 10.0.119.84/10 scope global podnet1
    inet 192.168.96.2/20 brd 192.168.111.255 scope global eth0

    root@2893ea649881:/# ping 10.0.119.84
PING 10.0.119.84 (10.0.119.84) 56(84) bytes of data.
64 bytes from 10.0.119.84: icmp_seq=1 ttl=64 time=60.3 ms
64 bytes from 10.0.119.84: icmp_seq=2 ttl=64 time=0.404 ms
64 bytes from 10.0.119.84: icmp_seq=3 ttl=64 time=0.441 ms


### on worker node - spot:

ssh ucii5nzqvqwb2w-64410ca1@ssh.runpod.io -i /Users/maaruni/runpod_ed

root@ed954057479d:/# ip addr | grep inet
    inet 127.0.0.1/8 scope host lo
    inet6 ::1/128 scope host 
    inet 172.20.0.2/16 brd 172.20.255.255 scope global eth0



Step	Node	Action
1	Head Node	ray start --head --node-ip-address=10.0.119.84 --port=33 --include-dashboard=True \
--dashboard-host=0.0.0.0 --dashboard-port=8265
2	Worker Node	ray start --address=10.0.119.84:33
4	Worker Node	nc -zv 10.0.119.84 33 must return succeeded
    from worker 
    nc -zv 4pimh07ke0o6a3.runpod.internal 33
3	All Pods	Set MASTER_ADDR=10.0.119.84, MASTER_PORT=33, NCCL_SOCKET_IFNAME=podnet1


nohup python code4.py > run4.log 2>&1 &
tail -f run4.log
rm -rf /tmp/ray/*

watch -n 1 df -h | grep workspace


nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1 > gpu_log.csv
htop     # for CPU/mem per process
iotop    # for disk usage
iftop        # real-time interface bandwidth
nload        # simple bandwidth graphs
iperf3       # to measure latency and throughput


iperf3 -s  # on master
iperf3 -c <master-ip>  # on worker



## how to create virtual interface between 2 pods for effective nvcc communication

wget https://go.dev/dl/go1.20.12.linux-amd64.tar.gz && \
sudo tar -C /usr/local -xzf go1.20.12.linux-amd64.tar.gz && \
export PATH="/usr/local/go/bin:$PATH" && \
go version



tskey-auth-kcrrBiqXdv11CNTRL-73qj4U4cko8XFyv3NjXWo85KCwD9u2661