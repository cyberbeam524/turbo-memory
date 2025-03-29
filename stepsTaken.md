https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html


https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/_private/aws/config.py

ray up example-full.yaml


Local node IP: 172.31.13.196

--------------------
Ray runtime started.
--------------------

Next steps
  To add another node to this Ray cluster, run
    ray start --address='172.31.13.196:6379'
  
  To connect to this Ray cluster:
    import ray
    ray.init()
  
  To submit a Ray job using the Ray Jobs CLI:
    RAY_ADDRESS='http://172.31.13.196:8265' ray job submit --working-dir . -- python my_script.py
  
  See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html 
  for more information on submitting Ray jobs to the Ray cluster.
  
  To terminate the Ray runtime, run
    ray stop
  
  To view the status of the cluster, use
    ray status
  
  To monitor and debug Ray, view the dashboard at 
    172.31.13.196:8265
  
  If connection to the dashboard fails, check your firewall settings and network configuration.
Shared connection to 18.138.241.17 closed.
  New status: up-to-date

Useful commands:
  To terminate the cluster:
    ray down /Users/maaruni/Documents/GitHub/raytutorials/example-full.yaml
  
  To retrieve the IP address of the cluster head:
    ray get-head-ip /Users/maaruni/Documents/GitHub/raytutorials/example-full.yaml
  
  To port-forward the cluster's Ray Dashboard to the local machine:
    ray dashboard /Users/maaruni/Documents/GitHub/raytutorials/example-full.yaml
  
  To submit a job to the cluster, port-forward the Ray Dashboard in another terminal and run:
    ray job submit --address http://localhost:<dashboard-port> --working-dir . -- python my_script.py
  
  To connect to a terminal on the cluster head for debugging:
    ray attach /Users/maaruni/Documents/GitHub/raytutorials/example-full.yaml
  
  To monitor autoscaling:
    ray exec /Users/maaruni/Documents/GitHub/raytutorials/example-full.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'



  







ray job submit --address http://localhost:8265 --working-dir . -- python code1.py


    Amazon EC2 P5e Instances have up to 8 NVIDIA Tesla H200 GPUs.

Amazon EC2 P5 Instances have up to 8 NVIDIA Tesla H100 GPUs.

Amazon EC2 P4 Instances have up to 8 NVIDIA Tesla A100 GPUs.

Amazon EC2 P3 Instances have up to 8 NVIDIA Tesla V100 GPUs.

Amazon EC2 G3 Instances have up to 4 NVIDIA Tesla M60 GPUs.

Amazon EC2 G4 Instances have up to 4 NVIDIA T4 GPUs.

Amazon EC2 G5 Instances have up to 8 NVIDIA A10G GPUs.

Amazon EC2 G6 Instances have up to 8 NVIDIA L4 GPUs.

Amazon EC2 G6e Instances have up to 8 NVIDIA L40S Tensor Core GPUs.

Amazon EC2 G5g Instances have Arm64-based AWS Graviton2 processors.




aws service-quotas get-service-quota --service-code ec2 --quota-code L-34B43A08 --region ap-southeast-1

ray up ray_example.yaml --restart-only

https://aws.amazon.com/ec2/instance-types/g5g/


aws ec2 describe-instance-types --region ap-southeast-1 --filters "Name=processor-info.supported-features,Values=gpu"


aws ec2 describe-instance-types --region ap-southeast-1 --query "InstanceTypes[?GpuInfo.Gpus[0].Manufacturer!=null].[InstanceType,GpuInfo.Gpus[0].Name,GpuInfo.Gpus[0].Count]" --output table


--------------------------------
|     DescribeInstanceTypes     |
+----------------+--------+-----+
|  g4dn.metal    |  T4    |  8  |
|  g4dn.12xlarge |  T4    |  4  |
|  p2.8xlarge    |  K80   |  8  |
|  g5g.2xlarge   |  T4g   |  1  |
|  p3.2xlarge    |  V100  |  1  |
|  g5g.16xlarge  |  T4g   |  2  |
|  g5g.8xlarge   |  T4g   |  1  |
|  g5g.metal     |  T4g   |  2  |
|  g4dn.xlarge   |  T4    |  1  |
|  p2.xlarge     |  K80   |  1  |
|  g4dn.16xlarge |  T4    |  1  |
|  p2.16xlarge   |  K80   |  16 |
|  p3.16xlarge   |  V100  |  8  |
|  p3.8xlarge    |  V100  |  4  |
|  g5g.xlarge    |  T4g   |  1  |
|  g4dn.4xlarge  |  T4    |  1  |
|  g4dn.2xlarge  |  T4    |  1  |
|  g4dn.8xlarge  |  T4    |  1  |
|  g5g.4xlarge   |  T4g   |  1  |
+----------------+--------+-----+
~












## RUNPOD

ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`


pip install ray transformers datasets evaluate ray[train] pandas ray[default] scikit-learn transformers[torch] 'accelerate>=0.26.0' 
apt update && apt install lsof -y
apt update && apt install vim -y
apt update && apt install iputils-ping y
apt update && apt install netcat -y
apt update && apt install redis-server -y


sudo apt update && sudo apt install iputils-ping -y



54535
redis-server --port 54535

RAY_REDIS_ADDRESS=0.0.0.0:54535 ray start --include-dashboard=True --head \
--port=33 --dashboard-host=0.0.0.0 --dashboard-port=8265 

–address=0.0.0.0:54535

RAY_REDIS_ADDRESS=opm3gaj0j4b50d.runpod.internal:54535 ray start --address='opm3gaj0j4b50d.runpod.internal:33'


ray start --head \
  --port=33 \
  --node-ip-address=opm3gaj0j4b50d-64410eb0 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8888



  lsof -i :54535


/Users/maaruni/runpod_ed

ssh opm3gaj0j4b50d-64410eb0@ssh.runpod.io -i /Users/maaruni/runpod_ed

opm3gaj0j4b50d.runpod.internal



/Users/maaruni/runpod_ed.pub

scp -i /Users/maaruni/runpod_ed -r -P 30203 ./code1.py opm3gaj0j4b50d-64410eb0@ssh.runpod.io:/

export RAY_record_ref_creation_sites=1
nohup python code1.py > run.log 2>&1 &

tail -f run.log

RAY_ADDRESS='http://172.21.0.2:8265' ray job submit --working-dir . -- python code1.py > run.log 2>&1 &

nohup ray job submit --working-dir . -- python code1.py > run.log 2>&1 &





[1] 9359



pkill -f code1.py


ps aux | grep code1.py
kill <PID>



:1,$d


https://docs.runpod.io/pods/configuration/expose-ports
opm3gaj0j4b50d.runpod.internal
https://{POD_ID}-{INTERNAL_PORT}.proxy.runpod.net


https://opm3gaj0j4b50d-8265.proxy.runpod.net


worker
 ping 6gzcft8b6pwef9.runpod.internal
nc -zv 6gzcft8b6pwef9.runpod.internal 33

head
 ping opm3gaj0j4b50d.runpod.internal
nc -zv opm3gaj0j4b50d.runpod.internal 33




ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: podnet1: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1300 qdisc tbf state UNKNOWN group default qlen 500
    link/none 
    inet 10.0.72.204/10 scope global podnet1
       valid_lft forever preferred_lft forever
    inet6 fe80::9f97:a1e3:20d4:51fc/64 scope link stable-privacy 
       valid_lft forever preferred_lft forever
6783: eth0@if6784: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UP group default 
    link/ether 02:42:ac:15:00:02 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 172.21.0.2/16 brd 172.21.255.255 scope global eth0
       valid_lft forever preferred_lft forever




       echo "MASTER_ADDR=$MASTER_ADDR"
       echo "MASTER_PORT=$MASTER_PORT"
       echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

       export MASTER_ADDR=opm3gaj0j4b50d.runpod.internal
       export MASTER_PORT=33
       export NCCL_SOCKET_IFNAME=podnet1





Step	Node	Action
1	Head Node	ray start --head --node-ip-address=10.0.72.204 --port=6379
2	Worker Node	ray start --address=10.0.72.204:6379
3	All Pods	Set MASTER_ADDR=10.0.72.204, MASTER_PORT=29500, NCCL_SOCKET_IFNAME=podnet1
4	Worker Node	nc -zv 10.0.72.204 29500 must return succeeded





got this on worker node:

root@b3fa0d4fc603:/# ip addr | grep inet
    inet 127.0.0.1/8 scope host lo
    inet6 ::1/128 scope host
    inet 10.0.30.226/10 scope global podnet1
    inet6 fe80::54ee:2ade:be45:5277/64 scope link stable-privacy
    inet 172.22.0.2/16 brd 172.22.255.255 scope global eth0

on head node:

root@98eacdea7171:/# ip addr | grep inet
    inet 127.0.0.1/8 scope host lo
    inet6 ::1/128 scope host 
    inet 10.0.72.204/10 scope global podnet1
    inet6 fe80::9f97:a1e3:20d4:51fc/64 scope link stable-privacy 
    inet 172.21.0.2/16 brd 172.21.255.255 scope global eth0


ChatGPT said:
Memory updated
Thanks for sharing the network interfaces. Based on what you've posted, your Ray head node is at 172.21.0.2, but your worker is on a different subnet: 172.22.0.2, which likely explains why NCCL (or PyTorch DDP) cannot connect to the eth0 interface of the head node—they're on different isolated internal networks.

✅ Here's how to fix it:
Use podnet1 as the NCCL communication interface across both pods since both are on the 10.x.x.x range:

1. Confirm podnet1 is usable on both nodes
Head node IP on podnet1: 10.0.72.204

Worker node IP on podnet1: 10.0.30.226

These are reachable! ✅





ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`







# debugging

. GPU Not Fully Utilized / CPU Bottleneck
If one worker is slower (e.g., due to CPU contention or a shared GPU), DDP will wait for the slowest. In your log:

bash
Copy
Edit
238.06s/it vs 37.88s/it
This is classic straggler behavior. One worker is much slower than the other — could be due to:

GPU sharing on one pod

Different GPU types or availability

CPU preprocessing not distributed correctly (e.g., not setting num_proc for .map())



watch -n 1 nvidia-smi
htop




https://medium.com/@ed.sealing/multi-node-gh200-nccl-testing-dc2fc64d97a0

https://aws.amazon.com/blogs/hpc/scaling-your-llm-inference-workloads-multi-node-deployment-with-tensorrt-llm-and-triton-on-amazon-eks/

https://github.com/aws-samples/awsome-inference/tree/main

https://greennode.ai/blog/distributed-training-for-llm

https://huggingface.co/docs/optimum/en/habana/usage_guides/multi_node_training



aws



ray start --address='18.138.228.31:6379'

docker run -it --rm --net=host --name ray_container \
  --ulimit nofile=65536:65536 \
  rayproject/ray-ml:latest-gpu