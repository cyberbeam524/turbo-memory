Ray AWS Cluster Setup & Commands

This guide provides step-by-step commands and tips to manage a Ray cluster on AWS using the Ray Autoscaler and Docker containers. This includes additonal debugging tips and understanding of how ray works under the hood on AWS by building on [Offical Ray AWS guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html)

🚀 Getting Started

✅ Launch the Cluster
Make sure your example-full.yaml is configured and run:
```
ray up ./example-full.yaml
```

💠 Useful Ray Commands

🔴 Terminate the Cluster
```
ray down ./example-full.yaml
```

🌐 Get the Cluster Head IP
```
ray get-head-ip ./example-full.yaml
```

📊 Port-Forward Ray Dashboard - default port: 8265
```
ray dashboard ./example-full.yaml
```
Then open: http://localhost:8265

📦 Submit a Job to the Cluster

In a new terminal:

ray job submit --address http://localhost:8265 --working-dir . -- python my_script.py

🔧 SSH into the Head Node
```
ray attach ./example-full.yaml
```
However, this is limitted to head node.

### Manual SSH setup
To access any node from local computer, add keys manually:

For this to work, we would need to create a keypair from AWS so that AWS has the public key and we can download the private key with .pem extension and place it in a local path such as ./local/rayscaler.pem.
![Alt text](img/createkeypair.jpg)

<img src="img/keypairtoggle.jpg" width="33%">

The example-full.yaml file should have reference. So, add reference to the .pem file in example-full.yaml and the name of the keypair (e.g. "Rayscaler") to both head worker node config:

<img src="img/sshprivatekeypath.jpg" width="33%">

<img src="img/headnodeconfig.jpg" width="33%">

<img src="img/workernodeconfig.jpg" width="33%">

After using ```ray up ray up example-full.yaml```, you should see all nodes referencing keypair (e.g. "Rayscaler").
<img src="img/keynameref.jpg" width="33%">

Now you can ssh into any node based on their public address given here on the ec2 info page:
![Alt text](img/sshpublicaddress.png)

SSH into the Worker or Head Node
```
ssh -i "./local/rayscaler.pem" ubuntu@<publicaddress>
```

🐳 Docker Container Debugging

🔍 Inspect Container Status
```
docker ps -a               # Is the container running?
docker logs ray_container  # Any startup logs/errors?
```

📦 Start Ray Docker Container (GPU)
```
nohup docker run -d --gpus all --name ray_container rayproject/ray-ml:latest-gpu sleep infinity >> dockerc.log 2>&1 &
```


Since ray is running **within** containers named ray_container, we have to enter the containers before checking for ray status which indicates which nodes are already added to the cluster and the total GPU and CPU power of this cluster:

👤 Enter the Running Container
```
docker exec -it ray_container bash
```



to submit a ray job, start dashboard on port 8265:
```
ray dashboard ./example-full.yaml
```

submit job via dashboard locally:
```
ray job submit --address http://localhost:8265 --working-dir . -- python trainingscripts/code1.py
```

to stop a ray job:
```
ray job stop <job_id> --address http://localhost:<dashboard-port>
```

📊 Monitor Autoscaling
```
ray exec ./example-full.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```


⚙️ Ray Runtime Control

When using 'rayproject/ray-ml:latest-gpu' image for containers, these commands are run within in the containers to start ray runtime on head and worker nodes respectively:
<img src="img/containercommands.jpg" width="70%">

## Running Ray without containers

Ensure that both nodes are pingable from each other. In order for reachability between nodes, they need to be located in the same subnet and security group in same region as stated in example-full.yaml file:
<img src="img/troubleshoot/sameregion.png" width="70%">

SSH into the Worker node:
```
ssh -i "./local/rayscaler.pem" ubuntu@<publicaddress>
```

Find private ip of head node:
![Alt text](img/troubleshoot/findprivateip.png)

Ping private ip of head from worker node:
```
ping <headprivateaddress>
```

You should see this:
<img src="img/troubleshoot/lowlatency.jpg" width="50%">
This indicates **low latency** and good connection between the nodes with around 0.257ms to transmit packets.

Example of higher latency(0.700ms to transmit packets) that will result in longer training time:

<img src="img/troubleshoot/highlatency.png" width="50%">

▶️ Start Ray Head Node
```
nohup ray start --head --port=6379 --object-manager-port=8076 \
  --autoscaling-config=~/ray_bootstrap_config.yaml \
  --dashboard-host=0.0.0.0 >> install.log 2>&1 &
```

▶️ Start Ray Worker Node
```
ray start --address=172.31.1.20:6379 --object-manager-port=8076
```


## Training Scripts

In order to distribute the data and training process to different gpus we can make use of Ray configuration objects like ScalingConfig, RunConfig, etc. By stating **workers = 3**, it triggers 3 worker nodes to be used with GPU used in all with **use_GPU = True**. 

Script modification:

<img src="img/training/scriptmodification.jpg" width="70%">


Initial logs of training will perform ping on each node to see if they are reachable from each other:
![Alt text](img/training/3nodesused.png)

On dashboard, toggle to job page of job being run and find actors being used:
![Alt text](img/training/3nodesactortable.jpg)

Training Completed!🤩

<img src="img/training/trainingcompleted.jpg" width="70%">
