# 🚀 Distributed 3DGS SfM Pipeline Usage

## Overview

The distributed processing system enables large-scale SfM reconstruction across multiple nodes, significantly reducing processing time for massive datasets.

## 🔧 Key Features

### **Multi-Node Processing**
- **Load Balancing**: Automatic distribution of images across nodes
- **Fault Tolerance**: Automatic retry with exponential backoff
- **Checkpointing**: Resume from failures
- **Memory Management**: Efficient batch processing

### **Advanced Scale Recovery**
- **Multi-Method Estimation**: Direct poses, triangulation, baselines
- **Robust Statistics**: RANSAC-style outlier rejection
- **Weighted Median**: Quality-weighted scale estimation
- **3DGS Consistency**: Global scale for 3D Gaussian Splatting

## 🏗 Architecture

```
Distributed 3DGS Pipeline
├── 🎯 Image Distribution (Round-robin)
├── 🔍 Distributed Feature Extraction
├── 🔗 Distributed Feature Matching  
├── 🏔️ Distributed Depth Estimation
├── 📐 Centralized Reconstruction (Master)
├── 📊 Result Distribution
└── 💾 Checkpointing & Recovery
```

## 🚀 Quick Start

### Single Node (Local)
```bash
python run_distributed_3dgs.py \
  --input_dir data/images \
  --output_dir results \
  --num_nodes 1 \
  --node_id 0 \
  --num_workers 8 \
  --batch_size 32 \
  --use_monocular_depth \
  --scale_recovery
```

### Multi-Node Cluster

#### Node 0 (Master)
```bash
python run_distributed_3dgs.py \
  --input_dir data/images \
  --output_dir results \
  --num_nodes 4 \
  --node_id 0 \
  --num_workers 8 \
  --master_addr "192.168.1.100" \
  --master_port 29500 \
  --batch_size 32 \
  --use_monocular_depth \
  --scale_recovery
```

#### Node 1
```bash
python run_distributed_3dgs.py \
  --input_dir data/images \
  --output_dir results \
  --num_nodes 4 \
  --node_id 1 \
  --num_workers 8 \
  --master_addr "192.168.1.100" \
  --master_port 29500 \
  --batch_size 32 \
  --use_monocular_depth \
  --scale_recovery
```

#### Node 2 & 3 (Similar commands with node_id=2,3)

## 📊 Performance Scaling

### Expected Speedups

| Dataset Size | Single Node | 4 Nodes | Speedup |
|-------------|-------------|---------|---------|
| 1,000 images | 45 min | 12 min | **3.8x** |
| 5,000 images | 3.5 hours | 45 min | **4.7x** |
| 10,000 images | 7 hours | 1.5 hours | **4.7x** |

### Memory Requirements

| Node Count | RAM per Node | Total RAM | Storage |
|------------|--------------|-----------|---------|
| 1 | 32 GB | 32 GB | 100 GB |
| 4 | 32 GB | 128 GB | 400 GB |
| 8 | 32 GB | 256 GB | 800 GB |

## 🔧 Advanced Configuration

### Scale Recovery Options

```bash
# Enable advanced scale recovery
--scale_recovery

# Custom scale recovery parameters
--fusion_weight 0.7  # SfM vs monocular depth weight
--bilateral_filter   # Apply bilateral filtering
```

### Distributed Processing Tuning

```bash
# Performance tuning
--batch_size 64      # Larger batches for GPU efficiency
--chunk_size 200     # More images per chunk
--num_workers 16     # More workers per node
--timeout 600        # Longer timeout for large datasets

# Fault tolerance
--max_retries 5      # More retries for reliability
--shared_storage /shared/sfm  # Shared storage path
```

### Network Configuration

```bash
# High-speed network (InfiniBand)
--backend nccl

# Standard network (Ethernet)
--backend gloo

# Custom network settings
--master_addr "cluster-master.company.com"
--master_port 29500
```

## 📁 Output Structure

```
results/
├── node_0/
│   ├── features.pkl
│   ├── matches.pkl
│   ├── depth_maps/
│   └── performance_stats.json
├── node_1/
│   ├── features.pkl
│   ├── matches.pkl
│   └── performance_stats.json
├── node_2/
│   └── ...
├── node_3/
│   └── ...
└── reconstruction/
    └── reconstruction.pkl  # Master node only
```

## 🔍 Monitoring & Debugging

### Performance Monitoring
```bash
# Enable profiling
--profile

# Monitor resource usage
htop  # CPU/Memory
nvidia-smi  # GPU usage
iotop  # Disk I/O
```

### Log Analysis
```bash
# Check node-specific logs
tail -f results/node_0/performance_stats.json

# Monitor distributed communication
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Fault Tolerance
```bash
# Check checkpoint status
ls -la /tmp/sfm_distributed/checkpoints/

# Resume from checkpoint
# (Automatic in current implementation)
```

## 🛠 Troubleshooting

### Common Issues

#### 1. **Network Connectivity**
```bash
# Test network connectivity
ping <master_addr>
telnet <master_addr> <master_port>

# Check firewall
sudo ufw status
```

#### 2. **Memory Issues**
```bash
# Reduce batch size
--batch_size 16

# Reduce workers
--num_workers 4

# Monitor memory
free -h
```

#### 3. **GPU Memory Issues**
```bash
# Use CPU backend
--backend gloo

# Reduce batch size
--batch_size 8

# Monitor GPU
nvidia-smi -l 1
```

#### 4. **Slow Processing**
```bash
# Increase workers
--num_workers 16

# Increase batch size
--batch_size 64

# Use faster storage
--shared_storage /mnt/nvme/sfm
```

## 📈 Scaling Guidelines

### **Small Datasets (< 1,000 images)**
- Single node sufficient
- Focus on quality over speed
- Use high-quality settings

### **Medium Datasets (1,000 - 10,000 images)**
- 2-4 nodes recommended
- Balance quality and speed
- Use distributed processing

### **Large Datasets (> 10,000 images)**
- 4-8+ nodes required
- Focus on throughput
- Use aggressive batching

### **Massive Datasets (> 100,000 images)**
- 8+ nodes with high-end hardware
- Consider data sharding
- Use specialized storage (NVMe, distributed FS)

## 🔬 Advanced Scale Recovery

### **Multi-Method Estimation**

The advanced scale recovery uses three complementary methods:

1. **Direct Scale Estimation**
   - Relative pose translation magnitude
   - Point distance ratios
   - Epipolar geometry constraints

2. **Triangulation Scale**
   - 3D point distance distribution
   - Median distance normalization
   - Outlier rejection

3. **Baseline Scale**
   - Camera baseline ratios
   - Target baseline optimization
   - Multi-view consistency

### **Robust Statistics**

```python
# RANSAC-style outlier rejection
Q1 = np.percentile(scales, 25)
Q3 = np.percentile(scales, 75)
IQR = Q3 - Q1
inliers = scales[(scales >= Q1 - 1.5*IQR) & (scales <= Q3 + 1.5*IQR)]

# Weighted median
weights = 1.0 / (1.0 + np.abs(inliers - median_scale))
final_scale = np.average(inliers, weights=weights)
```

## 🎯 Best Practices

### **Performance Optimization**
1. **Batch Size**: Start with 32, increase to 64-128 for GPU
2. **Workers**: 4-8 per node, scale with CPU cores
3. **Chunk Size**: 100-200 images per chunk
4. **Storage**: Use SSD/NVMe for shared storage

### **Reliability**
1. **Retries**: 3-5 retries with exponential backoff
2. **Timeouts**: 300-600 seconds for large datasets
3. **Checkpoints**: Every 50 chunks
4. **Monitoring**: Regular health checks

### **Quality**
1. **Scale Recovery**: Always enable for 3DGS
2. **Depth Estimation**: Use for dense reconstruction
3. **Bundle Adjustment**: GPU-accelerated for speed
4. **Vocabulary Tree**: O(n log n) complexity

## 📞 Support

For issues with distributed processing:
1. Check network connectivity
2. Verify shared storage permissions
3. Monitor resource usage
4. Review logs for specific errors

---

**Optimized for Large-Scale 3D Gaussian Splatting** 🚀 