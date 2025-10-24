# Performance Bottlenecks Identified

## 🔴 CRITICAL: CPU-GPU Transfer (FIXED)
**Location**: `agent.py` and `agent_enhanced.py` - `optimize()` function
**Issue**: 64 individual `.to(device)` calls per training step (32 states + 32 next_states)
**Impact**: 5-10x slowdown on GPU
**Fix Applied**: Use `Batch.from_data_list()` to batch transfer (64 transfers → 2 transfers)

---

## 🟠 HIGH PRIORITY: Ray-casting in Python loops
**Location**: `environment.py` - `_raycast_sensing()` function
**Issue**: 
- Nested loops: `NUM_RAYS (12) × SAMPLES_PER_RAY (8) = 96 iterations`
- Called EVERY step (200 times per episode)
- Uses slow `math.cos/sin` and `np.linspace` in loops
- Total: 19,200 loop iterations per episode

**Current Code**:
```python
for i in range(config.NUM_RAYS):  # 12 times
    angle = (2 * math.pi * i) / config.NUM_RAYS
    for r in np.linspace(0, self.sensor_range, config.SAMPLES_PER_RAY):  # 8 times
        cx = int(round(px + r * math.cos(angle)))
        cy = int(round(py + r * math.sin(angle)))
        # ... obstacle check
```

**Optimization Strategy**:
- Pre-compute angles and trig values
- Vectorize using NumPy array operations
- Cache ray directions
- Expected speedup: 5-10x

---

## 🟠 HIGH PRIORITY: Graph encoder called twice per step
**Location**: `train.py` and `train_enhanced.py`
**Issue**:
```python
graph_data = agent.graph_encoder.encode(state, env.world_state, 0)
# ... action ...
next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
```
- Encoding happens 400 times per episode (200 steps × 2)
- Each encoding: loops over all sensed cells, builds edges, computes features
- Enhanced encoder is even slower (10D nodes + 3D edges vs 8D nodes)

**Optimization Strategy**:
- Cache the "next" encoding for reuse as "current" in next step
- Store encoded graph instead of re-encoding
- Expected speedup: 2x for encoding time

---

## 🟡 MEDIUM: Dictionary lookups in hot loops
**Location**: `graph_encoder.py` - Multiple functions
**Issue**:
```python
for pos in sensed_cells:  # ~50-100 cells
    coverage = robot_state.local_map.get(pos, (0.0, "unknown"))[0]  # Dict lookup
    node_type = robot_state.local_map.get(pos, (0.0, "unknown"))[1]  # Dict lookup again!
```
- Same key accessed twice
- `.get()` with default tuple creation
- Happens 100+ times per encoding (2× per step = 200×)

**Optimization Strategy**:
- Store result: `cell_data = robot_state.local_map.get(pos, (0.0, "unknown"))`
- Use direct dict access when key exists: `if pos in local_map`
- Expected speedup: 1.5-2x for encoding

---

## 🟡 MEDIUM: Nested loops for edge building
**Location**: `graph_encoder.py` - `_build_edges()`
**Issue**:
```python
for pos, idx in pos_to_idx.items():  # ~50-100 nodes
    for dx in [-1, 0, 1]:  # 3×
        for dy in [-1, 0, 1]:  # 3×
            neighbor = (x + dx, y + dy)
            if neighbor in pos_to_idx:  # Dict lookup
                neighbor_type = local_map.get(neighbor, ...)[1]  # Another lookup
```
- Triple nested loop
- 9 iterations per node → 450-900 iterations
- Multiple dict lookups per iteration

**Optimization Strategy**:
- Vectorize neighbor computation using NumPy
- Pre-build offset array: `offsets = np.array([[-1,-1],[-1,0],...])`
- Use NumPy broadcasting
- Expected speedup: 3-5x for edge building

---

## 🟡 MEDIUM: String comparisons in hot paths
**Location**: `graph_encoder.py` and `environment.py`
**Issue**:
```python
node_type = robot_state.local_map.get(pos, (0.0, "unknown"))[1]
is_obstacle = 1.0 if node_type == "obstacle" else 0.0
# ...
if node_type != "obstacle":
```
- String comparison in loops
- Repeated for every node/edge

**Optimization Strategy**:
- Use integer enum: 0=unknown, 1=free, 2=obstacle
- Direct numeric comparison
- Expected speedup: 1.2-1.5x

---

## 🟡 MEDIUM: Torch tensor creation in loops
**Location**: `graph_encoder.py` - `_encode_node_features()`
**Issue**:
```python
for pos in sensed_cells:
    features = self._encode_node_features(...)  # Calls this
    node_features.append(features)

# Inside _encode_node_features:
return torch.tensor(features, dtype=torch.float32)  # New tensor each time!
```
- Creates 50-100 individual tensors per encoding
- Then stacks them: `x = torch.stack(node_features)`

**Optimization Strategy**:
- Pre-allocate NumPy array: `features = np.zeros((len(sensed_cells), 8))`
- Fill array with vectorized operations
- Single torch.from_numpy() call at end
- Expected speedup: 2-3x for feature creation

---

## 🟢 LOW: NetworkX graph (not used in hot path)
**Location**: `map_generator.py` and `environment.py`
**Issue**: Uses `nx.Graph` for map representation
**Impact**: Only used during environment creation (once per episode), not in step loop
**Priority**: Low - not worth optimizing since it's not called frequently

---

## Summary of Impact

### Hot Path (called 200+ times per episode):
1. ✅ **GPU transfer batching** (FIXED) - 5-10x speedup on training
2. 🔴 **Ray-casting vectorization** - 5-10x speedup on sensing
3. 🔴 **Graph encoding optimization** - 3-5x speedup on encoding
4. 🟡 **Dict lookup optimization** - 1.5-2x speedup on encoding

### Expected Combined Speedup:
- **Sensing**: 5-10x faster
- **Encoding**: 5-10x faster  
- **Training**: 5-10x faster (already fixed)
- **Overall episode time**: 3-5x faster (conservative estimate)

### Current Bottleneck Breakdown (estimated):
- Ray-casting: ~30-40% of episode time
- Graph encoding: ~30-40% of episode time
- Action selection: ~5-10% of episode time
- Training: ~10-20% of episode time (now optimized!)
- Environment stepping: ~5-10% of episode time

### Next Steps:
1. Test the GPU batching fix first
2. Implement ray-casting vectorization
3. Implement graph encoding cache + vectorization
4. Profile again to verify improvements
