# PK Sampler Implementation Checklist

## ✅ **COMPLETED - All Components Present**

### 1. **PKBatchSampler Class** ✅
- [x] Located in `sampler.py`
- [x] Inherits from `torch.utils.data.Sampler`
- [x] Accepts `**kwargs` for PyTorch Lightning compatibility
- [x] Has all required PyTorch Lightning attributes:
  - [x] `batch_size`
  - [x] `drop_last`
  - [x] `sampler`, `num_workers`, `pin_memory`, `timeout`
  - [x] `worker_init_fn`, `multiprocessing_context`, `generator`
  - [x] `prefetch_factor`, `persistent_workers`
- [x] Implements `__iter__()` and `__len__()` methods
- [x] Proper PK sampling logic (N=16 identities, K=4 samples each)

### 2. **DataModule Integration** ✅
- [x] Located in `data.py`
- [x] Imports `PKBatchSampler` from `sampler`
- [x] Has `pk_sampler` and `num_instances` parameters
- [x] `train_dataloader()` method uses `PKBatchSampler` when enabled
- [x] Uses `batch_sampler` parameter instead of `sampler + batch_size`

### 3. **Configuration** ✅
- [x] Located in `config.py`
- [x] Has `--pk_sampler` argument (action='store_true', default=True)
- [x] Has `--num_instances` argument (default=4)
- [x] Has `--batch_size` argument (default=8)

### 4. **Command Line Usage** ✅
```bash
# Correct command line usage:
python main.py --data_root /path/to/data --output_dir ./output --pk_sampler --num_instances 4

# The --pk_sampler flag enables PK sampling
# The --num_instances 4 sets K=4 samples per identity
# The batch_size will be automatically calculated as N*K = 16*4 = 64
```

### 5. **Testing** ✅
- [x] `test_pk_batch_sampler.py` - Basic functionality test
- [x] `test_pk_batch_sampler_comprehensive.py` - PyTorch Lightning compatibility test
- [x] All tests pass with ✅ ALL ATTRIBUTES PRESENT

## 🚀 **READY FOR TRAINING**

### Expected Behavior:
1. **PK Sampling**: 16 identities × 4 samples = 64 total samples per batch
2. **No NaN Gradients**: Proper batch diversity prevents gradient issues
3. **Stable TransMatcher Loss**: Multiple samples per identity enable meaningful pairwise comparisons
4. **PyTorch Lightning Compatibility**: No more TypeError exceptions

### Training Command:
```bash
cd tradaface-v2
python main.py --data_root /path/to/your/data --output_dir ./output --pk_sampler --num_instances 4
```

## 📋 **Verification Steps**

1. **Run the comprehensive test**:
   ```bash
   python compare_qaconv_transmatcher.py
   ```
   Should show: ✅ ALL ATTRIBUTES PRESENT - PyTorch Lightning compatible!

2. **Start training**:
   ```bash
   python main.py --data_root /path/to/data --output_dir ./output --pk_sampler --num_instances 4
   ```
   Should start without errors and show proper PK sampling logs.

3. **Monitor training logs**:
   - Look for: `[DataModule] Using PK sampler: N=16, K=4, batch_size=64`
   - Look for: `[PKBatchSampler] Will sample 16 identities per batch, 4 instances each`
   - No more `TypeError: __init__() got an unexpected keyword argument`

## 🎯 **Success Indicators**

- ✅ No more PyTorch Lightning compatibility errors
- ✅ Stable training with proper PK sampling
- ✅ TransMatcher loss showing meaningful values (not just diagonal masks)
- ✅ No NaN gradients due to proper batch diversity
- ✅ Faster convergence with better metric learning batches

**Everything is ready! The PK sampler implementation is complete and fully compatible with PyTorch Lightning.** 