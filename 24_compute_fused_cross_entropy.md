# Fused Cross Entropy

## Context

Standard cross entropy materializes full logits tensor (vocab_size × batch_size × hidden_size). For large vocabs (50k+), this is huge memory.

**Solution:** Fuse final linear projection with cross entropy, avoiding full logits materialization.

## Implementation

```python
class FusedCrossEntropy:
    """Fused final linear + cross entropy"""
    
    def forward(self, hidden_states, weight, labels):
        """
        Compute cross entropy without materializing full logits.
        Only compute logits for labels (sparse computation).
        """
        # Fused kernel: linear + loss in one pass
        loss = fused_linear_cross_entropy_cuda(
            hidden_states, weight, labels
        )
        return loss  # Never created vocab_size × batch logits!
```

## Code Location

- **Implementation:** `megatron/core/fusions/fused_cross_entropy.py`

## Performance Impact

### Memory Savings

For vocab=50k, batch=1M tokens, hidden=4096:
- Full logits: 50k × 1M × 2 bytes = **100 GB**
- Fused: No logits materialization = **0 GB**
- **Saved: 100 GB!**

### Speedup

- **1.5-2x faster** loss computation
- Enables larger batch sizes

## When to Use

**Use when:**
- Language modeling (loss calculation)
- Large vocabulary (> 10k)
- Memory-constrained training

## Related Optimizations

- [FP8 Training](28_compute_fp8_training.md) - Can also apply to loss computation
- [Gradient Accumulation Fusion](26_compute_grad_accumulation_fusion.md) - Fusion principle

## References

- [Efficient Cross Entropy](https://arxiv.org/abs/2006.16362)
- [Label Smoothing](https://arxiv.org/abs/1512.00567)

