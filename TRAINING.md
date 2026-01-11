# Training Atlas From Scratch

This guide explains how to train your own Atlas model from scratch.

## Prerequisites

**Hardware:**
- NVIDIA GPU with 16GB+ VRAM (recommended: RTX 3090, 4090, or A100)
- 32GB+ System RAM
- 100GB+ free disk space

**Software:**
- Python 3.10+
- CUDA 11.8+ (for GPU support)

## Training Dependencies

Install training dependencies:
```bash
pip install torch torchvision  # For training only
```

## Training Process

### 1. Prepare Data
```bash
# Generate GRPO training data
python prepare_grpo_data.py

# This creates:
# - data/grpo_training.json (prompts with ranked responses)
# - data/validation.json (for validation)
```

### 2. Configure Training

Edit `train_config.json`:
```json
{
  "model_size": "1.1B",
  "learning_rate": 5e-6,
  "batch_size": 4,
  "epochs": 10,
  "grpo_preference_weight": 0.5
}
```

### 3. Start Training
```bash
# Start training (will take 8-12 hours)
python train_atlas.py --config train_config.json

# Monitor progress
tail -f logs/training.log
```

### 4. Save Model

Trained model will be automatically saved to:
```
models/atlas_1.1B/
├── model_weights.pkl
├── config.json
├── vocabulary.json
└── training_stats.json
```


Total parameters: 1,102,345,728

## GRPO Training Algorithm

Atlas implements Group Relative Policy Optimization:
```python
def grpo_loss(responses, rankings):
    # Compute log probabilities
    log_probs = model.compute_log_probs(responses)
    
    # Preference-weighted loss
    preferences = softmax(rankings * temperature)
    preference_loss = -(log_probs * preferences).sum()
    
    # Pairwise ranking loss
    pairwise_loss = compute_pairwise_ranking(log_probs, rankings)
    
    return preference_loss + 0.5 * pairwise_loss
```

## Training Tips

1. **Start small:** Test with 1 epoch on small dataset first
2. **Monitor loss:** Should decrease smoothly (target: <1.5)
3. **Save checkpoints:** Every 1000 steps
4. **Use mixed precision:** Saves memory
5. **Gradient clipping:** Prevents exploding gradients

## Hardware Requirements

| Model Size | GPU VRAM | RAM | Training Time |
|------------|----------|-----|---------------|
| 1.1B params | 16GB | 32GB | 9-12 hours |

## Troubleshooting

**Out of Memory:**
- Reduce batch size to 1
- Enable gradient checkpointing
- Use smaller model size

**Loss not decreasing:**
- Check data quality
- Adjust learning rate
- Verify GRPO implementation

For questions, open an issue on GitHub.
