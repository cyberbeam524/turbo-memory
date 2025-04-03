import evaluate
import numpy as np

# Minimal Example adapted from https://huggingface.co/docs/transformers/training
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ray.train import ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer

import os

# [1] Define a training function that includes all your training logic
# ====================================================================
def train_func(config):
    # Datasets
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_ds = dataset.map(tokenize_function, batched=True)

    small_train_ds = tokenized_ds["train"].shuffle(seed=42).select(range(1000))
    small_eval_ds = tokenized_ds["test"].shuffle(seed=42).select(range(1000))

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )

    # Evaluation metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch", report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_ds,
        eval_dataset=small_eval_ds,
        compute_metrics=compute_metrics,
    )

    # [2] Report metrics and checkpoints to Ray Train
    # ===============================================
    trainer.add_callback(RayTrainReportCallback())

    # [3] Prepare your trainer for Ray Data integration
    # =================================================
    trainer = prepare_trainer(trainer)

    # Start Training
    trainer.train()

from ray.train import RunConfig, FailureConfig
from ray.train.torch import TorchTrainer, TorchConfig

# [4] Build a Ray TorchTrainer to launch `train_func` on all workers
# ==================================================================
trainer = TorchTrainer(
    train_func, scaling_config=ScalingConfig(num_workers=3, use_gpu=True
    ),
    run_config=RunConfig(
                    name="my_run",
                    failure_config=FailureConfig(max_failures=3)  # or -1 for infinite retries
                    ),
    torch_config=TorchConfig(
        # nccl, gloo
        backend="nccl",          # or "nccl" for GPU but gloo is simpler and easier to debug
        init_method="env",    # tells it to use your env vars
        timeout_s= 1800
    )
)

print("Using MASTER_ADDR:", os.environ.get("MASTER_ADDR"))
print("Using MASTER_PORT:", os.environ.get("MASTER_PORT"))
results = trainer.fit()

# https://docs.ray.io/en/latest/train/examples/transformers/transformers_torch_trainer_basic.html#transformers-torch-trainer-basic-example