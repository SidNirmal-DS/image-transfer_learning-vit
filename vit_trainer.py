from datasets import load_dataset
import matplotlib.pyplot as plt

# Load 1% of the CIFAR-10 dataset for quick training
train_dataset = load_dataset("cifar10", split="train[:1%]")
test_dataset = load_dataset("cifar10", split="test[:1%]")

# View dataset info
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

# Preview one sample image and its label
# example = train_dataset[0]
# plt.imshow(example['img'])
# plt.title(f"Label: {example['label']}")
# plt.axis('off')
# plt.show()

from transformers import ViTImageProcessor

# Load the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Preprocessing function
def transform(example_batch):
    # Convert to pixel tensors as expected by ViT
    images = example_batch["img"]
    pixel_values = feature_extractor(images=images, return_tensors="pt")["pixel_values"]
    example_batch["pixel_values"] = pixel_values
    return example_batch

 # Apply transform to training and test datasets
train_dataset = train_dataset.map(transform, batched=True, batch_size=32).with_format("torch")
test_dataset = test_dataset.map(transform, batched=True, batch_size=32).with_format("torch")   

from transformers import ViTImageProcessor

# Load the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Preprocessing function
def transform(example_batch):
    images = example_batch["img"]
    pixel_values = feature_extractor(images=images, return_tensors="pt")["pixel_values"]
    example_batch["pixel_values"] = pixel_values
    return example_batch

# Apply transform to training and test datasets
train_dataset = train_dataset.map(transform, batched=True, batch_size=32).with_format("torch")
test_dataset = test_dataset.map(transform, batched=True, batch_size=32).with_format("torch")

from transformers import ViTForImageClassification

# Load ViT model with a new classification head (10 classes for CIFAR-10)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10,  # CIFAR-10 has 10 classes
    ignore_mismatched_sizes=True  # Replace the final layer with new one
)

# Optional: Map label indices to human-readable labels
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
model.config.id2label = dict(enumerate(LABELS))

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# Define how we train
training_args = TrainingArguments(
    output_dir='./results',               # Where to save model outputs
    num_train_epochs=5,                   # Number of times to see full dataset
    per_device_train_batch_size=4,        # Small batch size to avoid memory overload
    evaluation_strategy='epoch',          # Evaluate after each epoch
    save_strategy='epoch',                # Save model after each epoch
    load_best_model_at_end=True,          # Restore the best performing model
    logging_dir='./logs',                 # For TensorBoard logs
    logging_steps=10
)

# Function to compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Now define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Optional: Evaluate the model after training
trainer.evaluate()

# Save the fine-tuned model
trainer.save_model("vit-cifar10-checkpoint")