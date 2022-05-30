from transformers import AutoFeatureExtractor, ViTModel
from transformers import Trainer, TrainingArguments
from torchvision import models
import torch
from datasets.custom_dataloader import DinoDataloader



def fine_tune_transformer(args):

    data_path = args.data_path
    model = args.model
    n_classes = args.n_classes

    if model == 'dino':        
        feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vitb16')
        model_ft = ViTModel.from_pretrained('facebook/dino-vitb16', num_labels=n_classes)

    training_args = TrainingArguments(
        output_dir = f'logs/fine_tune/{model}',
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    dataset = DinoDataloader(args.data_path, mode='train') #, transforms=transform

    trainer = Trainer(
        model = model_ft,
        args=training_args, 

    )