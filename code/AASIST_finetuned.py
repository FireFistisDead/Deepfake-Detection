import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from pathlib import Path
import soundfile as sf

from data_utils import Dataset_ASVspoof2019_train
from models.AASIST import Model
from utils import set_seed, create_optimizer
import json
from tqdm import tqdm

def main():
    # ========== Load Config ==========
    with open("aasist/config/AASIST_2.conf", "r") as f:
        config = json.load(f)

    # ========== Set seed and device ==========
    set_seed(42, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== Prepare Paths ==========
    base_path = Path(config["database_path"])
    protocol_file = base_path / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = base_path / "ASVspoof2019_LA_train"

    # ========== Parse Protocol ==========
    list_IDs = []
    labels = {}
    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[1]
            label_str = parts[-1]
            list_IDs.append(utt_id)
            labels[utt_id] = 1 if label_str == "bonafide" else 0

    # ========== Dataset and Dataloader ==========
    train_dataset = Dataset_ASVspoof2019_train(list_IDs[:1000], labels, audio_dir)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    # ========== Initialize Model ==========
    model = Model(config["model_config"])
    model.load_state_dict(torch.load(config["model_path"]))
    model.to(device)

    # ========== Loss Function ==========
    criterion = nn.CrossEntropyLoss()

    # ========== Optimizer and Scheduler ==========
    config["optim_config"]["epochs"] = config["num_epochs"]
    config["optim_config"]["steps_per_epoch"] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), config["optim_config"])

    # ========== Training Loop ==========
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[0], targets)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs[0], 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            progress.set_postfix(loss=loss.item(), acc=100. * correct / total)

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # ========== Save Fine-Tuned Model ==========
    save_path = r"C:\Users\yash khare\Desktop\DEEPFAKE-DEBUNK\aasist\models\aasist_finetuned2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Fine-tuned model saved to {save_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
