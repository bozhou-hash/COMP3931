import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.q_network import QNetwork


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_dataset_path(filename):
    return os.path.join(get_project_root(), "datasets", filename)


def get_checkpoint_dir():
    checkpoint_dir = os.path.join(get_project_root(), "training", "imitation_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def load_dataset(filename):
    dataset_path = get_dataset_path(filename)
    data = np.load(dataset_path)

    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)

    return states, actions


def evaluate_model(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)

            logits = model(states)
            loss = criterion(logits, actions)

            total_loss += loss.item() * states.size(0)

            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == actions).sum().item()
            total_samples += states.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def train_dqn_imitation(
    dataset_filename="minimax_tactical_dataset_depth6.npz",
    batch_size=128,
    learning_rate=0.001,
    num_epochs=20,
    validation_split=0.2,
    save_filename="dqn_imitation_tactical_depth6_model_final.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states, actions = load_dataset(dataset_filename)
    dataset = TensorDataset(states, actions)

    validation_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = QNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    validation_losses = []
    validation_accuracies = []

    print(f"Training imitation model on {len(dataset)} samples")
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {validation_size}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            logits = model(batch_states)
            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch_states.size(0)
            total_train_samples += batch_states.size(0)

        average_train_loss = running_loss / total_train_samples
        validation_loss, validation_accuracy = evaluate_model(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        train_losses.append(average_train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {average_train_loss:.5f} | "
            f"Val Loss: {validation_loss:.5f} | "
            f"Val Acc: {validation_accuracy:.2%}"
        )

    checkpoint_dir = get_checkpoint_dir()

    final_model_path = os.path.join(checkpoint_dir, save_filename)
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal imitation model saved to: {final_model_path}")

    history_path = os.path.join(checkpoint_dir, "imitation_training_history.npz")
    np.savez(
        history_path,
        train_losses=np.array(train_losses, dtype=np.float32),
        validation_losses=np.array(validation_losses, dtype=np.float32),
        validation_accuracies=np.array(validation_accuracies, dtype=np.float32),
    )
    print(f"Training history saved to: {history_path}")

    return model, train_losses, validation_losses, validation_accuracies


def main():
    train_dqn_imitation(
        dataset_filename="minimax_tactical_dataset_depth6.npz",
        batch_size=128,
        learning_rate=0.001,
        num_epochs=20,
        validation_split=0.2,
        save_filename="dqn_imitation_tactical_depth6_model_final.pth",
    )


if __name__ == "__main__":
    main()