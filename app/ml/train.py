import os

from app.ml.model import DEFAULT_CHECKPOINT, train_model


def int_from_env(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def bool_from_env(name, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main():
    checkpoint_path = os.getenv("TRAIN_CHECKPOINT_PATH", DEFAULT_CHECKPOINT)
    games = int_from_env("TRAIN_GAMES", 96)
    simulations = int_from_env("TRAIN_SIMULATIONS", 200)
    epochs = int_from_env("TRAIN_EPOCHS", 5)
    self_play_batch_size = os.getenv("TRAIN_SELF_PLAY_BATCH_SIZE")
    if self_play_batch_size:
        self_play_batch_size = int(self_play_batch_size)
    else:
        self_play_batch_size = None

    train_model(
        games=games,
        simulations=simulations,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        force_retrain=bool_from_env("TRAIN_FORCE_RETRAIN"),
        resume_training=bool_from_env("TRAIN_RESUME"),
        self_play_batch_size=self_play_batch_size,
    )


if __name__ == "__main__":
    main()
