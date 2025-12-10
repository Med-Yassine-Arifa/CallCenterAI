import torch


def check_device():
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
    else:
        print("⚠️ CUDA not available. Using CPU.")


if __name__ == "__main__":
    check_device()
