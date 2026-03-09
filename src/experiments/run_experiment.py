import argparse
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Run NMT Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.prepare_data()
    trainer.initialize_model()
    
    print("Starting experiment training...")
    trainer.train()
    print("Experiment completed.")

if __name__ == "__main__":
    main()
