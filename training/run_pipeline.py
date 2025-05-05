import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Optional, Union, Any

def setup_logging(log_file: Optional[str] = None) -> None:
    # note kayıtları
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

def run_command(cmd: List[str], description: Optional[str] = None) -> bool:
    if description:
        logging.info(description)
    
    logging.info(f"pipeline komut: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            logging.error(f"pipeline komut başarısız {process.returncode}")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"pipline komut hatası: {e}")
        return False

def run_training_pipeline(args: argparse.Namespace) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # çıktı dizinleri
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    features_dir = os.path.join(output_dir, "features")
    model_dir = os.path.join(output_dir, "model")
    evaluation_dir = os.path.join(output_dir, "evaluation")
    log_dir = os.path.join(output_dir, "logs")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # note kayıtları
    log_file = os.path.join(log_dir, "pipeline.log")
    setup_logging(log_file)
    
    logging.info(f"eğitim pipeline baş: {timestamp}")
    logging.info(f"veri dizini: {args.data_dir}")
    logging.info(f"çıktı dizini: {output_dir}")
    
    logging.info("============== Özellik çıkarım ==============")

    feature_extraction_cmd = [
        "python", "-m", "training.feature_extraction",
        "--data_dir", args.data_dir,
        "--output_dir", features_dir
    ]
    
    if not run_command(feature_extraction_cmd, "Özellik çıkarım başarısız. Pipeline durduruldu."):
        logging.error("Özellik çıkarım başarısız. Pipeline durduruldu.")
        return False

    logging.info("============== model eğitimi ==============")
    train_model_cmd = [
        "python", "-m", "training.train_model",
        "--data_dir", args.data_dir,
        "--output_dir", model_dir,
        "--seed", str(args.seed)
    ]
    
    if not run_command(train_model_cmd, "Model eğitimi başarısız. Pipeline durduruldu."):
        logging.error("Model eğitimi başarısız. Pipeline durduruldu.")
        return False
    

    if os.path.exists(os.path.join(features_dir, "features.pkl")) and os.path.exists(os.path.join(features_dir, "labels.pkl")):
        logging.info("============== model değerlendirme ==============")
        evaluate_cmd = [
            "python", "-m", "training.model_evaluation",
            "--model_dir", model_dir,
            "--test_features", os.path.join(features_dir, "features.pkl"),
            "--test_labels", os.path.join(features_dir, "labels.pkl"),
            "--output_dir", evaluation_dir
        ]
        
        if not run_command(evaluate_cmd, "Model değerlendirme başarısız. Pipeline durduruldu."):
            logging.error("Model değerlendirme başarısız. Pipeline durduruldu.")
            return False
    else:
        logging.warning("Test veri sorunu")
    
    logging.info("============== pipeline tamamlandı ==============")
    logging.info(f"model kaydedildi {model_dir}")
    
    if os.path.exists(evaluation_dir):
        logging.info(f"değerlendirme sonuçları kaydedildi {evaluation_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="training_outputs")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    success = run_training_pipeline(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 