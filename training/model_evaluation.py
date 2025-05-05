import os
import sys
import logging
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the parent directory to the path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import utility functions
from training.utils import setup_logging, create_directory

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"<kullanılan model> {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None

def load_speaker_mapping(mapping_path: str) -> Dict[int, str]:
    try:
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        logging.info(f"seslendirici eşleme yüklendi {mapping_path}")
        return mapping
    except Exception as e:
        logging.error(f"seslendirici eşleme yüklerken hata: {mapping_path}: {e}")
        return {}

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_dir: str
) -> None:
    create_directory(output_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('karmaşıklık Matrisi (takip)', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.tight_layout()
    plt.ylabel('gerçek label değerler', fontsize=14)
    plt.xlabel('Tahmini değerler', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Also save a normalized version
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('düzeltilmiş karmaşıklık Matrisi (takip)', fontsize=16)
    plt.colorbar()
    plt.xticks(tick_marks, labels, rotation=45, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black",
                    fontsize=12)
    
    plt.tight_layout()
    plt.ylabel('gerçek label değerler', fontsize=14)
    plt.xlabel('tahmin edilen değerler', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300)
    plt.close()
    
    logging.info(f"karmaşıklık matrisleri kayd: {output_dir}")

def plot_class_metrics(
    classification_report_dict: Dict[str, Dict[str, float]],
    output_dir: str
) -> None:
    """
    Plot and save class-specific metrics.
    
    Args:
        classification_report_dict: Classification report as dictionary
        output_dir: Directory to save the plots
    """
    create_directory(output_dir)
    
    # Extract class metrics (excluding averages)
    classes = []
    precision = []
    recall = []
    f1_scores = []
    
    for label, metrics in classification_report_dict.items():
        if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(label)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])
    
    # Plot class-specific metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Kesinlik')
    plt.bar(x, recall, width, label='yaklaşık doğruluk')
    plt.bar(x + width, f1_scores, width, label='F1-skoru')
    
    plt.xlabel('ses', fontsize=14)
    plt.ylabel('skore', fontsize=14)
    plt.title('ses kalitesi', fontsize=16)
    plt.xticks(x, classes, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(precision):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(recall):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(f1_scores):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'tanimlama_matris.png'), dpi=300)
    plt.close()
    
    logging.info(f"sınıflandırma matrisi resim yolu: {output_dir}")

def plot_feature_importance(
    model,
    output_dir: str
) -> None:
    create_directory(output_dir)
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('MODEL <--- Özellikleri', fontsize=16)
            plt.bar(range(len(importances)), importances[indices], align='center')
            
            """
            # mean=veri ortalaması, öğrnek: sakin ve net sesler için uyugun
            # std=standart sapma, sesin ne kadar değiştiğini gösterir, öğrnek: kadın,erkek,çocuk,yaşlı gibi belirgin signallar için uygundur
            # max=en yüksek değer, en yüksek enerji seviyesi, öğrnek: bağrma ani patlama, darbeli signaller için uygundur
            # min=en düşük değer, en düşük enerji seviyesi, öğrnek: temiz veri çakilmiş, sakin sesler için uygundur
            # energy=enerji seviyesi, sesin gücünü gösterir, öğrnek: RMS derlerin farklılığı
            # ZCR=sıfır geçiş oranı, sesin sıfır geçiş sayısını gösterir, öğrnek: kalın,ince sesler için uygundur
            # Segment Energy=segment enerji seviyesi, öğrnek: segment enerjilerinin farklılığı, farkılı diller konuşmuş sesler için uyugundur
            """

            feature_names = ['Mean', 'Std', 'Max', 'Min', 'Energy', 'ZCR']
            for i in range(10):
                feature_names.append(f'{i+1} enerji')
            
            # DEBUG:tekrarlayan özellikleri kaldır
            n_features = min(len(importances), len(feature_names))
            plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90, fontsize=12)
            
            plt.xlabel('yöntemler', fontsize=14)
            plt.ylabel('özellik değiri', fontsize=14)
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
            plt.close()
            
            logging.info(f"özellik grafiği kayd: {output_dir}")
    except Exception as e:
        logging.warning(f"özellik grafiği oluşturulamadı: {e}")

def plot_pca_visualization(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    speaker_mapping: Dict[int, str],
    output_dir: str
) -> None:
    create_directory(output_dir)
    
    try:
        from sklearn.decomposition import PCA
        
        # ------ PCA ------ 
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        
        plt.figure(figsize=(15, 10))
        
        # doğru sınıflandırma
        mask_correct = y_test == y_pred
        classes = np.unique(y_test)
        
        plt.subplot(1, 2, 1)
        for cls in classes:
            mask = (y_test == cls) & mask_correct
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=f"{speaker_mapping.get(cls, f'Ses {cls}')} (doğrulu)",
                       alpha=0.7)
        
        plt.title('Doğru sınıflandırma detayı', fontsize=16)
        plt.xlabel(f'PC1 eksende yonluk ({pca.explained_variance_ratio_[0]:.2%})', fontsize=14)
        plt.ylabel(f'PC2 eksende yonluk ({pca.explained_variance_ratio_[1]:.2%})', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        # yanlış sınıflandırma
        plt.subplot(1, 2, 2)
        for cls in classes:
            mask = (y_test == cls) & ~mask_correct
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       marker='x', s=100,
                       label=f"{speaker_mapping.get(cls, f'Ses {cls}')} (yanlış sınıflandırma)",
                       alpha=0.7)
        
        plt.title('yanlış sınıflandırma detayı', fontsize=16)
        plt.xlabel(f'PC1 eksende yonluk ({pca.explained_variance_ratio_[0]:.2%})', fontsize=14)
        plt.ylabel(f'PC2 eksende yonluk ({pca.explained_variance_ratio_[1]:.2%})', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300)
        plt.close()
        
        logging.info(f"PCA görselleştirme kaydı: {output_dir}")
    except Exception as e:
        logging.warning(f"PCA görselleştirme oluşturulamadı: {e}")

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    speaker_mapping: Dict[int, str],
    output_dir: Optional[str] = None
) -> Dict[str, Any]: 
    # ------ Tahminler ------ 
    y_pred = model.predict(X_test)
    
    # ------ Metrikler ------ 
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"Model doğruluğu: {accuracy:.4f}")
    logging.info(f"Makro F1 skoru: {macro_f1:.4f}")
    
    label_names = [speaker_mapping.get(i, f"Ses {i}") for i in range(len(speaker_mapping))]
        
    # ------ Sınıflandırma Raporu ------ 
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    logging.info("Sınıflandırma Raporu:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logging.info(f"  {label}: F1-skoru={metrics['f1-score']:.4f}, Kesinlik={metrics['precision']:.4f}, yaklaşık doğruluk={metrics['recall']:.4f}")
    
    if output_dir:
        create_directory(output_dir)
        
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(classification_report(y_test, y_pred, target_names=label_names))
        
        metrics = {
            'doğruluk': accuracy,
            'Makro F1 skoru': macro_f1,
            'rapor': report
        }
        
        with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        
        plot_confusion_matrix(y_test, y_pred, label_names, output_dir)
        plot_class_metrics(report, output_dir)
        plot_feature_importance(model, output_dir)
        plot_pca_visualization(X_test, y_test, y_pred, speaker_mapping, output_dir)
        
        logging.info(f"değerlendirme sonuçları ve görselleştirmeler kaydı: {output_dir}")
    
    return {
        'doğruluk': accuracy,
        'Makro F1 skoru': macro_f1,
        'rapor': report,
        'tahmin': y_pred
    }

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_dir", type=str, required=True, help="Model ve ses eşleme dosyası içeren dizin")
    parser.add_argument("--test_features", type=str, help="Test özellikleri (pickle dosyası)")
    parser.add_argument("--test_labels", type=str, help="Test etiketleri (pickle dosyası)")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Değerlendirme sonuçlarını kaydetmek için dizin")
    
    args = parser.parse_args()
    
    # ------ Loglama ------ 
    setup_logging()
    
    # ------ Model ve ses eşleme yükleme ------ 
    model_path = os.path.join(args.model_dir, 'model.pkl')
    mapping_path = os.path.join(args.model_dir, 'speaker_mapping.pkl')
    
    model = load_model(model_path)
    speaker_mapping = load_speaker_mapping(mapping_path)
    
    if model is None or not speaker_mapping:
        logging.error("Model veya ses eşleme yüklenemedi")
        return 1
    
    # ------ Test verileri yükleme ------ 
    try:
        with open(args.test_features, 'rb') as f:
            X_test = pickle.load(f)
        
        with open(args.test_labels, 'rb') as f:
            y_test = pickle.load(f)
        
        logging.info(f"Test verileri yüklendi: {len(X_test)} örnek")
    except Exception as e:
        logging.error(f"Test verileri yüklenemedi: {e}")
        return 1
    
    # ------ Model değerlendirme ------ 
    evaluate_model(model, X_test, y_test, speaker_mapping, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 