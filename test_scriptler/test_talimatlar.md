# Konuşmacı Tanıma Test Adımları

Bu döküman, konuşmacı tanıma modelini test etmek için adım adım rehberlik sağlar.

## 1. Eğitim Pipeline'ını Çalıştırma

İlk olarak eğitim pipeline'ını çalıştırarak bir model eğitmeliyiz:

```bash
cd /home/zir_huz/Documents/robotek.subu.edu.tr
python -m training.run_pipeline --data_dir dataset --output_dir training_outputs
```

Bu komut şunları yapacaktır:
1. Ses dosyalarından özellikler çıkarır
2. RandomForest modeli eğitir
3. Modeli değerlendirir
4. Sonuçları kaydeder

## 2. Modeli Test Etme

### Basit Test

En temel test için `test_tanma.py` kullanılır:

```bash
cd /home/zir_huz/Documents/robotek.subu.edu.tr
python test_scriptler/test_tanma.py dataset/person1/chunk_50.wav
```

Çıktı örneği:
```
Audio file: dataset/person1/chunk_50.wav
Identified speaker: person1
Confidence: 92.35%

All probabilities:
  person1: 92.35%
  person2: 3.42%
  person3: 1.78%
  person4: 2.45%
```

### Detaylı Test (Loglama ile)

Daha detaylı testler için loglama özellikli `derin_tanma.py` kullanılır:

```bash
cd /home/zir_huz/Documents/robotek.subu.edu.tr
python test_scriptler/derin_tanma.py --audio_path dataset/person1/chunk_11.wav
```

Çıktı örneği:
```
2023-07-10 15:24:32 - INFO - Loading model from training_outputs/run_20230710_152245/model/model.pkl
2023-07-10 15:24:32 - INFO - Loading speaker mapping from training_outputs/run_20230710_152245/model/speaker_mapping.pkl
2023-07-10 15:24:32 - INFO - Extracting features from dataset/person1/chunk_11.wav
2023-07-10 15:24:32 - INFO - Predicting speaker
2023-07-10 15:24:32 - INFO - Getting prediction probabilities
2023-07-10 15:24:32 - INFO - Prediction confidence: 88.75%
2023-07-10 15:24:32 - INFO - Probability for person1: 88.75%
2023-07-10 15:24:32 - INFO - Probability for person2: 5.25%
2023-07-10 15:24:32 - INFO - Probability for person3: 3.12%
2023-07-10 15:24:32 - INFO - Probability for person4: 2.88%
2023-07-10 15:24:32 - INFO - Identified speaker: person1

Audio file: dataset/person1/chunk_11.wav
Identified speaker: person1
Confidence: 88.75%
```

### Konuşmacı Tanıma ve İlişkili Resmi Bulma

Konuşmacıyı tanıyıp, ilişkili resmi bulmak için `tanma_resim_gosterim.py` kullanılır:

```bash
cd /home/zir_huz/Documents/robotek.subu.edu.tr
python test_scriptler/tanma_resim_gosterim.py dataset/person2/chunk_1040.wav
```

Çıktı örneği:
```
==================================================
SPEAKER IDENTIFICATION RESULTS
==================================================
Audio file: dataset/person2/chunk_1040.wav
Identified speaker: person2
Confidence: 78.45%

Speaker image: dataset/person2/profile.png
To view the image, you can use an image viewer or display it in a GUI application.
==================================================
```

## 3. Tanıma Güvenilirliği (Confidence)

Tüm test scriptleri artık tanıma güvenilirliğini (confidence) göstermektedir. Bu değer, modelin yaptığı tahmin için ne kadar emin olduğunu yüzde (%) olarak gösterir.

### Tüm Olasılıkları Görüntüleme

Modelin tüm konuşmacılar için tahmin olasılıklarını görmek için `--show_all` parametresini kullanabilirsiniz:

```bash
python test_scriptler/test_tanma.py dataset/person1/chunk_50.wav --show_all
python test_scriptler/derin_tanma.py --audio_path dataset/person1/chunk_50.wav --show_all
python test_scriptler/tanma_resim_gosterim.py dataset/person1/chunk_50.wav --show_all
```

Güvenilirlik değeri 70%'in altındaysa, tüm olasılıklar otomatik olarak gösterilir. Bu, model hangi konuşmacı olduğu konusunda emin değilse, alternatif olasılıkları görebilmenizi sağlar.

## 4. Model Dizinini Manuel Olarak Belirtme

Eğer birden fazla eğitilmiş modeliniz varsa, hangisini kullanacağınızı manuel olarak belirleyebilirsiniz:

```bash
python test_scriptler/tanma_resim_gosterim.py dataset/person3/chunk_75.wav --model_dir training_outputs/run_20230709_183015/model
```

## 5. Eski Model Yapısını Kullanma

Eğer eski model dizin yapısı ile çalışıyorsanız (`outputs/models_*`), `--legacy` parametresini kullanabilirsiniz:

```bash
python test_scriptler/derin_tanma.py --audio_path dataset/person1/chunk_11.wav --legacy
```

## 6. Toplu Değerlendirme

Çok sayıda ses dosyasını toplu değerlendirmek için bir script kullanabilirsiniz:

```bash
# Örnek: test_all.sh
#!/bin/bash

TEST_DIR="test_data"
OUTPUT_FILE="test_results.txt"

echo "Starting batch evaluation..." > $OUTPUT_FILE
echo "Timestamp: $(date)" >> $OUTPUT_FILE
echo "----------------------------------------" >> $OUTPUT_FILE

for audio_file in $TEST_DIR/*.wav; do
    echo "Testing: $audio_file" >> $OUTPUT_FILE
    python test_scriptler/test_tanma.py "$audio_file" >> $OUTPUT_FILE 2>&1
    echo "----------------------------------------" >> $OUTPUT_FILE
done

echo "Evaluation complete!" >> $OUTPUT_FILE
```

## Sorun Giderme

### 1. Model Bulunamadı Hatası

Eğer "No trained models found" hatası alırsanız:
- Eğitim pipeline'ının başarıyla çalıştığından emin olun
- `--model_dir` parametresiyle model dizinini manuel olarak belirtin

### 2. Ses Dosyası Hatası

Eğer "Audio file not found" hatası alırsanız:
- Dosya yolunun doğru olduğundan emin olun
- Mutlak yerine göreceli dosya yolları kullanmayı deneyin

### 3. Özellik Çıkarım Hatası

Özellik çıkarım sorunları genellikle bozuk ses dosyalarından kaynaklanır:
- Ses dosyasının oynatılabilir olduğunu kontrol edin
- `extract_features` fonksiyonunun hata mesajlarını inceleyin

### 4. Düşük Güvenilirlik Değeri

Eğer düşük güvenilirlik değeri (Confidence < 70%) görüyorsanız:
- Eğitim veri setinde daha fazla ses örneği kullanmayı deneyin
- Özellik çıkarım parametrelerini iyileştirin
- Modeli daha fazla kriter ve daha fazla ağaç ile yeniden eğitin 