# Konuşmacı Tanıma Test Scriptleri / Speaker Identification Test Scripts

Bu dizin, konuşmacı tanıma modellerini değerlendirmek ve kullanmak için test scriptlerini içerir.

## Scriptler / Scripts

- `test_tanma.py`: Konuşmacı tanıma için temel script / Basic script for speaker identification
- `derin_tanma.py`: Konuşmacı tanıma için gelişmiş loglama özellikleri olan script / Advanced script with logging for speaker identification
- `tanma_resim_gosterim.py`: Konuşmacı tanıma ve resim gösterimi için script / Script for speaker identification with image display
- `simple_extract.py`: Özellik çıkarım yardımcı modülü / Feature extraction utility module

## Kullanım / Usage

### Hazırlık / Preparation

Test scriptlerini kullanmadan önce eğitim pipeline'ını çalıştırdığınızdan emin olun:

```bash
python -m training.run_pipeline --data_dir dataset --output_dir training_outputs
```

Bu komut, ses dosyalarından özellikleri çıkarır, modeli eğitir ve değerlendirir. Tüm çıktılar timestamp eklenmiş dizinlere kaydedilir.

### Temel Test / Basic Testing

En basit konuşmacı tanıma testi için:

```bash
python test_tanma.py <ses_dosyasi_yolu> [--model_dir <model_dizini>]
```

Örnek:
```bash
python test_tanma.py dataset/person1/chunk_50.wav
```

Örnek çıktı:
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

### Gelişmiş Test (Loglama ile) / Advanced Testing with Logging

Detaylı loglama ile konuşmacı tanıma için:

```bash
python derin_tanma.py --audio_path <ses_dosyasi_yolu> [--model_dir <model_dizini>] [--legacy] [--show_all]
```

Örnek:
```bash
python derin_tanma.py --audio_path dataset/person1/chunk_11.wav
```

### Konuşmacı Tanıma ve Resim Gösterimi / Speaker Identification with Image Display

Konuşmacıyı tanımlayıp ilgili resmi bulmak için:

```bash
python tanma_resim_gosterim.py <ses_dosyasi_yolu> [--model_dir <model_dizini>] [--data_dir <veri_dizini>] [--legacy] [--show_all]
```

Örnek:
```bash
python tanma_resim_gosterim.py dataset/person2/chunk_1040.wav
```

## Tanıma Güvenilirliği / Confidence Score

Tüm test scriptleri, tanıma sonuçlarıyla birlikte bir güvenilirlik (confidence) değeri gösterir. Bu değer, modelin tahmininde ne kadar emin olduğunu belirtir.

- Yüksek güvenilirlik değeri (>90%): Model sonuçtan oldukça emindir
- Orta güvenilirlik değeri (70-90%): Model sonuçtan emin ancak alternatifler de olabilir
- Düşük güvenilirlik değeri (<70%): Model sonuçtan çok emin değildir

### Tüm Olasılıkları Görüntüleme / Show All Probabilities

Tüm olasılıkları görmek için `--show_all` parametresini kullanabilirsiniz:

```bash
python test_tanma.py dataset/person1/chunk_50.wav --show_all
```

Güvenilirlik değeri 70%'in altında olduğunda script otomatik olarak tüm olasılıkları gösterir.

## Notlar / Notes

- Scriptler hem yeni eğitim pipeline çıktı yapısı (`training_outputs/run_*/model/`) hem de eski yapı (`outputs/models_*/`) ile çalışabilir.
- Eski dizin yapısını kullanmak için `--legacy` bayrağını kullanın.
- Eğer `--model_dir` belirtilmezse, scriptler en son modeli otomatik olarak bulur.
- Eğer `--data_dir` belirtilmezse, varsayılan olarak `dataset` dizini kullanılır.

## Hata Giderme / Troubleshooting

- **Model Bulunamadı Hatası**: Eğer "No trained models found" hatası alırsanız, eğitim pipeline'ının başarıyla çalıştığından emin olun veya `--model_dir` parametresiyle model dizinini manuel olarak belirtin.
- **Ses Dosyası Hatası**: Eğer "Audio file not found" hatası alırsanız, dosya yolunun doğru olduğundan emin olun.
- **Özellik Çıkarım Hatası**: Özellik çıkarım sorunları genellikle bozuk ses dosyalarından kaynaklanır. Ses dosyasının düzgün çalıştığından emin olun.
- **Düşük Güvenilirlik**: Eğer düşük güvenilirlik değerleri görüyorsanız, modeli daha fazla ses örneği ile yeniden eğitmeyi veya model parametrelerini ayarlamayı deneyin.