import os
import glob
import json
import random
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import torch
from pathlib import Path

# Konfigürasyon
class Config:
    # Veri kaynak türü: "txt" veya "jsonl"
    DATA_SOURCE_TYPE = "jsonl"
    
    # Türkçe veri yolu (tek dosya veya klasör)
    TURKCE_JSONL_PATH = "./turkce_veriseti"  # JSONL dosya veya klasör - DEĞİŞTİRİN!
    
    # JSONL parametreleri
    JSONL_TEXT_FIELD = "text"  # JSONL içindeki metin alanı
    JSONL_EXTENSION = "*.jsonl"  # JSONL dosya uzantısı
    
    # İngilizce veri oranı parametreleri
    INGILIZCE_VERI_ORANI = 0.3  # İngilizce verinin toplam veri içindeki oranı (0.0-1.0)
    INGILIZCE_JSONL_PATH = "./ingilizce_veriseti"  # İngilizce JSONL dosya veya klasör
    
    # Tokenizer parametreleri
    BASE_MODEL = "gpt2"  # Temel model
    VOCAB_SIZE = 50000  # Kelime dağarcığı boyutu
    MIN_FREQUENCY = 2  # Minimum token frekansı
    OUTPUT_DIR = "./turkce_ingilizce_tokenizer"  # Çıktı dizini

    # Veri örnekleme ve işleme parametreleri
    SAMPLE_SIZE = None  # Veri örnekleme boyutu (None=tümü)
    MAX_SAMPLE_LENGTH = 100000  # Her bir metin örneğinin maksimum uzunluğu

def get_jsonl_files(path):
    """Belirtilen yoldan JSONL dosyalarını bulur ve listeler"""
    jsonl_files = []
    
    # Eğer tek bir dosya ise
    if os.path.isfile(path) and path.endswith('.jsonl'):
        return [path]
    
    # Eğer klasör ise, içindeki tüm jsonl dosyalarını bul
    elif os.path.isdir(path):
        jsonl_files = glob.glob(os.path.join(path, Config.JSONL_EXTENSION))
        print(f"'{path}' klasöründe {len(jsonl_files)} JSONL dosyası bulundu.")
        return jsonl_files
    
    else:
        print(f"UYARI: '{path}' bulunamadı veya bir JSONL dosyası/klasörü değil.")
        return []

def load_jsonl_data_from_files(jsonl_files, text_field="text", sample_size=None, max_length=100000):
    """Birden çok JSONL dosyasından metin verilerini okur"""
    texts = []
    processed_files = 0
    total_lines = 0
    read_lines = 0
    
    # Örnekleme boyutu kontrolü
    actual_sample_size = sample_size
    if sample_size is not None:
        # Her dosyadan kaç örnek alınacağını hesapla
        per_file_samples = max(1, sample_size // len(jsonl_files))
        print(f"Her dosyadan yaklaşık {per_file_samples} örnek alınacak.")
    
    # Her dosyayı işle
    for jsonl_file in jsonl_files:
        file_texts = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    total_lines += 1
                    
                    if sample_size is not None and read_lines >= sample_size:
                        break
                    
                    try:
                        # JSON satırını parse et
                        data = json.loads(line.strip())
                        
                        # Belirtilen alandan metni al
                        if text_field in data:
                            text = data[text_field]
                            
                            # Boş veya çok kısa metinleri atla
                            if text and len(text) > 10:
                                # Metni maksimum uzunluğa kısıtla
                                text = text[:max_length]
                                file_texts.append(text)
                                read_lines += 1
                        
                    except json.JSONDecodeError:
                        print(f"Hatalı JSON satırı ({jsonl_file}, satır {i+1}): {line[:50]}...")
                    except Exception as e:
                        print(f"Satır işleme hatası ({jsonl_file}, satır {i+1}): {str(e)}")
            
            # Dosyadan okunan metinleri ana listeye ekle
            texts.extend(file_texts)
            processed_files += 1
            print(f"Dosya işlendi: {jsonl_file} - {len(file_texts)} metin parçası alındı.")
            
        except FileNotFoundError:
            print(f"HATA: {jsonl_file} dosyası bulunamadı.")
        except Exception as e:
            print(f"Dosya okuma hatası ({jsonl_file}): {str(e)}")
    
    print(f"Toplam {processed_files}/{len(jsonl_files)} dosya işlendi.")
    print(f"Toplam {read_lines}/{total_lines} satır okundu.")
    print(f"Toplam {len(texts)} metin parçası toplandı.")
    
    return texts

def create_temp_text_files(texts, output_dir):
    """Metin listesinden geçici metin dosyaları oluşturur"""
    os.makedirs(output_dir, exist_ok=True)
    
    text_files = []
    chunk_size = 1000  # Her dosyada maksimum metin sayısı
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        file_path = os.path.join(output_dir, f"chunk_{i//chunk_size + 1}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(chunk))
        
        text_files.append(file_path)
        print(f"Dosya oluşturuldu: {file_path} ({len(chunk)} metin)")
    
    return text_files

def main():
    """Ana fonksiyon - tokenizer eğitimi sürecini yönetir"""
    config = Config()
    
    # Çıktı dizini oluşturma
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Geçici dizin oluşturma
    temp_dir = os.path.join(config.OUTPUT_DIR, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Türkçe JSONL dosyalarını bul
    print("Türkçe JSONL dosyaları aranıyor...")
    turkish_jsonl_files = get_jsonl_files(config.TURKCE_JSONL_PATH)
    
    if not turkish_jsonl_files:
        print("HATA: Hiç Türkçe JSONL dosyası bulunamadı!")
        return
    
    # Türkçe metinleri yükle
    print("Türkçe JSONL dosyalarından metin yükleniyor...")
    turkish_texts = load_jsonl_data_from_files(
        turkish_jsonl_files,
        text_field=config.JSONL_TEXT_FIELD,
        sample_size=config.SAMPLE_SIZE,
        max_length=config.MAX_SAMPLE_LENGTH
    )
    
    # Türkçe metinleri geçici dosyalara yaz
    print("Türkçe metinlerden geçici dosyalar oluşturuluyor...")
    turkish_files = create_temp_text_files(turkish_texts, os.path.join(temp_dir, "turkish"))
    print(f"Toplam {len(turkish_files)} Türkçe metin dosyası hazırlandı.")
    
    # İngilizce metinleri hazırla
    english_files = []
    if config.INGILIZCE_VERI_ORANI > 0:
        # İngilizce JSONL dosyalarını bul
        print("İngilizce JSONL dosyaları aranıyor...")
        english_jsonl_files = get_jsonl_files(config.INGILIZCE_JSONL_PATH)
        
        if english_jsonl_files:
            # İngilizce metinleri yükle
            print("İngilizce JSONL dosyalarından metin yükleniyor...")
            english_texts = load_jsonl_data_from_files(
                english_jsonl_files,
                text_field=config.JSONL_TEXT_FIELD,
                sample_size=config.SAMPLE_SIZE,
                max_length=config.MAX_SAMPLE_LENGTH
            )
            
            # İngilizce metinleri geçici dosyalara yaz
            print("İngilizce metinlerden geçici dosyalar oluşturuluyor...")
            english_files = create_temp_text_files(english_texts, os.path.join(temp_dir, "english"))
            print(f"Toplam {len(english_files)} İngilizce metin dosyası hazırlandı.")
        else:
            print("UYARI: İngilizce JSONL dosyası bulunamadı.")
            # Varsayılan bir İngilizce örnek oluştur
            print("Basit bir İngilizce örnek dosyası oluşturuluyor...")
            eng_sample_dir = os.path.join(temp_dir, "english")
            os.makedirs(eng_sample_dir, exist_ok=True)
            eng_sample_file = os.path.join(eng_sample_dir, "english_sample.txt")
            
            # Basit bir İngilizce metin oluştur
            with open(eng_sample_file, 'w', encoding='utf-8') as f:
                f.write("This is a sample English text. It's used to ensure the tokenizer has some English tokens.")
                f.write("The tokenizer will be trained on both Turkish and English to support bilingual capabilities.")
            
            english_files = [eng_sample_file]
    
    # Dil oranını dengele
    all_files = turkish_files + english_files
    
    if english_files and turkish_files and config.INGILIZCE_VERI_ORANI > 0:
        # Dosya boyutlarını hesapla
        total_turkish_size = sum(os.path.getsize(f) for f in turkish_files)
        total_english_size = sum(os.path.getsize(f) for f in english_files)
        
        # Mevcut oran
        if total_turkish_size + total_english_size > 0:
            current_english_ratio = total_english_size / (total_english_size + total_turkish_size)
            
            print(f"Mevcut İngilizce veri oranı: {current_english_ratio:.2f}")
            print(f"Hedef İngilizce veri oranı: {config.INGILIZCE_VERI_ORANI:.2f}")
            
            # Eğer İngilizce oranı hedeften düşükse, İngilizce veriyi çoğalt
            if current_english_ratio < config.INGILIZCE_VERI_ORANI and english_files:
                repeat_count = max(1, int((config.INGILIZCE_VERI_ORANI * total_turkish_size) / 
                                ((1 - config.INGILIZCE_VERI_ORANI) * total_english_size)))
                
                print(f"İngilizce veri {repeat_count}x çoğaltılıyor...")
                english_files = english_files * repeat_count
                
                # Dosya listesini güncelle
                all_files = turkish_files + english_files
    
    # Veri karıştırma
    random.shuffle(all_files)
    print(f"Tokenizer eğitimi için toplam {len(all_files)} dosya hazırlandı.")
    
    # Tokenizer eğitimi
    print("Tokenizer eğitiliyor...")
    tokenizer = ByteLevelBPETokenizer()
    
    tokenizer.train(
        files=all_files, 
        vocab_size=config.VOCAB_SIZE, 
        min_frequency=config.MIN_FREQUENCY,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    
    # Tokenizer'ı kaydet
    print(f"Tokenizer kaydediliyor: {config.OUTPUT_DIR}")
    tokenizer.save_model(config.OUTPUT_DIR)
    
    # Transformers uyumlu tokenizer oluştur
    print("Transformers uyumlu tokenizer oluşturuluyor...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.OUTPUT_DIR)
    
    # GPT-2 tokenizer ile uyumlu olması için gereken ayarlar
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    
    # Tokenizer'ı test et
    test_texts = [
        # Türkçe test cümleleri
        "Türkçe dil modellemesi için özel bir tokenizer eğitiyoruz.",
        "İstanbul Boğazı'nın muhteşem manzarası eşliğinde çay içmek paha biçilemez.",
        # İngilizce test cümleleri
        "We are training a special tokenizer for Turkish language modeling.",
        "Drinking tea with the magnificent view of the Bosphorus in Istanbul is priceless."
    ]
    
    print("Tokenizer test ediliyor...")
    for i, test_text in enumerate(test_texts):
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"\nTest {i+1}: {'Türkçe' if i < 2 else 'İngilizce'}")
        print(f"Metin: {test_text}")
        print(f"Token sayısı: {len(encoded)}")
        print(f"Çözülmüş: {decoded}")
    
    # Tokenizer'ı kaydet
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"Transformers uyumlu tokenizer kaydedildi: {config.OUTPUT_DIR}")
    
    # Model ile uyumlu hale getir
    print(f"Temel model yükleniyor: {config.BASE_MODEL}")
    try:
        # Modeli yükle
        model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL)
        
        # Modelin kelime dağarcığı boyutunu kontrol et
        old_vocab_size = model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(tokenizer)
        
        print(f"Eski kelime dağarcığı boyutu: {old_vocab_size}")
        print(f"Yeni kelime dağarcığı boyutu: {new_vocab_size}")
        
        # Embedding katmanını genişlet
        print("Embedding katmanı genişletiliyor...")
        model.resize_token_embeddings(new_vocab_size)
        
        # Modeli kaydet
        output_model_dir = os.path.join(config.OUTPUT_DIR, "model")
        os.makedirs(output_model_dir, exist_ok=True)
        model.save_pretrained(output_model_dir)
        print(f"Uyumlu model kaydedildi: {output_model_dir}")
        
    except Exception as e:
        print(f"Model uyumluluk işlemi sırasında hata: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Orijinal tokenizer ile karşılaştırma
    print("\nTokenizer karşılaştırması:")
    original_tokenizer = GPT2Tokenizer.from_pretrained(config.BASE_MODEL)
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token
    
    languages = ["Türkçe", "Türkçe", "İngilizce", "İngilizce"]
    
    for i, text in enumerate(test_texts):
        original_tokens = original_tokenizer.encode(text)
        new_tokens = tokenizer.encode(text)
        
        print(f"\n{languages[i]} metin: {text}")
        print(f"Orijinal tokenizer: {len(original_tokens)} token")
        print(f"Yeni tokenizer: {len(new_tokens)} token")
        
        if languages[i] == "Türkçe":
            performance_metric = len(new_tokens) < len(original_tokens)
            if performance_metric:
                efficiency = (1 - len(new_tokens) / len(original_tokens)) * 100
                print(f"Türkçe verimlilik artışı: %{efficiency:.2f}")
            else:
                inefficiency = (len(new_tokens) / len(original_tokens) - 1) * 100
                print(f"Türkçe verimlilik kaybı: %{inefficiency:.2f}")
        else:
            performance_metric = len(new_tokens) <= len(original_tokens) * 1.2  # %20 tolerans
            if performance_metric:
                print("İngilizce yeteneği korunmuş (kabul edilebilir token sayısı)")
            else:
                inefficiency = (len(new_tokens) / len(original_tokens) - 1) * 100
                print(f"İngilizce verimlilik kaybı: %{inefficiency:.2f}")
    
    # Geçici dosyaları temizle (opsiyonel)
    # print("Geçici dosyalar temizleniyor...")
    # import shutil
    # shutil.rmtree(temp_dir)
    
    print("\nÖNEMLİ KULLANIM TALİMATLARI:")
    print("-----------------------------")
    print(f"1. Eğitim kodunuzda, model ve tokenizer yüklemek için şu yolu kullanın: {config.OUTPUT_DIR}")
    print("2. Fine-tuning veri setinizde hem Türkçe hem İngilizce içerik bulundurun.")
    print("3. Eğitim sırasında ilk epoch'larda düşük öğrenme hızı kullanarak embedding katmanının adapte olmasını sağlayın.")

if __name__ == "__main__":
    main()
