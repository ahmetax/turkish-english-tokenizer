import os
import json
import argparse
import math

def split_jsonl(input_file, output_dir, max_size_mb=80):
    """
    Büyük JSONL dosyasını belirli boyut sınırına göre parçalara böler
    
    Args:
        input_file (str): JSONL girdi dosyası yolu
        output_dir (str): Çıktı dizini
        max_size_mb (int): Maksimum parça boyutu (MB)
    """
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Dosya adı temelini al
    base_filename = os.path.basename(input_file)
    name_without_ext = os.path.splitext(base_filename)[0]
    
    # Maksimum boyutu byte cinsinden hesapla
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Dosya boyutunu kontrol et
    file_size = os.path.getsize(input_file)
    if file_size <= max_size_bytes:
        # Dosya zaten yeterince küçükse, kopyala ve bitir
        output_path = os.path.join(output_dir, base_filename)
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        print(f"Dosya boyutu zaten sınırın altında. Kopyalandı: {output_path}")
        return [output_path]
    
    # Tahmini parça sayısını hesapla
    num_chunks = math.ceil(file_size / max_size_bytes)
    print(f"Dosya boyutu: {file_size / (1024 * 1024):.2f} MB")
    print(f"Tahmini parça sayısı: {num_chunks}")
    
    # Dosyayı satır bazında oku ve parçalara böl
    output_files = []
    chunk_index = 1
    current_size = 0
    current_chunk = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_size = len(line.encode('utf-8'))
            
            # Bu satırın eklenmesi durumunda parça boyutu aşılacak mı kontrol et
            if current_size + line_size > max_size_bytes and current_chunk:
                # Mevcut parçayı kaydet
                output_path = os.path.join(output_dir, f"{name_without_ext}_part{chunk_index:03d}.jsonl")
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write("".join(current_chunk))
                
                output_files.append(output_path)
                print(f"Parça {chunk_index} oluşturuldu: {output_path} ({current_size / (1024 * 1024):.2f} MB)")
                
                # Yeni parçaya geç
                chunk_index += 1
                current_size = 0
                current_chunk = []
            
            # Satırı mevcut parçaya ekle
            current_chunk.append(line)
            current_size += line_size
    
    # Son parçayı kaydet
    if current_chunk:
        output_path = os.path.join(output_dir, f"{name_without_ext}_part{chunk_index:03d}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write("".join(current_chunk))
        
        output_files.append(output_path)
        print(f"Parça {chunk_index} oluşturuldu: {output_path} ({current_size / (1024 * 1024):.2f} MB)")
    
    print(f"Toplam {chunk_index} parça oluşturuldu.")
    return output_files

def main():
    parser = argparse.ArgumentParser(description='Büyük JSONL dosyasını küçük parçalara böl')
    parser.add_argument('input_file', help='Giriş JSONL dosyası')
    parser.add_argument('output_dir', help='Çıktı dizini')
    parser.add_argument('--max-size', type=int, default=80, help='Maksimum parça boyutu (MB)')
    
    args = parser.parse_args()
    split_jsonl(args.input_file, args.output_dir, args.max_size)

if __name__ == '__main__':
    main()
