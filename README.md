# TurkishEnglish Tokenizer

## Projenin öyküsü: 

Grok 3 danışmanlığında başlattığım, yerel bilgisayarlarda çalışabilecek bir LLM'e ince ayar uygulaması çalışması, bir yan ürün olarak bu projeyi ortaya çıkardı.

Kullandığım 8 çekirdekli bilgisayarın 32GB RAM'i ve 6GB VRAM'i var. Diskim, SSD. 

Önce Mixtral-7B-v0.1 modelini denedik. Veriseti olarak Project Gutenberg çevirilerinden pasajlar, Türkçe atasözleri ve Türkçe deyimlerden oluşan bir set kullandık. Kullanmak istediğimiz modeli yüklemek mümkün olmadı. Bunun üzerine mistralai/Mistral-7B-v3.3 modeline geçtik. (Aslında Grok'un önerdiği mistralai/Mistral-7B-Instruct-v0.3 modeliydi. Bir karşıklık nedeniyle base modeli yüklemiş olduk.

Yaptığımız denemeler, 6GB VRAM'in bu model için yeterli olmadığını gösterdi.

Mecburen sadece CPU kullanacak bir koda geçtik. Ama o da bir çözüm sağlamadı. Hatta bir kaç kez bilgisayarı çökerttik.

Kuantizasyon denemeleri de başarısızlıkla sonuçlandı.

Sayısız denemeden sonra, modeli değiştirmeye karar verdik ve daha küçük bir modele geçtik: distilgpt2.

Bu model, CPU üzerinde sorunsuz şekilde çalıştı.

Eğitim çok uzun süreceği için, olası aksaklıklara önlem olarak belli aralıklarla checkpoint'ler oluşmasını ve kaldığımız yerden devam edebilmeyi sağladık.

Küçük bir verisetiyle yaptığımız deneme eğitimi yaklaşık 20 saat sürdü. Elde ettiğimiz sonuç da pek başarılı olmadı. Daha büyk bir verisetiyle, çok daha uzun sürecek bir eğitime geçmemiz şarttı.

Bu aşamada, Claude 3.7 Sonnet'e de danıştım.

Yaptığımız sohbetler ve testler sonucunda, daha etkin bir bellek optimizasyonu için standart tokenizer'ı kullanmak yerine, Türkçe verisetiyle eğitilecek bir tokenizer'ın daha yararlı olabileceğine karar verdik. Daha sonra bu kararımızı hem Türkçe, hem de İngilizce verileri uygun şekilde işleyecek bir tokenizer geliştirme yönünde değiştirdik.

Sonunda bu proje ortaya çıktı.

Yeni tokenizer, orijinal tokenizer'a kıyasla, Türkçe için %50'den fazla avantaj sağlarken, İngilizcede sadece %10 civarında bir kayba neden oluyor.

Projede hem Türkçe, hem de İngilizce verisetleri için Wikipedia kaynaklarını; temel model olarak da GPT2'yi kullandık.


## Project history:

This project emerged as a byproduct of a study I started with Grok 3 to fine-tune an LLM that could run on local computers.

The 8-core computer I use has 32GB RAM and 6GB VRAM. My disk is an SSD.

First, we tried the Mixtral-7B-v0.1 model. We used a dataset consisting of Project Gutenberg translations, Turkish proverbs, and Turkish idioms. It was not possible to load the model we wanted to use. So we switched to the mistralai/Mistral-7B-v3.3 model. (Actually, the model Grok suggested was mistralai/Mistral-7B-Instruct-v0.3. Due to a problem, we loaded the base model.

Our experiments showed that 6GB VRAM was not enough for this model.

We had to switch to a code that would only use the CPU. But that didn't provide a solution either. We even crashed the computer a few times.

Quantization experiments also failed.

After countless experiments, we decided to change the model and switched to a smaller model: distilgpt2.

This model worked flawlessly on the CPU.

Since the training would take a long time, we ensured that checkpoints were created at certain intervals to prevent possible problems and to continue where we left off.

The test training we did with a small dataset lasted about 20 hours. The result we got was not very successful either. We had to switch to a much longer training with a larger dataset.

At this stage, Claude 3.7 I also consulted Sonnet.

As a result of our conversations and tests, we decided that instead of using the standard tokenizer for more efficient memory optimization, a tokenizer that would be trained on the Turkish dataset would be more useful. Later, we changed our decision to develop a tokenizer that would handle both Turkish and English data properly.

Finally, this project emerged.

The new tokenizer provides more than 50% advantage for Turkish compared to the original tokenizer, while only causing a loss of around 10% for English.

In the project, we used Wikipedia sources for both Turkish and English datasets and GPT2 as the base model.


