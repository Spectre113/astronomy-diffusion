Небольшая инструкция, как запустить обучение и как проверить результат.

1. Не пытайтесь запустить код обучения локально, в колабе, в датахабе от Инно. Единственный способ адекватного запуска - каггл с 2 ГПУ. Либо надо понижать параметры

2. Сохранены чекпоинты 4 и 5. Для проверки откройте iigenerator.ipynb и выполните последние 2 ячейки, начиная с комментария "Run this cell and the cell below for inference". Первые ячейки, обучение и CSV для проверки не нужны. Рабочая папка должна быть корнем репозитория или Task2.

3. Можете добавить свои картинки (желательно, с космосом), но в последней ячейке надо будет указать названия файлов

4. Чекпоинт можно поменять в предпоследней ячейке: checkpoint_name = "checkpoint-5". Также доступен "checkpoint-4".

5. Проверку можно запускать локально!!!

Зависимости для проверки (Python 3.10):
Для NVIDIA GPU сначала установите PyTorch с подходящей версией CUDA по официальной инструкции:
https://pytorch.org/get-started/previous-versions/#v280
Например, для CUDA 12.6:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

Затем установите зависимости:
python -m pip install -r Task2/requirements-inference.txt
Если рабочая папка - Task2, используйте requirements-inference.txt без префикса Task2/.
В редакторе ноутбука выберите окружение Python, в которое установлены зависимости.

При первом запуске скачиваются базовые компоненты Stable Diffusion Inpainting с Hugging Face. Локально сохранены только LoRA-адаптеры.
Для проверки используется зеркало https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting вместо недоступного runwayml/stable-diffusion-inpainting.
