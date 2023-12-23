# poetry-tree

Первая домашка была сделана на простом дереве решений, но дальше стало понятно, что это не годится, и я переделал все на модель с трансформером, из домашки по NLP.
Модель осуществляет перевод с русского языка на английский. Трансформер сильно упрощен для ускорения, всего один слой, по одному блоку внимания, и размерность внутренних состояний всего 64. Языковые словари составляются на основе данных только тренировочного датасета.

Данные для обучения, валидации и теста из файла data/data.txt
В параметрах количество эпох обучения указано 2, для быстрой проверки работоспособности модели. Но для хорошего перевода нужно около 10 эпох.

Через Fire организован запуск в commands.py, доступные комманды: train, infer, run_onnx

Вся конфигурация организована через Hydra, конфигурация находится в config/config.yaml

Хранение модели, результатов и прочего организовано через DVC, в гугл-диске https://drive.google.com/drive/folders/1M4Fe8nT36ufluQJB-IRDCHz03aB7iOoc

Логирование обучения модели сделано через MLFlow, выведены 4 графика метрик, сохранен тег с ID текущего коммита, словари с параметрами из Hydra, и полученными во время обучения.

Этап инференса модели по комманде infer проводит полную обработку тестовой части датасета и считает BLEU метрику качества перевода.

Этап инференса модели по комманде run_onnx загружает последнюю сохраненную модель в формате ONNX и производит перевод одного короткого предложения моделью с русского языка на английский.

Этап инференса модели на Triton Server можно запустить командой run_triton_inference (файл client.py). Для упрощения проверки реализована также как в предыдущем случае захардкоженная проверка работоспособности переводом одного предложения. В историях коммитов еще осталась версия, где я в соответствии с лекцией реализовывал 3 бэкенда с ансамблем и питоновским бэкендом для токенизации предложений и ONNX моделью. Но я решил убрать этот вариант, т.к. моя кастомная версия токенизатора тянет в докер-образ PyTorch, что, думаю, неоправданно усложнит проверку работоспособности. Но вообще все здорово получилось, и передать сохраненные словари в докер-образ, и установить внутри библиотеки, и отправить ансамблевым бэкендом туда данные, и далее в модель, и получить результат. В моем случае модель предусматривает циклический запуск до появления на выходе токена конца предложения, и я не разобрался как реализовать такую логику прямо внутри Triton Server, а не со стороны клиента дергать модель, пока она не выдаст конечный токен. Если проверяющая наши домашки команда доберется до этого - буду рад совету.

Также не смотря на то, что у моего ноутбука есть внешняя NVIDIA видеокарта, и PyTorch её успешно видит, и модель нормально работает на GPU - мне так и не удалось разобраться, почему докер с Triton Server успешно запускает модель с требованием инференсить её на GPU, но при подаче входных данных выдает ошибку "CUDA failure 209: no kernel image is available for execution on the device". Я пробовал другие версии докер-образа - ничего не помогло. Тоже буду рад совету по этому поводу.

Системная конфигурация:
ОС: Windows 10 Version 22H2, OS Build 19045.3803, вся работа с ДЗ ведется в WSL2 c Ubuntu 20.04.6 LTS
CPU: Intel Core i7-5700HQ CPU @ 2.70GHz
vCPU: 8 (для докер-образа при проверке производительности было доступно 4)
RAM: 16 GB
В model_repository по вышеописанным причинам осталась только ONNX-модель 1-й версии с config.pbtxt:

<!-- prettier-ignore-start -->
❯ tree ./model_repository/
./model_repository/
└── [4.0K]  onnx-model
    ├── [4.0K]  1
    │   ├── [  12]  .gitignore
    │   ├── [   0]  .gitkeep
    │   ├── [6.8M]  model.onnx
    │   └── [  93]  model.onnx.dvc
    └── [1.5K]  config.pbtxt

2 directories, 5 files
<!-- prettier-ignore-end -->

Т.к. мне не удалось заставить модель работать с GPU - все замеры производительности проводились на CPU.
Максимальный батч 4, без динамического батчинга:
Concurrency: 1, throughput: 284.175 infer/sec, latency 3515 usec
Concurrency: 2, throughput: 392.583 infer/sec, latency 5089 usec
Concurrency: 3, throughput: 416.183 infer/sec, latency 7202 usec
Concurrency: 4, throughput: 409.053 infer/sec, latency 9771 usec
Concurrency: 5, throughput: 308.106 infer/sec, latency 16212 usec
Concurrency: 6, throughput: 390.961 infer/sec, latency 15357 usec
Concurrency: 7, throughput: 400.339 infer/sec, latency 17473 usec

Максимальный батч 4, с динамическим батчингом, но без max_queue_delay_microseconds:
Concurrency: 1, throughput: 271.653 infer/sec, latency 3677 usec
Concurrency: 2, throughput: 381.166 infer/sec, latency 5243 usec
Concurrency: 3, throughput: 407.745 infer/sec, latency 7353 usec
Concurrency: 4, throughput: 413.767 infer/sec, latency 9659 usec
Concurrency: 5, throughput: 377.673 infer/sec, latency 13233 usec
Concurrency: 6, throughput: 379.769 infer/sec, latency 15794 usec
Concurrency: 7, throughput: 369.949 infer/sec, latency 18911 usec

Максимальный батч 4, с динамическим батчингом, max_queue_delay_microseconds=200:
Concurrency: 1, throughput: 283.118 infer/sec, latency 3528 usec
Concurrency: 2, throughput: 354.961 infer/sec, latency 5630 usec
Concurrency: 3, throughput: 367.307 infer/sec, latency 8162 usec
Concurrency: 4, throughput: 363.095 infer/sec, latency 11007 usec
Concurrency: 5, throughput: 380.779 infer/sec, latency 13133 usec
Concurrency: 6, throughput: 371.118 infer/sec, latency 16159 usec
Concurrency: 7, throughput: 368.243 infer/sec, latency 19001 usec

Максимальный батч 4, с динамическим батчингом, max_queue_delay_microseconds=2000:
Concurrency: 1, throughput: 169.05 infer/sec, latency 5911 usec
Concurrency: 2, throughput: 270.529 infer/sec, latency 7387 usec
Concurrency: 3, throughput: 281.94 infer/sec, latency 10636 usec
Concurrency: 4, throughput: 397.508 infer/sec, latency 10055 usec
Concurrency: 5, throughput: 413.81 infer/sec, latency 12082 usec

Максимальный батч 4, с динамическим батчингом, max_queue_delay_microseconds=5000:
Concurrency: 1, throughput: 119.857 infer/sec, latency 8336 usec
Concurrency: 2, throughput: 192.577 infer/sec, latency 10382 usec
Concurrency: 3, throughput: 234.272 infer/sec, latency 12798 usec
Concurrency: 4, throughput: 366.214 infer/sec, latency 10920 usec
Concurrency: 5, throughput: 376.089 infer/sec, latency 13286 usec
Concurrency: 6, throughput: 415.764 infer/sec, latency 14419 usec
Concurrency: 7, throughput: 396.044 infer/sec, latency 17657 usec

Максимальный батч 4, с динамическим батчингом, max_queue_delay_microseconds=2000, добавлена Instance Group Count 2:
Concurrency: 1, throughput: 96.8046 infer/sec, latency 10324 usec
Concurrency: 2, throughput: 134.779 infer/sec, latency 14835 usec
Concurrency: 3, throughput: 167.545 infer/sec, latency 17900 usec
Concurrency: 4, throughput: 201.867 infer/sec, latency 19809 usec
Concurrency: 5, throughput: 219.744 infer/sec, latency 22710 usec
Concurrency: 6, throughput: 222.848 infer/sec, latency 26913 usec
Concurrency: 7, throughput: 213.447 infer/sec, latency 32733 usec

В общем, результат таков, что всё что можно получить от машины без GPU - использовать по максимуму имеющиеся возможности по обработке несколькими потоками. И если у моей машины доступно докеру 4 потока, то и максимум производительности наблюдается с concurrency 3-5. Даже и без динамического батчинга и без второго инстанса, которые только уменьшают производительность.
