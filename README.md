# Диплом

Тема: Система автоматического контроля присутствия студентов на занятиях. 

Система, используя камеры наблюдения в аудиториях, детектирует все лица студентов во время занятия (массив лиц А). Затем сервер обращается к базе данных лиц университета (массив лиц В) и сопоставляет каждое лицо массива А к каждому лицу массива В *(сложность - O(n^2))*, таким образом определяя всех присутствующих студентов на занятии. После, все собранные данные записываются в электронный журнал посещаемости университета и хранилище фото-и видеоизображений.

Больше - тут: https://docs.google.com/document/d/1XiYXLKXKrHDYk3An2qt_oZPoI16rCP5xBJP9rzDxYfQ/edit#heading=h.9i6pm23ncttd

# Требования
  1. Python >= 3.8
  2. Все пакеты, указанные в requirenments.txt

# Запуск
  1. Поместите в папку input/database/ фотографии лиц, которые вы хотите распознать на изображении. При этом на фотографии настоятельно рекомендую оставить именно лицо, обрезая весь лишний фон. 
  2. Запустите main.py, указав параметр mode. Он может быть:
      - image - распознавание по фотографии
      - webcam - живое распознавание с вебкамерой
      - video - 5-ти секундная запись видео через вебкамеру и распознавание на нем
  



*readme будет дополняться по мере работы*

ОмГТУ, 2021
