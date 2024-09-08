<body>
  <h1 align='center'>Решение команды "MISIS GO"</h1>
  <p align='center'></p>
  <div>
    <h2>Содержание / Навигация:</h2>
    <ul>
      <li><a href='#11'>Описание</a></li>
      <ol>
        <li><a href='#12'>Библиотеки</a></li>
        <li><a href='#13'>Навигация по директории</a></li>
        <li><a href='#14'>Запуск моделей</a></li>
      </ol>
      <li><a href='#21'>Решение первой задачи</a></li>
      <ol>
        <li><a href='#22'>Описание</a></li>
        <li><a href='#23'>Решение</a></li>
        <li><a href='#24'>Преимущества</a></li>
      </ol>
      <li><a href='#31'>Решение второй задачи</a></li>
      <ol>
        <li><a href='#32'>Описание</a></li>
        <li><a href='#33'>Решение</a></li>
        <li><a href='#34'>Преимущества</a></li>
      </ol>
      <li><a href='#41'>Решение дополнительной задачи</a></li>
      <ol>
        <li><a href='#42'>Описание</a></li>
        <li><a href='#43'>Решение</a></li>
        <li><a href='#44'>Преимущества</a></li>
      </ol>
    </ul>
  </div>
  <div>
    <h2 id='11'>Описание:</h2>
    <h3 id='12'>👉 Библиотеки</h3>
    <p>Все <b>необходимые</b> библиотеки можно установить, написав в терминал: <code>pip install -r requirements.txt</code></p>
    <h3 id='13'>👉 Навигация по директории:</h3>
    <pre>
      - full_launch.py // модель для запуска всего проекта
      - requirements.txt // текстовый файл с названиями библиотек и их версиями
      - <b>first_task // папка с решением задачи определения профессии</b>
      - >> main.py // модель
      - >> *filename*.joblib // веса к модели
      - <b>second_task // папка с решением задачи определения заработной платы</b>
      - >> main.py // модель
      - >> *filename*.joblib // веса к модели
      - <b>third_task // папка с решением дополнительной задачи по унификации профессий</b>
      - >> main.py // модель
    </pre>
    <h3 id='14'>👉 Запуск моделей:</h3>
    <p>Для запуска <b>всего проекта</b>: <code>py full_launch.py <путь к test.csv 1> <путь к test.csv 2> <куда выгружать submissions.csv 1> <куда выгружать submissions.csv 2></code></p>
    <blockquote>Чтобы запустить <b>full_launch.py</b> необходимо, чтобы был указан путь к каждому <b>test.csv</b>, кроме того, желательно, чтобы каждый <b>submissions.csv</b> находился в своей папки</blockquote>
    <p>Для запуска <b>одной из модели</b>:</p>
    <p>
      1. Переходим в директорию, где лежит модель: <b>first_task</b>, <b>second_task</b>, <b>third_task</b>. </p>
     <p> 2. Запускаем модель и указываем путь к <b>test.csv</b> и к <b>submissions.csv</b> с помощью <code>py main.py <путь test.csv> <путь submissions.csv></code>
    </p>
  </div>
  <div>
    <h2 id='21'>1️⃣ Решение первой задачи:</h2>
    <h3 id='22'>👉 Описание:</h3>
        <p>Для решения первой задачи "<b>Определение наиболее предпочтительной профессии</b>" были использованы библиотеки: <code>numpy, pandas, nltk, stop-words, scikit-learn</code>. Само решение предполагало использование следующих подходов: <b>препроцессинг</b>, <b>векторизация</b>, <b>классификация</b>.</p>
    <h3 id='23'>👉 Решение:</h3>
    <p>На первом шаге производится препроцессинг текстов резюме с помощью методов обработки естественного языка (NLP), в том числе первичная обработка текста, удаление стоп-слов и стемминг.</p>
    <p>На втором шаге производилась векторизация предобработанных текстов с помощью метода TF-IDF.</p>
    <p>На третьем шаге для определения итоговой профессии по резюме применялась модель классификации LogisticRegression с предварительной настройкой параметров с помощью поиска по сетке.</p>
    <h3 id='24'>👉 Преимущества:</h3>
    <ul>
      <li>Высокая точность классификации.</li>
      <li>Возможность работать с маломощными вычислительными ресурсами.</li>
    </ul>
  </div>
    <div>
    <h2 id='31'>2️⃣ Решение второй задачи</h2>
    <h3 id='32'>👉 Описание:</h3>
      <p>  Для решения второй задачи "<b>Предсказывание возможной зарплатной перспективы для специальности</b>" были использованы библиотеки: <code>numpy, pandas, nltk, scikit-learn</code>. Само решение предполагало использование следующих подходов: <b>отбор признаков</b>, <b>предобработка данных</b>, <b>регрессионный анализ</b>. </p>
    <h3 id='33'>👉 Решение:</h3>
      <p>На первом шаге производится отбор наиболее релевантных для предсказания зарплаты признаков: <b>местоположение</b>, <b>опыт работы</b>, <b>требуемые навыки</b>, <b>название профессиональной сферы</b>.</p>
      <p>Далее производится обработка текстовых, категориальных и целочисленных признаков для улучшения точности предсказаний:</p>
      <ul>
        <li><b>Для текстовых</b>: обработка NLP методами.</li>
        <li><b>Для категориальных</b>: one-hot encoding.</li>
        <li><b>Для целочисленных</b>: удаление выбросов и стандартизация.</li>
      </ul>
      <p>На получившихся признаках применяется модель регрессии RandomForest с предварительной настройкой параметров с помощью поиска по сетке.</p>
    <h3 id='34'>👉 Преимущества:</h3>
      <ul>
        <li>Гибкая модель, которая адаптируется под различные данные.</li>
      </ul>
  </div>
    <div>
    <h2 id='41'>3️⃣ Решение третьей задачи:</h2>
    <h3 id='42'>👉 Описание:</h3>
      <p>  Для решения третьей задачи "<b>Унификация профессий</b>" были использованы библиотеки: <code>numpy, pandas, nltk, transformer, pytorch</code>. Само решение предполагало использование следующих подходов: <b>transformer</b>, <b>векторизация</b>, <b>кластеризация</b>. </p>
    <h3 id='43'>👉 Решение:</h3>
      <p>На первом шаге происходит корректировка данных, удаление ненужных символов для упрощения анализа файла и дубликатов уже существующих профессий</p>
      <p>На втором шаге производится <b>токенизация</b> и <b>лемматизация</b>, после чего убираются стоп-слова по типу: <b>на</b>, <b>и</b>, <b>это</b> и т.д.</p>
      <p>На последующем шаге используется <b>transformer</b> от 🤗 Hugging Face для решения задачи на основе <b>векторизации текстов</b> и последующей <b>кластеризации</b>.</p>
      <p>На конечном шаге происходит группировка профессий.</p>
    <h3 id='44'>👉 Преимущества:</h3>
      <ul>
        <li>Точная семантическая кластеризация.</li>
      </ul>
  </div>
</body>