# Q&A Chat Bot for Labor Code 
Чат бот для консультирования по Трудовому Кодексу РФ
![Иллюстрация к проекту](https://github.com/DeGrimer/RetriLLM/raw/main/image.png)
## Технологии
 - LangChain
 - FastApi
 - React
 - PostgreSQL

## Использование
```sh
docker-compose -f docker-compose.db.yml up -d
```

```sh
pip install -r requirements.txt
```

Запуск сервера
```sh
-m uvicorn main:app --host "localhost" --port 80 --reload
```

Запуск react приложения
```sh
./qachatbot npm start
```

