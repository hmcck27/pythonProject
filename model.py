import socket

## using asyncio for multi preprocessing
## 동시에 여러 프로그램을 돌려야 하니까.
## app.py는 nginx와 uwsgi를 통해서 병렬 처리가 가능한 상태.

import asyncio
import aiohttp


HOST = '127.0.0.1'
PORT = 65432

def preprocess() :
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                print('connected by', addr)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    conn.sendall(data)

if __name__ == "__main__" :
    asyncio.get_event_loop().run_until_complete(preprocess())