import socket

import constants

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect(("localhost", constants.PORT))

while True:
    msg = sock.recv(1024)
    msgs = msg.split(b'|')
    nums = msgs[1].split(b',')
    for num in nums:
        print(num.decode('utf-8'), end=", ")
    print()
