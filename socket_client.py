import argparse
import logging
import socket

import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--localhost", help="Listen to localhost:5800. Is the default", action="store_const", const="localhost")
parser.add_argument("--talon", help="Listen to team2502-tinker.local:5800", action="store_const", const="team2502-tinker.local")
parser.add_argument("--uri", help="Listen to some other address", action="store", type=str)

parser.add_argument("--port", help="Port to listen on", action="store", type=int, default=constants.PORT)

args = parser.parse_args()

uri = str(args.localhost or args.talon or args.uri or "localhost")

logger.debug("uri = " + uri)
logger.debug("port = " + str(args.port))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect((uri, args.port))

while True:
    msg = sock.recv(1024)
    msgs = msg.split(b'\n')
    nums = msgs[0].split(b',')
    for num in nums:
        print(num.decode('utf-8'), end=", ")
    print()
