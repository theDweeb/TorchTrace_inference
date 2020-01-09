import socket
import sys

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 8080        # The port used by the server

cmd = sys.argv[1]
cmd += '\n'


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(cmd.encode())
    data = s.recv(8080)

msg = data.decode()

print(msg)