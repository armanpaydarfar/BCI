# Networking.py

import socket

def send_udp_message(socket, ip, port, message):
    """
    Send a UDP message to the specified IP and port.

    Parameters:
        socket (socket.socket): The socket object for communication.
        ip (str): The target IP address.
        port (int): The target port.
        message (str): The message to send.
    """
    socket.sendto(message.encode('utf-8'), (ip, port))
    print(f"Sent UDP message to {ip}:{port}: {message}")
