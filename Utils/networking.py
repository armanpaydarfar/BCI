import socket

def send_udp_message(sock, ip, port, message, logger=None):
    """
    Send a UDP message to the specified IP and port.

    Parameters:
        sock (socket.socket): The socket object for communication.
        ip (str): The target IP address.
        port (int): The target port.
        message (str): The message to send.
        logger (optional): Logger instance to log the message. If None, defaults to print.
    """
    sock.sendto(message.encode('utf-8'), (ip, port))
    
    msg_str = f"Sent UDP message to {ip}:{port}: {message}"
    if logger is not None:
        logger.log_event(msg_str)
    else:
        print(msg_str)
