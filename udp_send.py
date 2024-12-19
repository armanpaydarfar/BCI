import socket

serverAddressPort = ("192.168.2.1", 8080)

bufferSize = 1024

# Create a UDP socket at client side
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Send to server using created UDP socket
print(serverAddressPort[0])
while True:
    val = input("Enter message: ")

    # to exit the program
    if val == "e":
        print("ending program, bye...")
        break
    print("Message to send: ", str(val))
    UDPClientSocket.sendto(str.encode(val), serverAddressPort)
