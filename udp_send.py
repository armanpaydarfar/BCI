import socket
import config
# Server address and port (update as needed)
server_address = config.UDP_MARKER["IP"]  # Change to the IP of the FES script's machine
server_port = config.UDP_MARKER["PORT"]            # Match the port used in the FES script
server_address_port = (server_address, server_port)

buffer_size = 1024

# Create a UDP socket on the client side
udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

print(f"Sending UDP commands to {server_address}:{server_port}")
while True:
    val = input("Enter message: ")

    # Exit the program
    if val.lower() == "e":
        print("Ending program, bye...")
        break

    print("Message to send: ", val)
    udp_client_socket.sendto(val.encode(), server_address_port)
