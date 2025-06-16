import socket
import numpy as np

def generate_close_points():
    cam = np.random.uniform(0, 10, 3)
    holo = cam + np.random.normal(0, 0.5, 3)
    return cam, holo

def main():
    HOST = 'localhost'
    PORT = 50007

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        print(f"Connected to GUI at {HOST}:{PORT}")
        print("Press [Enter] to send a new point pair. Press Ctrl+C to exit.")
    except ConnectionRefusedError:
        print("Could not connect to the GUI. Make sure it's running.")
        return

    try:
        while True:
            input(">> ")
            cam, holo = generate_close_points()
            line = ' '.join(map(str, np.concatenate((cam, holo)))) + '\n'
            sock.sendall(line.encode('utf-8'))
            print(f"Sent cam: {cam.round(2)}")
            print(f"Sent holo: {holo.round(2)}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()