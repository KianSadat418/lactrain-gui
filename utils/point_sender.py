import socket
import numpy as np
import json

def generate_close_points():
    cam = np.random.uniform(0, 10, 3)
    holo = cam + np.random.normal(0, 0.5, 3)
    return cam.tolist(), holo.tolist()

def main():
    HOST = 'localhost'
    PORT = 50007

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((HOST, PORT))
        print(f"Connected to GUI at {HOST}:{PORT}")
        print("Press [Enter] to send a new point pair. Press Ctrl+C to exit.")
        fileobj = sock.makefile('w')
    except ConnectionRefusedError:
        print("Could not connect to the GUI. Make sure it's running.")
        return

    try:
        n = 0
        payload_stored = {}
        while True:
            n += 1
            input(">> ")
            cam, holo = generate_close_points()
            payload_stored[n] = [cam, holo]
            payload = json.dumps(payload_stored)
            fileobj.write(payload + "\n")
            fileobj.flush()
            print(f"Sent cam: {np.round(cam, 2)}")
            print(f"Sent holo: {np.round(holo, 2)}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
