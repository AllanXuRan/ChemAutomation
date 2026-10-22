"""Optional DIGIT camera smoke test; run directly with camera hardware attached."""

def main():
    import cv2

    # Camera index: try 0, 1, 2, 3...
    cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Camera not found. Try index 1 or 2.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera connected. Press S to save, Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow("DIGIT Tactile Sensor", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            cv2.imwrite("digit_test.png", frame)
            print("Image saved!")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
