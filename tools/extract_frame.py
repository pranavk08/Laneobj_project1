import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video path or camera index")
    parser.add_argument("--output", required=True, help="Output image path")
    args = parser.parse_args()

    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit("Failed to grab a frame from source")
    cv2.imwrite(args.output, frame)


if __name__ == "__main__":
    main()
