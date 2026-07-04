import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean()


def is_bad_frame(frame, min_sharpness=80.0, min_brightness=40.0, max_brightness=220.0):
    sharp = sharpness_score(frame)
    bright = brightness_score(frame)

    if sharp < min_sharpness:
        return True, f"BLUR sharp={sharp:.1f}", sharp, bright

    if bright < min_brightness:
        return True, f"TOO_DARK bright={bright:.1f}", sharp, bright

    if bright > max_brightness:
        return True, f"TOO_BRIGHT bright={bright:.1f}", sharp, bright

    return False, "OK", sharp, bright


def open_log(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "class_name",
            "file_path",
            "width",
            "height",
            "target_fps",
            "measured_fps",
            "sharpness",
            "brightness",
            "status"
        ])

    return f, writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--classes", nargs="+", default=["helmet", "no_helmet"])
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--auto-interval", type=float, default=0.5)
    parser.add_argument("--min-sharpness", type=float, default=80.0)
    args = parser.parse_args()

    base_dir = Path("data/raw")
    log_path = Path("logs/capture_log.csv")

    for class_name in args.classes:
        (base_dir / class_name).mkdir(parents=True, exist_ok=True)

    counts = {
        class_name: len(list((base_dir / class_name).glob("*.jpg")))
        for class_name in args.classes
    }

    selected_idx = 0
    selected_class = args.classes[selected_idx]
    auto_mode = False
    last_auto_save = 0.0

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Camera open failed. 현재 --camera {args.camera}로 열 수 없습니다. "
            f"ls /dev/video* 결과를 보고 camera 번호를 바꿔보세요."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("=== Camera Setting ===")
    print(f"camera index: {args.camera}")
    print(f"requested: {args.width}x{args.height} @ {args.fps}fps")
    print(f"actual:    {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
    print()
    print("=== Key Guide ===")
    print("1, 2, 3... : 클래스 선택")
    print("SPACE      : 현재 클래스에 1장 저장")
    print("a          : 자동 저장 ON/OFF")
    print("q          : 종료")
    print()

    log_file, log_writer = open_log(log_path)

    prev_time = time.time()
    measured_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("frame read failed")
                break

            now = time.time()
            dt = now - prev_time
            prev_time = now

            if dt > 0:
                measured_fps = 1.0 / dt

            bad, status, sharp, bright = is_bad_frame(
                frame,
                min_sharpness=args.min_sharpness
            )

            display = frame.copy()

            info_1 = f"class: {selected_class} | count: {counts[selected_class]}/{args.target}"
            info_2 = f"FPS: {measured_fps:.1f} | {actual_w}x{actual_h} | auto: {auto_mode}"
            info_3 = f"quality: {status} | sharp={sharp:.1f} bright={bright:.1f}"

            cv2.putText(display, info_1, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, info_2, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, info_3, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            y = 140
            for i, class_name in enumerate(args.classes):
                line = f"{i + 1}: {class_name} = {counts[class_name]}"
                cv2.putText(display, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 30

            cv2.imshow("Dataset Capture Tool", display)

            key = cv2.waitKey(1) & 0xFF

            should_save = False

            if key == ord("q"):
                break

            elif key == ord("a"):
                auto_mode = not auto_mode
                print("auto_mode:", auto_mode)

            elif key == 32:
                should_save = True

            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(args.classes):
                    selected_idx = idx
                    selected_class = args.classes[selected_idx]
                    print("selected class:", selected_class)

            if auto_mode and (now - last_auto_save >= args.auto_interval):
                should_save = True
                last_auto_save = now

            if should_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                count = counts[selected_class]
                filename = f"{selected_class}_{timestamp}_{count:04d}.jpg"
                save_path = base_dir / selected_class / filename

                if bad:
                    print(f"SKIP: {status}")
                    log_writer.writerow([
                        datetime.now().isoformat(),
                        selected_class,
                        "",
                        actual_w,
                        actual_h,
                        args.fps,
                        f"{measured_fps:.2f}",
                        f"{sharp:.2f}",
                        f"{bright:.2f}",
                        status
                    ])
                    log_file.flush()
                    continue

                cv2.imwrite(str(save_path), frame)
                counts[selected_class] += 1

                print(f"saved: {save_path}")

                log_writer.writerow([
                    datetime.now().isoformat(),
                    selected_class,
                    str(save_path),
                    actual_w,
                    actual_h,
                    args.fps,
                    f"{measured_fps:.2f}",
                    f"{sharp:.2f}",
                    f"{bright:.2f}",
                    "SAVED"
                ])
                log_file.flush()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()

        print()
        print("=== Summary ===")
        for class_name in args.classes:
            print(f"{class_name}: {counts[class_name]} images")

        print(f"log saved: {log_path}")


if __name__ == "__main__":
    main()
