from pathlib import Path
import cv2


def inspect_class(class_dir):
    image_paths = sorted(class_dir.glob("*.jpg"))

    if not image_paths:
        return {
            "class": class_dir.name,
            "count": 0,
            "bad_read": 0,
            "sizes": []
        }

    bad_read = 0
    sizes = []

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            bad_read += 1
            continue

        h, w = img.shape[:2]
        sizes.append((w, h))

    return {
        "class": class_dir.name,
        "count": len(image_paths),
        "bad_read": bad_read,
        "sizes": sizes
    }


def main():
    raw_dir = Path("data/raw")

    if not raw_dir.exists():
        print("ERROR: data/raw 폴더가 없습니다.")
        print("현재 위치가 ~/edge-ai-day2인지 확인하세요.")
        return

    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    if not class_dirs:
        print("ERROR: data/raw 안에 클래스 폴더가 없습니다.")
        return

    print("=== Dataset Inspection ===")

    total = 0

    for class_dir in class_dirs:
        result = inspect_class(class_dir)
        total += result["count"]

        print()
        print(f"class: {result['class']}")
        print(f"count: {result['count']}")
        print(f"bad_read: {result['bad_read']}")

        if result["sizes"]:
            unique_sizes = sorted(set(result["sizes"]))
            print(f"image sizes: {unique_sizes}")
        else:
            print("image sizes: 없음")

        if result["count"] >= 150 and result["bad_read"] == 0:
            print("status: OK")
        elif result["count"] < 150:
            print("status: 부족함. 150장 이상 필요")
        elif result["bad_read"] > 0:
            print("status: 읽기 실패 이미지가 있음")

    print()
    print(f"total images: {total}")


if __name__ == "__main__":
    main()
