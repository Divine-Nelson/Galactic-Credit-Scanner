import cv2  # type: ignore
from collections import Counter, deque

from features import (
    detect_many_cards,
    detect_chips,
    decode_digits,
    compute_output_board,
    is_fake_card
)

class CreditAnalyzer:
    def __init__(self, video_mode=False):
        self.video_mode = video_mode
        self.fake_histories = {}
        self.total_history = deque(maxlen=5)

    def analyze(self, image):
        cards = detect_many_cards(image)
        overlay = image.copy()

        results = []
        frame_total = 0

        for card in cards:
            warped = card["warped"]
            m = card["metrics"]

            chips = detect_chips(warped)
            if not chips:
                continue

            dom_color = Counter(c["color"] for c in chips).most_common(1)[0][0]
            digits, invalid = decode_digits(chips)

            raw_fake, reason = is_fake_card(
                m, chips, dom_color, digits, invalid
            )

            key = m["bbox_x"] // 50
            is_fake = raw_fake

            if self.video_mode:
                hist = self.fake_histories.setdefault(key, deque(maxlen=7))
                hist.append(raw_fake)
                is_fake = sum(hist) >= (len(hist) // 2 + 1)

            value = 0 if is_fake else compute_output_board(dom_color, digits)
            frame_total += value

            color = (0, 0, 255) if is_fake else (0, 255, 0)
            label = (
                f"FAKE ({reason})"
                if is_fake
                else f"Value: {value} | {dom_color}"
            )

            x, y, w, h = m["bbox_x"], m["bbox_y"], m["bbox_w"], m["bbox_h"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                overlay, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

            results.append({
                "digits": digits,
                "color": dom_color,
                "value": value,
                "fake": is_fake
            })

        if self.video_mode:
            self.total_history.append(frame_total)
            frame_total = round(sum(self.total_history) / len(self.total_history))

        cv2.putText(
            overlay, f"Total Value: {frame_total}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 0), 2
        )

        return {
            "cards": results,
            "overlay": overlay,
            "total_value": frame_total
        }


def analyze_image(image, video_mode=False):
    return CreditAnalyzer(video_mode).analyze(image)
