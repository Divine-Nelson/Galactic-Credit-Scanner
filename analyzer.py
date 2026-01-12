# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – CREDIT ANALYZER (FINAL)
Robust against:
- ID churn
- Stationary cards
- Jitter
- Short videos
- Conveyor stalls

Author: Divine Ezeilo
"""

import cv2  # type: ignore
import math
from collections import Counter, deque

from features import (
    detect_many_cards,
    detect_chips,
    decode_digits,
    compute_output_board,
    is_fake_card,
    detect_overlap,
)

# ==========================================================
# CONFIG
# ==========================================================
MAX_CARDS_TO_COUNT = 20

EXIT_MARGIN_PX = 40                  # Physical exit threshold
MIN_FRAMES_BEFORE_FORCE_COUNT = 15   # Temporal fallback

VALUE_HISTORY_LEN = 10
FAKE_HISTORY_LEN  = 7

LOCK_PADDING_PX = 15                 # Spatial lock tolerance


# ==========================================================
# CREDIT ANALYZER
# ==========================================================
class CreditAnalyzer:
    def __init__(self, video_mode=False):
        self.video_mode = video_mode

        # --- tracking ---
        self.next_id = 1
        self.tracked_cards = {}      # track_id -> (cx, cy)

        # --- temporal stability ---
        self.fake_histories  = {}    # track_id -> deque(bool)
        self.value_histories = {}    # track_id -> deque(int)
        self.seen_frames     = {}    # track_id -> frame count

        # --- counting ---
        self.counted_cards = set()
        self.final_total   = 0

        # --- spatial locking (prevents double count) ---
        self.locked_regions = []     # list of (x, y, w, h)

    # ------------------------------------------------------
    # TRACKING ID ASSIGNMENT
    # ------------------------------------------------------
    def _assign_tracking_id(self, metrics):
        cx = metrics["bbox_x"] + metrics["bbox_w"] // 2
        cy = metrics["bbox_y"] + metrics["bbox_h"] // 2

        best_id = None
        best_dist = float("inf")

        for cid, (px, py) in self.tracked_cards.items():
            d = math.hypot(cx - px, cy - py)
            if d < best_dist:
                best_dist = d
                best_id = cid

        threshold = max(metrics["bbox_w"], metrics["bbox_h"]) * 0.6

        # --- reuse existing ID ---
        if best_dist < threshold:
            self.tracked_cards[best_id] = (cx, cy)
            return best_id

        # --- create new ID ---
        cid = self.next_id
        self.next_id += 1
        self.tracked_cards[cid] = (cx, cy)
        self.seen_frames[cid] = 0
        return cid

    # ------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------
    def analyze(self, image):
        cards = detect_many_cards(image)
        overlay = image.copy()

        # --------------------------------------------------
        # OVERLAP WARNING (VISUAL ONLY)
        # --------------------------------------------------
        if detect_overlap(cards):
            cv2.putText(
                overlay,
                "OVERLAPPING CARDS",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        # --------------------------------------------------
        # PER-CARD ANALYSIS
        # --------------------------------------------------
        for card in cards:
            metrics = card["metrics"]
            warped  = card["warped"]

            x, y, w, h = (
                metrics["bbox_x"],
                metrics["bbox_y"],
                metrics["bbox_w"],
                metrics["bbox_h"],
            )

            # --------------------------------------------------
            # 🚫 IGNORE LOCKED REGIONS (prevents double count)
            # --------------------------------------------------
            skip = False
            for lx, ly, lw, lh in self.locked_regions:
                if (
                    x > lx - LOCK_PADDING_PX and
                    y > ly - LOCK_PADDING_PX and
                    x + w < lx + lw + LOCK_PADDING_PX and
                    y + h < ly + lh + LOCK_PADDING_PX
                ):
                    skip = True
                    break

            if skip:
                continue

            # --------------------------------------------------
            # ASSIGN TRACKING ID
            # --------------------------------------------------
            track_id = self._assign_tracking_id(metrics)
            metrics["track_id"] = track_id
            self.seen_frames[track_id] = self.seen_frames.get(track_id, 0) + 1

            # --------------------------------------------------
            # CHIP DETECTION
            # --------------------------------------------------
            chips = detect_chips(warped)

            if not chips:
                raw_fake = True
                digits = []
                dom_color = None
                reason = "No chips"
            else:
                dom_color = Counter(c["color"] for c in chips).most_common(1)[0][0]
                digits, invalid_digit = decode_digits(chips)

                raw_fake, reason = is_fake_card(
                    metrics,
                    chips,
                    dom_color,
                    digits,
                    invalid_digit,
                    video_mode=self.video_mode
                )

            # --------------------------------------------------
            # FAKE / REAL STABILIZATION
            # --------------------------------------------------
            fake_hist = self.fake_histories.setdefault(
                track_id, deque(maxlen=FAKE_HISTORY_LEN)
            )
            fake_hist.append(raw_fake)

            is_fake = sum(fake_hist) >= (len(fake_hist) // 2 + 1)

            # --------------------------------------------------
            # VALUE STABILIZATION
            # --------------------------------------------------
            value = 0 if is_fake else compute_output_board(dom_color, digits)

            val_hist = self.value_histories.setdefault(
                track_id, deque(maxlen=VALUE_HISTORY_LEN)
            )
            val_hist.append(value)

            stable_value = (
                max(set(val_hist), key=val_hist.count)
                if val_hist else 0
            )

            # --------------------------------------------------
            # COUNTING LOGIC
            # --------------------------------------------------
            cy = metrics["bbox_y"] + metrics["bbox_h"]
            should_count = False

            # IMAGE MODE → immediate
            if not self.video_mode:
                should_count = True

            # VIDEO MODE → exit
            elif cy > image.shape[0] - EXIT_MARGIN_PX:
                should_count = True

            # VIDEO MODE → temporal fallback
            elif (
                not is_fake and
                self.seen_frames.get(track_id, 0) >= MIN_FRAMES_BEFORE_FORCE_COUNT
            ):
                should_count = True

            if (
                should_count and
                not is_fake and
                track_id not in self.counted_cards and
                len(self.counted_cards) < MAX_CARDS_TO_COUNT
            ):
                self.final_total += stable_value
                self.counted_cards.add(track_id)

                # 🔒 lock region
                self.locked_regions.append((x, y, w, h))

                # cleanup tracking
                self.tracked_cards.pop(track_id, None)
                self.fake_histories.pop(track_id, None)
                self.value_histories.pop(track_id, None)
                self.seen_frames.pop(track_id, None)

            # --------------------------------------------------
            # DRAW CARD
            # --------------------------------------------------
            label = (
                f"ID {track_id}: FAKE ({reason})"
                if is_fake else
                f"ID {track_id}: {stable_value} | {dom_color}"
            )

            color = (0, 0, 255) if is_fake else (0, 255, 0)
            self._draw_card(overlay, metrics, label, color)

        # --------------------------------------------------
        # DRAW TOTAL
        # --------------------------------------------------
        cv2.putText(
            overlay,
            f"Total Value: {self.final_total} ({len(self.counted_cards)}/{MAX_CARDS_TO_COUNT})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

        return {
            "overlay": overlay,
            "total_value": self.final_total,
            "cards_counted": len(self.counted_cards),
        }

    # ------------------------------------------------------
    # DRAW HELPER
    # ------------------------------------------------------
    def _draw_card(self, overlay, metrics, label, color):
        x, y, w, h = (
            metrics["bbox_x"],
            metrics["bbox_y"],
            metrics["bbox_w"],
            metrics["bbox_h"]
        )

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            overlay,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
