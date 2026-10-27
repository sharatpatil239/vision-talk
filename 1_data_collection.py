"""
1_data_collection.py - Reliable Webcam Data Collection Utility for VisionTalk

Features:
- Configurable class labels, image count, capture interval, camera index, and output directory.
- Verifies webcam hardware accessibility and frame integrity before capturing.
- Live non-blocking preview with animated countdown, capture counter, and interval timer.
- Instant responsive abort via 'q' or ESC (27) at any moment.
- Safe resource cleanup guaranteed via try...finally block.
- Keeps raw captures isolated in 'workspace/images/collectedimages/<class>/' for review.
- Detailed stdout logging of saved files, counts, and directory paths.
"""

import os
import sys
import time
import uuid
import argparse
import cv2


def draw_hud(frame, title, subtitle, counter_text=None, alert_text=None, flash=False):
    """Draws an informative HUD overlay on the preview frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Semi-transparent top and bottom banners
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    
    # Border flash effect on capture
    if flash:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), 10)
        
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Top banner text
    cv2.putText(frame, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Counter or secondary status
    if counter_text:
        (tw, _), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, counter_text, (w - tw - 25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Alert text in center of screen
    if alert_text:
        (tw, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        cx = (w - tw) // 2
        cy = h // 2
        cv2.putText(frame, alert_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, alert_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3, cv2.LINE_AA)
        
    # Bottom banner footer
    footer = "Press 'q' or 'ESC' to Stop Collection"
    cv2.putText(frame, footer, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def collect_images(labels, num_images=15, interval=2.0, camera_id=0,
                   output_dir=os.path.join("workspace", "images", "collectedimages"),
                   countdown_sec=5.0):
    """Executes the automated image collection workflow."""
    print("================================================================================")
    print("                VISIONTALK RELIABLE WEBCAM DATA COLLECTION                      ")
    print("================================================================================")
    print(f"Target Labels:     {labels}")
    print(f"Images per Label:  {num_images}")
    print(f"Capture Interval:  {interval:.1f} seconds")
    print(f"Camera Device ID:  {camera_id}")
    print(f"Output Directory:  {os.path.abspath(output_dir)}")
    print(f"Pre-roll Delay:    {countdown_sec:.1f} seconds")
    print("--------------------------------------------------------------------------------")

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize and verify webcam
    print(f"Connecting to video device {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera device {camera_id}.")
        print("Please check that your webcam is connected, not in use by another application,")
        print("and that camera permissions are granted.")
        return False, {}

    # Verify frame read capability
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"\n[ERROR] Camera device {camera_id} opened, but failed to grab test frame.")
        cap.release()
        return False, {}
        
    frame_h, frame_w = test_frame.shape[:2]
    print(f"Webcam initialized successfully! Stream resolution: {frame_w}x{frame_h}")
    print("Launching collection window. Focus window and press 'q' or 'ESC' to abort at any time.\n")

    saved_records = {}
    collection_aborted = False

    try:
        for label_idx, label in enumerate(labels, start=1):
            if collection_aborted:
                break
                
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            saved_records[label] = []

            print(f"\n>>> [{label_idx}/{len(labels)}] PREPARING COLLECTION FOR CLASS: '{label}' <<<")
            print(f"Target folder: {label_dir}")
            
            # Phase A: Countdown phase before capturing this class
            countdown_start = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\n[ERROR] Lost video frame during countdown.")
                    collection_aborted = True
                    break
                    
                elapsed = time.time() - countdown_start
                remaining = countdown_sec - elapsed
                
                if remaining <= 0:
                    break
                    
                draw_hud(
                    frame=frame,
                    title=f"PREPARE SIGN FOR: [{label.upper()}] ({label_idx}/{len(labels)})",
                    subtitle="Position hands in camera view. Capturing starts shortly.",
                    alert_text=f"Starting in: {int(remaining) + 1}s"
                )
                
                cv2.imshow("VisionTalk Data Collector", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    print("\nCollection aborted by user during countdown.")
                    collection_aborted = True
                    break
                    
            if collection_aborted:
                break
                
            # Phase B: Capture loop for specified number of images
            print(f"Capturing {num_images} images for '{label}' every {interval:.1f}s...")
            captured_count = 0
            last_capture_time = time.time()
            flash_until = 0

            while captured_count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("\n[ERROR] Failed to read frame from camera.")
                    collection_aborted = True
                    break
                    
                now = time.time()
                time_since_last = now - last_capture_time
                time_to_next = max(0.0, interval - time_since_last)
                
                # Capture frame when interval elapses
                if time_since_last >= interval:
                    clean_frame = frame.copy()
                    img_filename = f"{label}.{str(uuid.uuid1())}.jpg"
                    img_filepath = os.path.join(label_dir, img_filename)
                    
                    success = cv2.imwrite(img_filepath, clean_frame)
                    if success:
                        captured_count += 1
                        saved_records[label].append(img_filepath)
                        last_capture_time = now
                        flash_until = now + 0.15  # 150ms green flash
                        print(f"  [SAVED {captured_count:>2}/{num_images}] {img_filename}")
                    else:
                        print(f"  [ERROR] Failed to write image to: {img_filepath}")
                        
                    if captured_count >= num_images:
                        break

                # Draw UI for active capture
                is_flashing = now < flash_until
                draw_hud(
                    frame=frame,
                    title=f"COLLECTING CLASS: [{label.upper()}]",
                    subtitle=f"Next capture in: {time_to_next:.1f}s (Change hand angle / distance)",
                    counter_text=f"{captured_count}/{num_images}",
                    flash=is_flashing
                )
                
                cv2.imshow("VisionTalk Data Collector", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    print(f"\nCollection aborted by user. Saved {captured_count} images for '{label}'.")
                    collection_aborted = True
                    break

    finally:
        print("\nCleaning up camera and display resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and display windows closed.")

    # 3. Print final report
    print("\n================================================================================")
    print("                        COLLECTION SUMMARY REPORT                               ")
    print("================================================================================")
    total_saved = sum(len(files) for files in saved_records.values())
    for lbl, files in saved_records.items():
        print(f"Class: '{lbl:<12}' -> Successfully Saved: {len(files):>3} images ({output_dir}\\{lbl})")
    print(f"\nTotal New Images Captured: {total_saved}")
    print("Status: " + ("ABORTED EARLY BY USER" if collection_aborted else "COMPLETED SUCCESSFULLY"))
    print("NOTE: Captured images are kept in 'collectedimages/' separate from training data.")
    print("      Review and annotate them in LabelImg before splitting into train/test.")
    print("================================================================================")
    
    return not collection_aborted, saved_records


def main():
    parser = argparse.ArgumentParser(description="Reliable webcam data collector for VisionTalk sign language gestures.")
    parser.add_argument("--labels", nargs="+", default=["hello", "thankyou", "yes", "no", "iloveyou"],
                        help="Gesture class names to collect (e.g. --labels thankyou or --labels hello yes no).")
    parser.add_argument("--num_images", type=int, default=15,
                        help="Number of images to capture per class (default: 15).")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Interval in seconds between consecutive captures (default: 2.0).")
    parser.add_argument("--camera_id", type=int, default=0,
                        help="Camera device index (default: 0).")
    parser.add_argument("--output_dir", default=os.path.join("workspace", "images", "collectedimages"),
                        help="Directory to save raw collected images (default: workspace/images/collectedimages).")
    parser.add_argument("--countdown", type=float, default=5.0,
                        help="Countdown delay in seconds before each class begins capturing (default: 5.0).")
                        
    args = parser.parse_args()
    
    success, _ = collect_images(
        labels=args.labels,
        num_images=args.num_images,
        interval=args.interval,
        camera_id=args.camera_id,
        output_dir=args.output_dir,
        countdown_sec=args.countdown
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()