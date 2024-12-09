import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import os
import torch
import glob

class InteractiveMaskDetector:
    def __init__(self, image_path, predictor):
        self.image_path = image_path
        self.predictor = predictor
        
        self.image = cv2.imread(image_path)
        self.draw_image = self.image.copy()
        self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(self.rgb_image)
        
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.cur_x, self.cur_y = -1, -1
        
        window_name = 'Interactive Mask Detector'
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 2000, 400)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.combined_mask = np.zeros(self.image.shape[:2], dtype=bool)
        self.mask_count = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.draw_image = self.image.copy()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_image = self.image.copy()
                cv2.rectangle(self.draw_image, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, x2 = min(self.start_x, x), max(self.start_x, x)
            y1, y2 = min(self.start_y, y), max(self.start_y, y)
            w, h = x2 - x1, y2 - y1
            
            if w > 1 and h > 1:
                self.detect_and_save_mask(x1, y1, w, h)
    
    def detect_and_save_mask(self, x, y, w, h):
        input_point = np.array([[x + w//2, y + h//2]])
        input_label = np.array([1])
        input_box = np.array([x, y, x + w, y + h])
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=True
        )
        
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        score = scores[best_mask_idx]
        
        roi_mask = np.zeros_like(mask)
        roi_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
        
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded_mask = cv2.dilate(roi_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        self.combined_mask = self.combined_mask | expanded_mask
        self.mask_count += 1
        
        colored_mask = np.zeros_like(self.image)
        colored_mask[self.combined_mask] = [0, 0, 255]
        self.image = cv2.addWeighted(self.image, 1, colored_mask, 0.5, 0)
        self.draw_image = self.image.copy()
        
        self.save_results()
    
    def save_results(self):
        mask_image = (self.combined_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join('../data/test_mask', os.path.basename(self.image_path)), mask_image)
        
    
    def run(self):
        print("Processing {}".format(self.image_path))
        
        while True:
            cv2.imshow('Interactive Mask Detector', self.draw_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.save_results()
                break
            elif key == ord('r'):
                self.image = cv2.imread(self.image_path)
                self.draw_image = self.image.copy()
                self.combined_mask = np.zeros(self.image.shape[:2], dtype=bool)
                self.mask_count = 0
        
        cv2.destroyAllWindows()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    for i, image_path in enumerate(sorted(glob.glob("../data/test_input/*.png"))):
        detector = InteractiveMaskDetector(image_path, predictor)
        detector.run()

if __name__ == "__main__":
    main()
