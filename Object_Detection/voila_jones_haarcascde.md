OFFLINE (once):
Training Images
    ↓
Haar Feature Generation
    ↓
Integral Image
    ↓
AdaBoost Training
    ↓
Cascade Construction
    ↓
XML FILE


RUNTIME (every image):
Load XML
    ↓
Grayscale Image
    ↓
Integral Image
    ↓
Sliding Windows + Cascade Evaluation
    ↓
Final Detections ✅