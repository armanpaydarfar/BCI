"""
Pupil Labs Neon - Gaze Stream for Robot Control
Streams gaze X, Y coordinates and confidence at configurable rate
"""

from pylsl import StreamInlet, resolve_stream
import time

class NeonGazeStream:
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize Neon gaze stream receiver
        
        Args:
            confidence_threshold: Minimum confidence to accept gaze data (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.inlet = None
        self.stream_info = None
        self.scene_width = 1600  # Neon scene camera resolution
        self.scene_height = 1200
        
    def connect(self, timeout=10):
        """Find and connect to Neon gaze stream"""
        print("Searching for Neon gaze stream...")
        streams = resolve_stream('type', 'Gaze')
        
        if not streams:
            print("ERROR: No Neon gaze stream found!")
            print("Ensure LSL streaming is enabled in Companion app")
            return False
        
        self.stream_info = streams[0]
        self.inlet = StreamInlet(self.stream_info)
        
        print(f"✓ Connected to: {self.stream_info.name()}")
        print(f"  Sample rate: {self.stream_info.nominal_srate()} Hz")
        print(f"  Scene camera: {self.scene_width}x{self.scene_height} px")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        return True
    
    def get_gaze(self, timeout=1.0):
        """
        Get next gaze sample
        
        Returns:
            dict with keys: x, y, confidence, timestamp
            or None if no valid sample
        """
        if not self.inlet:
            return None
        
        sample, timestamp = self.inlet.pull_sample(timeout=timeout)
        
        if not sample:
            return None
        
        x = sample[0]
        y = sample[1]
        confidence = sample[15]
        
        # Filter low confidence samples
        if confidence < self.confidence_threshold:
            return None
        
        return {
            'x': x,
            'y': y,
            'confidence': confidence,
            'timestamp': timestamp,
            'x_normalized': x / self.scene_width,  # 0-1 range
            'y_normalized': y / self.scene_height  # 0-1 range
        }
    
    def get_gaze_normalized(self, timeout=1.0):
        """
        Get gaze coordinates normalized to 0-1 range
        
        Returns:
            tuple (x_norm, y_norm, confidence) or None
        """
        gaze = self.get_gaze(timeout)
        if gaze:
            return (gaze['x_normalized'], gaze['y_normalized'], gaze['confidence'])
        return None

def main():
    """Example usage"""
    # Initialize stream
    gaze_stream = NeonGazeStream(confidence_threshold=0.7)
    
    if not gaze_stream.connect():
        return
    
    print("\nStreaming normalized gaze data (Ctrl+C to stop)...")
    print("Format: Normalized X (0-1) | Normalized Y (0-1) | Confidence\n")
    
    sample_count = 0
    valid_count = 0
    
    try:
        while True:
            gaze = gaze_stream.get_gaze(timeout=1.0)
            
            if gaze:
                valid_count += 1
                x_norm = gaze['x_normalized']
                y_norm = gaze['y_normalized']
                conf = gaze['confidence']
                
                print(f"X: {x_norm:.4f} | Y: {y_norm:.4f} | Confidence: {conf:.3f}")
            
            sample_count += 1
            time.sleep(0.01)  # Small delay for readability
            
    except KeyboardInterrupt:
        print(f"\n\nStopped by user")
        print(f"Total samples attempted: {sample_count}")
        print(f"Valid samples (>= {gaze_stream.confidence_threshold} confidence): {valid_count}")
        if sample_count > 0:
            print(f"Valid sample rate: {100*valid_count/sample_count:.1f}%")

if __name__ == "__main__":
    main()