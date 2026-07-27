from PIL import Image

def get_vision_token_cost(img_path, max_dim=768):
    """Calculates the exact token budget for Qwen-VL architecture."""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            
            # 1. Simulate the safe_img.thumbnail((768, 768)) applied in app.py
            if width > max_dim or height > max_dim:
                ratio = min(max_dim / width, max_dim / height)
                width = int(width * ratio)
                height = int(height * ratio)
            
            # 2. Qwen divides images into 28x28 pixel patches.
            # It rounds the dimensions to the nearest multiple of 28.
            h_blocks = max(1, round(height / 28))
            w_blocks = max(1, round(width / 28))
            
            # 3. Total tokens is simply the grid of blocks
            return h_blocks * w_blocks
            
    except Exception:
        # Fallback to the absolute max tokens a 768x768 image could cost (27x27 blocks)
        return 729
'''
def get_vision_token_cost(img_path):
    """Calculates the token budget based on Gemma 4's resolution tiers."""
    try:
        # We only need the headers, so this is virtually instant and costs no RAM
        with Image.open(img_path) as img:
            width, height = img.size
            total_pixels = width * height
            
            # Match pixel count to the closest token allocation tier
            if total_pixels <= 50176:      # ~224x224
                return 70
            elif total_pixels <= 100352:   # ~316x316
                return 140
            elif total_pixels <= 200704:   # ~448x448
                return 280
            elif total_pixels <= 401408:   # ~632x632
                return 560
            else:                          # High-res ceiling (e.g., 768x768)
                return 1120
    except Exception:
        return 560 # Safe fallback if the image is corrupted'''