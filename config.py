from pathlib import Path

IMG_SIZE = 1024  # Set the image size for inference, the number of pixels the longest edge should be resized to
IMG_DIR: Path = Path("watch_folder")  # Set the path to the 'watch_folder' directory
COMPARE_FACES_TOLERANCE = 0.55  # the lower the tolerance, the more sensitive the algorithm is at matching faces
AUTO_CONFIRM_MATCHES_TIME = 1  # Number of seconds to wait before submitting predicted name
IMG_PREVIEW_WIDTH = 1500
