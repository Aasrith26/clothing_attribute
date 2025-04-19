import cv2
import matplotlib.pyplot as plt

def generate_region_proposals(image_path, scale=130, sigma=0.8, min_size=380):
    """
    Uses OpenCV's Selective Search to generate region proposals.
    """
    if not hasattr(cv2, 'ximgproc'):
        raise RuntimeError("OpenCV ximgproc not found. Install opencv-contrib-python.")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_bgr)
    ss.switchToSelectiveSearchFast(base_k=scale, inc_k=min_size, sigma=sigma)  # Fast mode
    rects = ss.process()  # List of (x, y, w, h)

    return image_bgr, rects

def visualize_proposals(image_path, num_proposals=100):
    """
    Generates and visualizes region proposals from Selective Search.
    """
    image_bgr, proposals = generate_region_proposals(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert for matplotlib

    # Draw first N proposals
    for (x, y, w, h) in proposals[:num_proposals]:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the image with proposals
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(f"Top {num_proposals} Selective Search Proposals")
    plt.show()

# Example usage
image_path = "dataset2/dress_0013.jpg"  # Change to your image path
visualize_proposals(image_path, num_proposals=50)
