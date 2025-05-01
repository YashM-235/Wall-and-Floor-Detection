import cv2
import numpy as np

# Load brick and floor tile textures
wall_texture = cv2.imread('textures/wall.jpeg')
floor_texture = cv2.imread('textures/floor.jpeg')

def apply_texture(mask, texture):
    """Overlay texture onto the mask."""
    texture = cv2.resize(texture, (mask.shape[1], mask.shape[0]))
    texture_masked = cv2.bitwise_and(texture, texture, mask=mask)
    return texture_masked

def overlay_bricks(original_frame, wall_mask, floor_mask):
    """Place realistic textures based on wall and floor masks."""
    wall_mask = cv2.cvtColor(wall_mask, cv2.COLOR_BGR2GRAY)
    floor_mask = cv2.cvtColor(floor_mask, cv2.COLOR_BGR2GRAY)

    _, wall_bin = cv2.threshold(wall_mask, 127, 255, cv2.THRESH_BINARY)
    _, floor_bin = cv2.threshold(floor_mask, 127, 255, cv2.THRESH_BINARY)

    wall_tex = apply_texture(wall_bin, wall_texture)
    floor_tex = apply_texture(floor_bin, floor_texture)

    combined = cv2.addWeighted(original_frame, 0.5, wall_tex, 0.5, 0)
    combined = cv2.addWeighted(combined, 0.8, floor_tex, 0.2, 0)

    return combined
