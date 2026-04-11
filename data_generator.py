"""Generate realistic synthetic retinal fundus images with lipid correlations."""

import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import config

def create_realistic_vessels(img, disc_center, disc_radius, lipid_values):
    """Create realistic blood vessel network with proper branching."""
    # Vessel parameters based on lipid values
    # High cholesterol -> thicker vessels, more tortuosity
    # High LDL -> more vessel narrowing and irregularities
    base_thickness = 2 + (lipid_values['total_cholesterol'] - 120) / 180 * 3
    vessel_density = 0.7 + (lipid_values['ldl'] - 50) / 150 * 0.3
    tortuosity = 0.05 + (lipid_values['total_cholesterol'] - 120) / 180 * 0.15
    
    # Major vessels (4 main arteries and 4 main veins)
    num_major_vessels = 8
    
    for vessel_idx in range(num_major_vessels):
        # Start from optic disc
        angle = vessel_idx * (2 * np.pi / num_major_vessels) + np.random.uniform(-0.2, 0.2)
        
        # Arteries (thinner, brighter) vs Veins (thicker, darker)
        is_artery = vessel_idx % 2 == 0
        
        if is_artery:
            thickness = base_thickness * 0.7
            color = (95, 45, 35)  # Brighter red
        else:
            thickness = base_thickness * 1.0
            color = (70, 30, 25)  # Darker red
        
        # Create main vessel path
        current_pos = np.array([
            disc_center[0] + disc_radius * np.cos(angle),
            disc_center[1] + disc_radius * np.sin(angle)
        ])
        
        current_angle = angle
        current_thickness = thickness
        
        # Draw main vessel with gradual thinning
        for step in range(60):
            # Add tortuosity (vessel curvature)
            current_angle += np.random.uniform(-tortuosity, tortuosity)
            
            # Move along vessel
            step_length = np.random.uniform(2, 4)
            next_pos = current_pos + step_length * np.array([np.cos(current_angle), np.sin(current_angle)])
            
            # Check bounds
            if not (10 < next_pos[0] < config.IMAGE_SIZE - 10 and 10 < next_pos[1] < config.IMAGE_SIZE - 10):
                break
            
            # Gradually thin the vessel
            current_thickness = max(1, thickness * (1 - step / 80))
            
            # Draw vessel segment
            cv2.line(img, 
                    tuple(current_pos.astype(int)), 
                    tuple(next_pos.astype(int)),
                    color, 
                    max(1, int(current_thickness)))
            
            # Add vessel branching (bifurcations)
            if step > 15 and step % 20 == 0 and np.random.random() < vessel_density:
                branch_angle = current_angle + np.random.choice([-0.5, 0.5])
                draw_vessel_branch(img, next_pos, branch_angle, current_thickness * 0.6, color, 25)
            
            current_pos = next_pos
    
    return img

def draw_vessel_branch(img, start_pos, angle, thickness, color, max_steps):
    """Draw a vessel branch."""
    current_pos = start_pos.copy()
    current_angle = angle
    
    for step in range(max_steps):
        current_angle += np.random.uniform(-0.08, 0.08)
        step_length = np.random.uniform(2, 3)
        next_pos = current_pos + step_length * np.array([np.cos(current_angle), np.sin(current_angle)])
        
        if not (5 < next_pos[0] < config.IMAGE_SIZE - 5 and 5 < next_pos[1] < config.IMAGE_SIZE - 5):
            break
        
        thickness = max(1, thickness * (1 - step / (max_steps * 1.5)))
        
        cv2.line(img, 
                tuple(current_pos.astype(int)), 
                tuple(next_pos.astype(int)),
                color, 
                max(1, int(thickness)))
        
        current_pos = next_pos

def add_exudates_and_deposits(img, lipid_values):
    """Add hard exudates and lipid deposits (yellow-white spots) based on lipid levels."""
    # High triglycerides and cholesterol -> more exudates
    num_exudates = int((lipid_values['triglycerides'] - 50) / 200 * 15 + 
                       (lipid_values['total_cholesterol'] - 120) / 180 * 10)
    
    if num_exudates > 0:
        for _ in range(num_exudates):
            x = np.random.randint(30, config.IMAGE_SIZE - 30)
            y = np.random.randint(30, config.IMAGE_SIZE - 30)
            radius = np.random.randint(2, 5)
            
            # Yellow-white deposits
            color = (180 + np.random.randint(0, 40), 
                    170 + np.random.randint(0, 40), 
                    100 + np.random.randint(0, 30))
            
            cv2.circle(img, (x, y), radius, color, -1)
            # Add slight glow
            cv2.circle(img, (x, y), radius + 1, color, 1)

def add_microaneurysms(img, lipid_values):
    """Add microaneurysms (small red dots) - more common with poor lipid profile."""
    # Poor lipid profile -> more microaneurysms
    risk_score = (lipid_values['ldl'] / 200 + lipid_values['triglycerides'] / 250) / 2
    num_microaneurysms = int(risk_score * 8)
    
    for _ in range(num_microaneurysms):
        x = np.random.randint(20, config.IMAGE_SIZE - 20)
        y = np.random.randint(20, config.IMAGE_SIZE - 20)
        cv2.circle(img, (x, y), 1, (60, 20, 15), -1)

def create_retinal_image(seed, lipid_values):
    """Create a realistic synthetic retinal fundus image with features correlated to lipid values."""
    np.random.seed(seed)
    
    # Create base image with realistic retinal background
    img = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    center = (config.IMAGE_SIZE // 2, config.IMAGE_SIZE // 2)
    
    # Realistic retinal background (reddish-orange gradient)
    # HDL affects overall retinal health and brightness
    health_factor = (lipid_values['hdl'] - 30) / 60  # 0 to 1
    base_brightness = int(140 + health_factor * 30)
    
    # Create radial gradient for realistic appearance
    y, x = np.ogrid[:config.IMAGE_SIZE, :config.IMAGE_SIZE]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    max_dist = config.IMAGE_SIZE / 2
    
    # Vignetting effect (darker at edges)
    vignette = 1 - (dist_from_center / max_dist) * 0.4
    vignette = np.clip(vignette, 0, 1)
    
    # Apply realistic color with vignetting
    img[:, :, 0] = (base_brightness * 0.5 * vignette).astype(np.uint8)  # Blue
    img[:, :, 1] = (base_brightness * 0.65 * vignette).astype(np.uint8)  # Green
    img[:, :, 2] = (base_brightness * 0.95 * vignette).astype(np.uint8)  # Red
    
    # Add subtle texture to retina
    texture = np.random.normal(0, 3, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    img = np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    
    # Optic disc (optic nerve head) - realistic appearance
    disc_x = center[0] + np.random.randint(-25, 25)
    disc_y = center[1] + np.random.randint(-25, 25)
    disc_radius = np.random.randint(18, 24)
    
    # Optic disc with realistic layers
    # Outer rim
    cv2.circle(img, (disc_x, disc_y), disc_radius, (200, 180, 150), -1)
    # Middle layer
    cv2.circle(img, (disc_x, disc_y), int(disc_radius * 0.7), (230, 210, 180), -1)
    # Central cup
    cv2.circle(img, (disc_x, disc_y), int(disc_radius * 0.4), (250, 235, 210), -1)
    
    # Add slight blur to optic disc for realism
    mask = np.zeros_like(img)
    cv2.circle(mask, (disc_x, disc_y), disc_radius + 5, (255, 255, 255), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Macula (darker region temporal to optic disc)
    macula_x = disc_x + np.random.randint(35, 50)
    macula_y = disc_y + np.random.randint(-10, 10)
    macula_radius = 25
    
    # Create subtle darker region for macula
    overlay = img.copy()
    cv2.circle(overlay, (macula_x, macula_y), macula_radius, 
              (int(base_brightness * 0.4), int(base_brightness * 0.5), int(base_brightness * 0.7)), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # Fovea (small bright spot in center of macula)
    cv2.circle(img, (macula_x, macula_y), 3, 
              (int(base_brightness * 0.6), int(base_brightness * 0.7), int(base_brightness * 0.9)), -1)
    
    # Create realistic blood vessel network
    img = create_realistic_vessels(img, (disc_x, disc_y), disc_radius, lipid_values)
    
    # Add pathological features based on lipid levels
    add_exudates_and_deposits(img, lipid_values)
    add_microaneurysms(img, lipid_values)
    
    # Add realistic lighting variations
    lighting_variation = np.random.normal(1.0, 0.05, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * lighting_variation, 0, 255).astype(np.uint8)
    
    # Apply final blur for photorealistic appearance
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # Add slight chromatic aberration at edges (camera lens effect)
    dist_mask = (dist_from_center / max_dist) > 0.8
    if np.any(dist_mask):
        img[:, :, 0] = np.where(dist_mask, np.clip(img[:, :, 0] * 0.95, 0, 255), img[:, :, 0]).astype(np.uint8)
    
    return img

def generate_lipid_values(seed):
    """Generate lipid values with realistic medical correlations."""
    np.random.seed(seed)
    
    # Generate with realistic medical correlations
    # Total cholesterol = LDL + HDL + (Triglycerides / 5)
    
    # Start with LDL (most variable)
    ldl = np.random.uniform(*config.LIPID_RANGES['ldl'])
    
    # HDL inversely correlated with LDL (high LDL often means low HDL)
    hdl_mean = 60 - (ldl - 125) / 150 * 15  # Inverse relationship
    hdl = np.clip(np.random.normal(hdl_mean, 10), *config.LIPID_RANGES['hdl'])
    
    # Triglycerides positively correlated with LDL
    trig_mean = 100 + (ldl - 125) / 150 * 80
    triglycerides = np.clip(np.random.normal(trig_mean, 30), *config.LIPID_RANGES['triglycerides'])
    
    # Total cholesterol follows medical formula
    total_chol = ldl + hdl + (triglycerides / 5)
    total_chol = np.clip(total_chol, *config.LIPID_RANGES['total_cholesterol'])
    
    return {
        'total_cholesterol': round(total_chol, 2),
        'ldl': round(ldl, 2),
        'hdl': round(hdl, 2),
        'triglycerides': round(triglycerides, 2)
    }

def generate_dataset():
    """Generate complete synthetic dataset."""
    print("Generating synthetic retinal fundus dataset...")
    
    # Create directories
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    
    data_records = []
    
    for i in tqdm(range(config.DATASET_SIZE), desc="Generating images"):
        # Generate lipid values
        lipid_values = generate_lipid_values(i)
        
        # Create image
        img = create_retinal_image(i, lipid_values)
        
        # Save image
        img_filename = f'retinal_{i:05d}.png'
        img_path = os.path.join(config.IMAGES_DIR, img_filename)
        cv2.imwrite(img_path, img)
        
        # Record data
        record = {'image': img_filename}
        record.update(lipid_values)
        data_records.append(record)
    
    # Save labels
    df = pd.DataFrame(data_records)
    df.to_csv(config.LABELS_FILE, index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"Total images: {len(df)}")
    print(f"Images saved to: {config.IMAGES_DIR}")
    print(f"Labels saved to: {config.LABELS_FILE}")
    print(f"\nLipid value statistics:")
    print(df.describe())

if __name__ == '__main__':
    generate_dataset()
