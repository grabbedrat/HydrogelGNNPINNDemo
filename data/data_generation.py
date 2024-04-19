import numpy as np

def generate_random_data(num_samples):
    # Generate random hydrogel properties
    hydrogel_properties = np.random.rand(num_samples, 3)
    
    # Generate random drug release profiles
    drug_release_profiles = np.random.rand(num_samples, 10)
    
    return hydrogel_properties, drug_release_profiles