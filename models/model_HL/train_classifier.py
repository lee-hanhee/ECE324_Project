def _normalize_instrument_classes(self, classes):
    """
    Normalize instrument class names and combine similar classes.
    
    Args:
        classes: List of instrument class names
        
    Returns:
        Normalized list of class names with duplicates combined
    """
    # Create a mapping for classes to combine
    class_mapping = {}
    normalized_classes = []
    
    # Handle specific merges
    sound_effects_found = False
    strings_found = False
    
    for cls in classes:
        # Normalize case
        norm_cls = cls.strip()
        
        # Handle Sound Effects variants
        if norm_cls.lower() == "sound effects":
            if not sound_effects_found:
                normalized_classes.append("Sound Effects")
                sound_effects_found = True
            class_mapping[cls] = "Sound Effects"
        
        # Handle Strings variants
        elif norm_cls.lower() == "strings" or norm_cls.lower() == "strings (continued)":
            if not strings_found:
                normalized_classes.append("Strings")
                strings_found = True
            class_mapping[cls] = "Strings"
        
        # Keep other classes as is
        elif norm_cls not in class_mapping:
            normalized_classes.append(norm_cls)
            class_mapping[cls] = norm_cls
    
    print(f"Original classes: {len(classes)}, Normalized classes: {len(normalized_classes)}")
    print(f"Normalized instrument classes: {normalized_classes}")
    
    return normalized_classes 